"""Runs the optimizer comparison (section 12 table) on the image classifier."""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .classifier import default_hparams, train_classifier
from .config import Config
from .optimizers import METHODS
from .utils import save_json


def _stratified_subset(df, n, seed):
    if len(df) <= n:
        return df
    frac = n / len(df)
    parts = [g.sample(max(1, int(round(len(g) * frac))), random_state=seed)
             for _, g in df.groupby("label_idx")]
    return pd.concat(parts)


def make_fitness(train_df, val_df, cfg: Config, mask_mode="mask_crop"):
    """fitness(hp) = validation accuracy after a short training run on a subset."""
    tr = _stratified_subset(train_df, cfg.hpo_subset, cfg.seed)
    va = _stratified_subset(val_df, max(200, cfg.hpo_subset // 3), cfg.seed)

    def fitness(hp):
        _, info = train_classifier(tr, va, cfg, hp, epochs=cfg.hpo_epochs, mask_mode=mask_mode,
                                   patience=cfg.hpo_epochs, verbose=False)
        return info["best_val_acc"]
    return fitness


def compare_optimizers(train_df, val_df, cfg: Config, methods=None, mask_mode="mask_crop"):
    """Same budget for every method. Returns (table DataFrame, results dict)."""
    fitness = make_fitness(train_df, val_df, cfg, mask_mode)
    rows, results = [], {}
    # Progress is saved after every method, so a Colab disconnect only loses the method in progress.
    ckpt_path = cfg.results / "optimizer_progress.json"
    key = {"budget": cfg.hpo_budget, "pop": cfg.hpo_pop, "epochs": cfg.hpo_epochs,
           "subset": cfg.hpo_subset, "seed": cfg.seed, "mask_mode": mask_mode}
    ckpt = {}
    if ckpt_path.exists():
        try:
            ckpt = json.loads(ckpt_path.read_text())
        except Exception:
            ckpt = {}
        if ckpt.get("key") != key:
            ckpt = {}
    ckpt["key"] = key
    ckpt.setdefault("methods", {})
    save = lambda: ckpt_path.write_text(json.dumps(ckpt, default=str))

    if "baseline" in ckpt:
        base, base_time = ckpt["baseline"], ckpt["baseline_time"]
        print(f"[resume] manual settings already done: {base:.3f}")
    else:
        t0 = time.time()
        base = fitness(default_hparams(cfg))
        base_time = round(time.time() - t0, 1)
        ckpt["baseline"], ckpt["baseline_time"] = base, base_time
        save()
    rows.append({"Method": "Manual settings (baseline)", "Best validation accuracy": base,
                 "Training runs used": 1, "Time taken (s)": base_time,
                 "Best settings": default_hparams(cfg)})
    for name in (methods or list(METHODS)):
        if name in ckpt["methods"]:
            d = ckpt["methods"][name]
            print(f"\n===== {name} ===== [resume] already done: {d['best_score']:.3f}")
        else:
            print(f"\n===== {name} =====")
            res = METHODS[name](fitness, budget=cfg.hpo_budget, pop=cfg.hpo_pop, seed=cfg.seed)
            d = res.to_dict()
            ckpt["methods"][name] = d
            save()
        results[name] = d
        rows.append({"Method": name, "Best validation accuracy": d["best_score"],
                     "Training runs used": d["runs_used"], "Time taken (s)": round(d["time_s"], 1),
                     "Best settings": d["best_hp"]})
    table = pd.DataFrame(rows)
    table.to_csv(cfg.results / "optimizer_comparison.csv", index=False)
    save_json({"baseline": base, "methods": results}, cfg.results / "optimizer_results.json")
    plot_convergence(results, base, cfg.results / "optimizer_convergence.png")
    return table, results


def plot_convergence(results, baseline, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, r in results.items():
        ax.plot(range(1, len(r["curve"]) + 1), r["curve"], marker="o", label=name)
    ax.axhline(baseline, ls="--", color="gray", label="Manual settings")
    ax.set_xlabel("Round (one round = one population of training runs)")
    ax.set_ylabel("Best validation accuracy so far")
    ax.set_title("Optimizer convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
