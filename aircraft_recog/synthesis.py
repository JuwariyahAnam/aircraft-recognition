"""Feature 3: small-to-large data synthesis experiment.

Start from 30 images per class, expand them offline with augmentation
(flip, rotation, scale, brightness/contrast) and train on growing sets.
The accuracy-vs-size plot demonstrates the "synthesize the data" requirement.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .classifier import evaluate, strip_probs, train_classifier
from .config import Config
from .datasets import apply_mask, image_augmenter
from .utils import read_gray, read_rgb, save_json, write_rgb


def synthesize(seed_df, per_class: int, out_dir, mask_mode="mask_crop", seed=0):
    """Return a dataframe with `per_class` samples per class: the originals plus augmented copies."""
    rng = np.random.RandomState(seed)
    aug = image_augmenter()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for lab, g in seed_df.groupby("label_idx"):
        g = g.reset_index(drop=True)
        for k in range(per_class):
            r = g.iloc[k % len(g)]
            img = read_rgb(r["path"])
            mp = r.get("pred_mask_path", "")
            if mask_mode != "none" and isinstance(mp, str) and mp:
                img = apply_mask(img, read_gray(mp), mask_mode)
            if k >= len(g):                      # originals first, then synthetic copies
                np.random.seed(rng.randint(1 << 30))
                img = aug(image=img)["image"]
            p = out_dir / f"{r['id']}_{k:04d}.png"
            write_rgb(p, img)
            rows.append({**r.to_dict(), "path": str(p), "pred_mask_path": "", "id": f"{r['id']}_{k}"})
    return pd.DataFrame(rows)


def run_synthesis_experiment(train_df, val_df, test_df, cfg: Config, seed_per_class=30,
                             sizes_per_class=(30, 75, 300, 750), hp=None, mask_mode="mask_crop"):
    rng = np.random.RandomState(cfg.seed)
    seed_df = pd.concat([g.sample(min(seed_per_class, len(g)), random_state=rng.randint(1 << 30))
                         for _, g in train_df.groupby("label_idx")])
    results = []
    for n in sizes_per_class:
        print(f"\n--- synthetic training set: {n} per class ---")
        d = synthesize(seed_df, n, cfg.processed / "synth" / f"n{n}", mask_mode, cfg.seed)
        model, info = train_classifier(d, val_df, cfg, hp, mask_mode=mask_mode, augment=False,
                                       verbose=False)
        # val/test rows still carry their U-Net masks, so evaluate with the same masking
        m = strip_probs(evaluate(model, test_df, cfg, mask_mode))
        results.append({"per_class": n, "total": len(d), "test_accuracy": m["accuracy"],
                        "test_macro_f1": m["macro_f1"]})
        print(results[-1])
    save_json(results, cfg.results / "synthesis_experiment.json")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["total"] for r in results], [r["test_accuracy"] for r in results], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel(f"Training samples (from {seed_per_class} real images per class)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Small-to-large data synthesis")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.results / "synthesis_experiment.png", dpi=120)
    plt.close(fig)
    return pd.DataFrame(results)
