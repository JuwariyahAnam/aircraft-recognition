"""The build plan of section 11 as callable steps (used by the notebook and run_all.py).

Every step reads the manifest / models written by the previous step, so after
a Colab disconnect you can re-run from any step.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import experiments as ex
from .classifier import (default_hparams, evaluate, load_model, save_model, split_frames,
                         strip_probs, train_classifier)
from .config import CLASSES, PRETTY, Config
from .data_prep import build_dataset, download_all, load_manifest, save_manifest
from .utils import read_rgb, save_json, seed_everything, timer


# ------------------------------------------------------------- steps 1-4
def step1_download(cfg: Config, fgvc=True, military=True, aerosonic=True):
    download_all(cfg, fgvc, military, aerosonic)


def step3_build_dataset(cfg: Config, denoise=True):
    cfg.make_dirs(); cfg.save()
    return build_dataset(cfg, denoise_images=denoise)


# ---------------------------------------------------------------- step 5
def step5_segmentation(cfg: Config):
    from .masks import generate_masks
    from .segmentation import predict_all_masks, train_unet
    df = load_manifest(cfg)
    df = generate_masks(df, cfg.processed / "masks")
    save_manifest(df, cfg)
    model, metrics = train_unet(df, cfg)
    df = predict_all_masks(model, df, cfg)
    save_manifest(df, cfg)
    return metrics


# ---------------------------------------------------------------- step 6
def step6_baselines(cfg: Config):
    df = load_manifest(cfg)
    hp = default_hparams(cfg)
    print("\n### Baseline image CNN, no segmentation")
    m, r = ex.train_image_model(df, cfg, hp, mask_mode="none", name="baseline image")
    ex.record(cfg, "baseline_image", r)
    print("\n### Image CNN with U-Net segmentation")
    m, r = ex.train_image_model(df, cfg, hp, mask_mode=cfg.mask_mode, name="seg image")
    ex.record(cfg, "seg_image", r)
    ex.plot_training_curves(r["info"], "Image CNN (with segmentation)", cfg.results / "curves_image_seg.png")
    print("\n### Baseline audio CNN")
    m, r = ex.train_audio_model(df, cfg, hp, name="baseline audio")
    if r:
        ex.record(cfg, "baseline_audio", r)
        ex.plot_training_curves(r["info"], "Audio CNN (baseline)", cfg.results / "curves_audio.png")
    return ex.results_table(cfg)


# --------------------------------------------------------------- step 6b
def step6b_fine_grained(cfg: Config, rebuild: bool = False):
    """Train the 'which aircraft model' classifier (Boeing 737, F-16, ...).

    Needs the raw downloads the first time (it crops the original images);
    after that the processed crops in Drive are reused."""
    from .fine_grained import build_subtype_dataset, load_subtype_data, model_for
    if rebuild or not (cfg.processed / "subtype_manifest.csv").exists():
        build_subtype_dataset(cfg)
    df, names = load_subtype_data(cfg)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    hp = ex.get(cfg, "hp_image_pso") or default_hparams(cfg)
    print(f"\n### Fine-grained CNN: {len(names)} aircraft models, {len(tr)} training images")
    model, info = train_classifier(tr, va, cfg, hp, epochs=cfg.subtype_epochs, mask_mode="none",
                                   num_classes=len(names))
    m = strip_probs(evaluate(model, te, cfg, "none"))
    save_model(model, cfg.models / "subtype_cnn.pt", classes=[n["name"] for n in names],
               extra={"parents": [n["class"] for n in names], "size": cfg.cls_size})
    with3d = [n["name"] for n in names if model_for(n["name"])]
    res = {"n_models": len(names), "test": m, "info": info, "aircraft_models": names, "with_3d_model": with3d}
    save_json(res, cfg.results / "fine_grained_metrics.json")
    print(f"Fine-grained TEST: top-1 {m['accuracy']:.3f}, top-5 {m.get('top5_accuracy', float('nan')):.3f}, "
          f"macro-F1 {m['macro_f1']:.3f}")
    print("Aircraft models with a realistic 3D model:", ", ".join(with3d) or "none")
    return res


# ---------------------------------------------------------------- step 7
def step7_optimize(cfg: Config, methods=None, tune_audio=True):
    """Compare the optimizers on the image CNN; tune the audio CNN with PSO."""
    from .hpo import compare_optimizers
    from .optimizers import pso
    from .hpo import make_fitness
    df = load_manifest(cfg)
    lab, _, val, _ = split_frames(df, ex.IMG_MODS)
    with timer("optimizer comparison"):
        table, res = compare_optimizers(lab, val, cfg, methods, mask_mode=cfg.mask_mode)
    print(table.to_string())
    pso_hp = res.get("Particle Swarm Optimization", {}).get("best_hp") or default_hparams(cfg)
    best_row = table.iloc[1:].sort_values("Best validation accuracy", ascending=False).iloc[0] \
        if len(table) > 1 else table.iloc[0]
    ex.record(cfg, "hp_image_pso", pso_hp)
    ex.record(cfg, "hp_image_best_method", {"method": best_row["Method"], "hp": best_row["Best settings"]})
    print(f"\n### Retrain image CNN with PSO settings on full data: {pso_hp}")
    m, r = ex.train_image_model(df, cfg, pso_hp, mask_mode=cfg.mask_mode, name="PSO image")
    ex.record(cfg, "pso_image", r)
    save_model(m, cfg.models / "image_cnn_pso.pt")

    alab, _, aval, _ = split_frames(df, ex.AUD_MODS)
    if len(alab) and tune_audio:
        print("\n### PSO for the audio CNN")
        res_a = pso(make_fitness(alab, aval, cfg, mask_mode="none"), budget=cfg.hpo_budget,
                    pop=cfg.hpo_pop, seed=cfg.seed)
        ex.record(cfg, "hp_audio_pso", res_a.best_hp)
        save_json(res_a.to_dict(), cfg.results / "pso_audio.json")
        m, r = ex.train_audio_model(df, cfg, res_a.best_hp, name="PSO audio")
        ex.record(cfg, "pso_audio", r)
        save_model(m, cfg.models / "audio_cnn_pso.pt")
    return table


# ---------------------------------------------------------------- step 8
def step8_semi_supervised(cfg: Config, require_cluster_agreement=False):
    from .semi_supervised import cluster_analysis, extract_features, make_pseudo_labels, plot_tsne
    from .config import CLASS_TO_IDX
    df = load_manifest(cfg)
    report = {}
    for kind, mods, mask_mode in (("image", ex.IMG_MODS, cfg.mask_mode), ("audio", ex.AUD_MODS, "none")):
        ck = cfg.models / f"{kind}_cnn_pso.pt"
        lab, unl, val, test = split_frames(df, mods)
        if not ck.exists() or len(unl) == 0:
            continue
        print(f"\n### Semi-supervised learning: {kind} ({len(unl)} unlabelled samples)")
        model = load_model(ck)
        feats, probs = extract_features(model, unl, cfg, mask_mode)
        true_idx = unl["true_label"].map(CLASS_TO_IDX).values     # for reporting only
        clus_rep, clusters = cluster_analysis(feats, true_idx, cfg.n_clusters, cfg.seed)
        pseudo, all_unl, ps_rep = make_pseudo_labels(unl, probs, clusters, cfg.pseudo_threshold,
                                                     require_cluster_agreement)
        print("clustering:", clus_rep)
        print("pseudo-labels:", ps_rep)
        plot_tsne(feats, clusters, true_idx, cfg.results / f"tsne_{kind}.png",
                  title=f"t-SNE of {kind} CNN features (unlabelled set)")
        hp = ex.get(cfg, f"hp_{kind}_pso") or default_hparams(cfg)
        if kind == "image":
            m, r = ex.train_image_model(df, cfg, hp, mask_mode, extra_train=pseudo, name="pseudo image")
        else:
            m, r = ex.train_audio_model(df, cfg, hp, extra_train=pseudo, name="pseudo audio")
        r["clustering"], r["pseudo"] = clus_rep, ps_rep
        ex.record(cfg, f"pseudo_{kind}", r)
        save_model(m, cfg.models / f"{kind}_cnn_pseudo.pt")
        # The demo uses whichever of the two (PSO / PSO + pseudo-labels) did better on validation.
        before = ex.get(cfg, f"pso_{kind}")["info"]["best_val_acc"]
        after = r["info"]["best_val_acc"]
        final = m if after >= before else load_model(ck)
        save_model(final, cfg.models / f"{kind}_cnn.pt")
        report[kind] = {"val_before": before, "val_after": after, **ps_rep, "clustering": clus_rep,
                        "final_model": "pseudo" if after >= before else "pso"}
        all_unl.drop(columns=[c for c in all_unl.columns if c.startswith("x")], errors="ignore") \
            .to_csv(cfg.results / f"pseudo_labels_{kind}.csv", index=False)
    save_json(report, cfg.results / "semi_supervised.json")
    return report


# ---------------------------------------------------------------- step 9
def step9_explain(cfg: Config, n_each=4, lime_n=3):
    """Grad-CAM gallery for correct and wrong test predictions + LIME comparison."""
    from .datasets import AircraftDataset
    from .explain import gradcam_image, lime_image
    df = load_manifest(cfg)
    out = {}
    for kind, mods, mask_mode in (("image", ["image", "video"], cfg.mask_mode), ("audio", ["audio"], "none")):
        ck = cfg.models / f"{kind}_cnn.pt"
        if not ck.exists():
            continue
        model = load_model(ck)
        test = df[df.modality.isin(mods) & (df.split == "test")].reset_index(drop=True)
        if test.empty:
            continue
        ds = AircraftDataset(test, cfg.cls_size, False, mask_mode=mask_mode)
        m = evaluate(model, test, cfg, mask_mode)
        pred = m["_probs"].argmax(1)
        ok = np.where(pred == test.label_idx.values)[0]
        bad = np.where(pred != test.label_idx.values)[0]
        rng = np.random.RandomState(0)
        pick = [("correct", i) for i in rng.permutation(ok)[:n_each]] + \
               [("wrong", i) for i in rng.permutation(bad)[:n_each]]
        if not pick:
            continue
        fig, ax = plt.subplots(2, len(pick), figsize=(2.6 * len(pick), 5.4))
        ax = np.array(ax).reshape(2, -1)
        for k, (tag, i) in enumerate(pick):
            img = ds.load(i)
            vis, _, idx, probs = gradcam_image(model, img, cfg.cls_size)
            ax[0, k].imshow(cv2_resize(img, cfg.cls_size)); ax[0, k].axis("off")
            ax[0, k].set_title(f"{tag}\ntrue: {PRETTY[test.label[i]]}", fontsize=8)
            ax[1, k].imshow(vis); ax[1, k].axis("off")
            ax[1, k].set_title(f"pred: {PRETTY[CLASSES[idx]]} {probs[idx]:.0%}", fontsize=8)
        fig.suptitle(f"Grad-CAM on {kind} test samples (top: input, bottom: heatmap)")
        fig.tight_layout(); fig.savefig(cfg.results / f"gradcam_{kind}.png", dpi=110); plt.close(fig)
        ex.plot_confusion(m["confusion"], f"Final {kind} model - test set",
                          cfg.results / f"confusion_{kind}.png")
        out[kind] = strip_probs(m)
        if kind == "image" and lime_n:
            idxs = rng.permutation(len(test))[:lime_n]
            fig, ax = plt.subplots(len(idxs), 3, figsize=(8, 2.8 * len(idxs)))
            ax = np.array(ax).reshape(-1, 3)
            for row, i in enumerate(idxs):
                img = ds.load(i)
                vis, _, idx, _ = gradcam_image(model, img, cfg.cls_size)
                lv, _ = lime_image(model, img, cfg.cls_size, num_samples=200)
                for a, im, t in zip(ax[row], (cv2_resize(img, cfg.cls_size), vis, lv),
                                    ("input", f"Grad-CAM ({PRETTY[CLASSES[idx]]})", "LIME")):
                    a.imshow(im); a.set_title(t, fontsize=9); a.axis("off")
            fig.tight_layout(); fig.savefig(cfg.results / "gradcam_vs_lime.png", dpi=110); plt.close(fig)
    save_json(out, cfg.results / "final_test_metrics.json")
    return out


def cv2_resize(img, s):
    import cv2
    return cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------- extras
def extra_svm_baseline(cfg: Config):
    from .baseline_svm import run_svm
    df = load_manifest(cfg)
    rows = []
    for name, mods in (("Images + video frames (HOG)", ex.IMG_MODS), ("Audio (MFCC)", ex.AUD_MODS)):
        lab, _, _, test = split_frames(df, mods)
        m = run_svm(lab, test)
        if m:
            key = "image" if mods == ex.IMG_MODS else "audio"
            cnn = (ex.get(cfg, f"baseline_{key}") or {}).get("test", {})
            cnn_acc = np.mean([v["accuracy"] for v in cnn.values()]) if cnn else np.nan
            rows.append({"Data": name, "SVM accuracy": m["accuracy"], "SVM macro F1": m["macro_f1"],
                         "Baseline CNN accuracy": cnn_acc})
    t = pd.DataFrame(rows)
    t.to_csv(cfg.results / "svm_vs_cnn.csv", index=False)
    return t


def extra_synthesis(cfg: Config, sizes_per_class=(30, 75, 300, 750)):
    from .synthesis import run_synthesis_experiment
    df = load_manifest(cfg)
    lab, _, val, test = split_frames(df, ["image"])
    return run_synthesis_experiment(lab, val, test, cfg, 30, sizes_per_class,
                                    hp=ex.get(cfg, "hp_image_pso"), mask_mode=cfg.mask_mode)


def extra_ablation(cfg: Config):
    """Remove one part at a time. 'Without pseudo-labels' = the PSO model, already trained."""
    df = load_manifest(cfg)
    pseudo_csv = cfg.results / "pseudo_labels_image.csv"
    pseudo = pd.DataFrame()
    if pseudo_csv.exists():
        p = pd.read_csv(pseudo_csv)
        p = p[p.confidence >= cfg.pseudo_threshold]
        pseudo = df[df.id.isin(p.id)].merge(p[["id", "pseudo_label_idx"]], on="id")
        pseudo["label_idx"] = pseudo["pseudo_label_idx"]
    hp = ex.get(cfg, "hp_image_pso") or default_hparams(cfg)
    print("\n### Ablation: without segmentation")
    _, r = ex.train_image_model(df, cfg, hp, mask_mode="none", extra_train=pseudo, name="no-seg")
    ex.record(cfg, "ablate_noseg", r)
    print("\n### Ablation: without PSO")
    _, r = ex.train_image_model(df, cfg, default_hparams(cfg), mask_mode=cfg.mask_mode,
                                extra_train=pseudo, name="no-PSO")
    ex.record(cfg, "ablate_nopso", r)
    return ex.ablation_table(cfg)
