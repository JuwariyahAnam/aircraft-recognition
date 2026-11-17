"""High-level experiment steps used by the Colab notebook and scripts/run_all.py.

Each function saves its outputs under cfg.results / cfg.models, so the
notebook can be resumed after a Colab disconnect.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .classifier import (default_hparams, evaluate, save_model, split_frames, strip_probs,
                         train_classifier)
from .config import CLASSES, PRETTY, Config
from .utils import load_json, save_json

IMG_MODS = ["image", "video"]
AUD_MODS = ["audio"]


# --------------------------------------------------------------- bookkeeping
def _store(cfg):
    p = cfg.results / "experiments.json"
    return load_json(p) if p.exists() else {}


def record(cfg, key, value):
    d = _store(cfg)
    d[key] = value
    save_json(d, cfg.results / "experiments.json")


def get(cfg, key, default=None):
    return _store(cfg).get(key, default)


def _eval_both(model, test_df, cfg, mask_mode):
    """Test metrics split by modality (image, video frame + clip vote, audio)."""
    out = {}
    for mod in ("image", "video", "audio"):
        d = test_df[test_df.modality == mod]
        if len(d):
            out[mod] = strip_probs(evaluate(model, d, cfg, mask_mode))
    return out


# --------------------------------------------------------------- experiments
def train_image_model(df, cfg, hp=None, mask_mode="mask_crop", extra_train=None, name="image",
                      verbose=True):
    lab, _, val, test = split_frames(df, IMG_MODS)
    train = lab if extra_train is None or len(extra_train) == 0 else pd.concat([lab, extra_train])
    print(f"[{name}] train {len(train)} (of which pseudo {0 if extra_train is None else len(extra_train)}),"
          f" val {len(val)}, test {len(test)}, mask={mask_mode}")
    model, info = train_classifier(train, val, cfg, hp, mask_mode=mask_mode, verbose=verbose)
    return model, {"info": info, "test": _eval_both(model, test, cfg, mask_mode)}


def train_audio_model(df, cfg, hp=None, extra_train=None, name="audio", verbose=True):
    lab, _, val, test = split_frames(df, AUD_MODS)
    if len(lab) == 0:
        print("[audio] no audio data - skipping")
        return None, None
    train = lab if extra_train is None or len(extra_train) == 0 else pd.concat([lab, extra_train])
    print(f"[{name}] train {len(train)}, val {len(val)}, test {len(test)}")
    model, info = train_classifier(train, val, cfg, hp, mask_mode="none", verbose=verbose)
    return model, {"info": info, "test": _eval_both(model, test, cfg, "none")}


def summarize(m: dict | None) -> dict:
    """Pull the three accuracy numbers for the results table."""
    if not m:
        return {"Image accuracy": np.nan, "Video accuracy": np.nan, "Audio accuracy": np.nan}
    t = m["test"]
    vid = t.get("video", {})
    vid_acc = vid.get("accuracy", np.nan)
    clip = (vid.get("clip_vote") or {}).get("clip_accuracy")
    return {"Image accuracy": t.get("image", {}).get("accuracy", np.nan),
            "Video accuracy": vid_acc,
            "Video clip-vote accuracy": clip if clip is not None else np.nan,
            "Audio accuracy": t.get("audio", {}).get("accuracy", np.nan)}


def results_table(cfg) -> pd.DataFrame:
    """Section 15 table: what each technique contributed."""
    rows = [("Baseline CNN, no segmentation", "baseline_image", "baseline_audio"),
            ("With U-Net segmentation", "seg_image", None),
            ("With PSO tuning", "pso_image", "pso_audio"),
            ("With pseudo-labelled data", "pseudo_image", "pseudo_audio")]
    out = []
    for name, ik, ak in rows:
        i = summarize(get(cfg, ik))
        a = summarize(get(cfg, ak)) if ak else {"Audio accuracy": "Not applicable"}
        out.append({"Experiment": name, "Image accuracy": i["Image accuracy"],
                    "Video accuracy": i["Video accuracy"],
                    "Video clip-vote accuracy": i.get("Video clip-vote accuracy", np.nan),
                    "Audio accuracy": a["Audio accuracy"]})
    t = pd.DataFrame(out)
    t.to_csv(cfg.results / "results_table.csv", index=False)
    return t


def ablation_table(cfg) -> pd.DataFrame:
    rows = [("Full system (segmentation + PSO + pseudo-labels)", "pseudo_image"),
            ("Without segmentation", "ablate_noseg"),
            ("Without PSO (manual settings)", "ablate_nopso"),
            ("Without pseudo-labels", "pso_image")]
    out = []
    for name, k in rows:
        s = summarize(get(cfg, k))
        out.append({"Configuration": name, "Image accuracy": s["Image accuracy"],
                    "Video accuracy": s["Video accuracy"],
                    "Macro F1 (image)": (get(cfg, k) or {}).get("test", {}).get("image", {}).get("macro_f1", np.nan)})
    t = pd.DataFrame(out)
    t.to_csv(cfg.results / "ablation_table.csv", index=False)
    return t


def plot_confusion(cm, title, path, labels=None):
    cm = np.array(cm)
    labels = labels or [PRETTY[c] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_training_curves(info, title, path):
    h = pd.DataFrame(info["history"])
    if h.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(h.epoch, h.loss, marker="o"); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(h.epoch, h.val_acc, marker="o", label="accuracy")
    ax[1].plot(h.epoch, h.val_f1, marker="o", label="macro F1")
    ax[1].set_title("Validation"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
