"""Step 6: CNN classifiers with transfer learning.

Two models with the same design:
  * image model  - photos and video frames (masked by the U-Net)
  * audio model  - mel-spectrograms
Backbone: ResNet-18 (or EfficientNet-B0) pretrained on ImageNet; the last
layer is replaced by Dropout + Linear(4).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import CLASSES, NUM_CLASSES, Config
from .datasets import AircraftDataset
from .utils import get_device, seed_everything


class AircraftNet(nn.Module):
    def __init__(self, arch="resnet18", num_classes=NUM_CLASSES, dropout=0.3,
                 unfrozen=2, pretrained=True):
        super().__init__()
        import torchvision.models as tvm
        self.arch = arch
        if arch == "resnet18":
            w = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = tvm.resnet18(weights=w)
            feat = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            blocks = [self.backbone.layer4, self.backbone.layer3, self.backbone.layer2, self.backbone.layer1]
        elif arch == "efficientnet_b0":
            w = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = tvm.efficientnet_b0(weights=w)
            feat = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            f = self.backbone.features   # 9 stages; group them into 4 "blocks" from the top
            blocks = [nn.ModuleList([f[8], f[7]]), nn.ModuleList([f[6], f[5]]),
                      nn.ModuleList([f[4], f[3]]), nn.ModuleList([f[2], f[1], f[0]])]
        else:
            raise ValueError(arch)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, num_classes))
        # Freeze everything, then unfreeze the top `unfrozen` blocks.
        # (Without pretrained weights a frozen layer would stay random, so then train everything.)
        for p in self.backbone.parameters():
            p.requires_grad = not pretrained
        for b in blocks[: int(np.clip(unfrozen, 0, 4))]:
            for p in b.parameters():
                p.requires_grad = True
        self.hparams = dict(arch=arch, dropout=dropout, unfrozen=int(unfrozen))

    def embed(self, x):
        """Feature vector from the layer before the final one (used for clustering)."""
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.backbone(x))

    def target_layer(self):
        """Last convolutional block, used by Grad-CAM."""
        if self.arch == "resnet18":
            return self.backbone.layer4[-1]
        return self.backbone.features[-1]


def default_hparams(cfg: Config) -> dict:
    return dict(lr=cfg.cnn_lr, dropout=cfg.cnn_dropout, batch=cfg.cnn_batch, unfrozen=cfg.cnn_unfrozen)


def make_loader(df, cfg, train, batch, label_col="label_idx", mask_mode="none", augment=None):
    ds = AircraftDataset(df, cfg.cls_size, train=train if augment is None else augment,
                         label_col=label_col, mask_mode=mask_mode, seed=cfg.seed)
    return DataLoader(ds, batch_size=int(batch), shuffle=train, num_workers=cfg.num_workers,
                      drop_last=train and len(df) > batch, pin_memory=torch.cuda.is_available())


def train_classifier(train_df, val_df, cfg: Config, hp: dict | None = None, epochs: int | None = None,
                     label_col="label_idx", mask_mode="none", patience: int | None = None,
                     verbose=True, seed: int | None = None, augment: bool | None = None):
    """Train one CNN. Returns (best model, info dict with history and best val accuracy)."""
    seed_everything(cfg.seed if seed is None else seed)
    hp = {**default_hparams(cfg), **(hp or {})}
    epochs = epochs or cfg.cnn_epochs
    patience = cfg.early_stop_patience if patience is None else patience
    device = get_device()
    model = AircraftNet(cfg.cnn_arch, NUM_CLASSES, hp["dropout"], hp["unfrozen"], cfg.pretrained).to(device)
    tl = make_loader(train_df, cfg, True, hp["batch"], label_col, mask_mode, augment)
    vl = make_loader(val_df, cfg, False, 64, "label_idx", mask_mode)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=hp["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_acc, best_state, history, bad = -1.0, None, [], 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        it = tqdm(tl, desc=f"epoch {ep + 1}/{epochs}", leave=False, disable=not verbose)
        for x, y in it:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(x); n += len(x)
        sched.step()
        val = evaluate_loader(model, vl, device)
        history.append({"epoch": ep + 1, "loss": tot / max(1, n), "val_acc": val["accuracy"],
                        "val_f1": val["macro_f1"]})
        if verbose:
            print(f"  epoch {ep + 1}: loss {tot / max(1, n):.3f}  val acc {val['accuracy']:.3f}  "
                  f"macro-F1 {val['macro_f1']:.3f}")
        if val["accuracy"] > best_acc:
            best_acc, best_state, bad = val["accuracy"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print("  early stopping")
                break
    model.load_state_dict(best_state)
    model.hparams.update(hp)
    return model, {"history": history, "best_val_acc": best_acc, "hparams": hp,
                   "train_time_s": time.time() - t0}


@torch.no_grad()
def predict_loader(model, loader, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        ys.append(y.numpy())
    if not probs:
        return np.zeros((0, NUM_CLASSES)), np.zeros(0, int)
    return np.concatenate(probs), np.concatenate(ys)


def metrics_from(y_true, probs):
    if len(y_true) == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "n": 0, "confusion": []}
    pred = probs.argmax(1)
    present = sorted(set(y_true.tolist()))   # audio has no fighter class
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=present, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, pred, labels=present, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, pred, labels=present, average="macro", zero_division=0)),
        "n": int(len(y_true)),
        "confusion": confusion_matrix(y_true, pred, labels=list(range(NUM_CLASSES))).tolist(),
    }


def evaluate_loader(model, loader, device=None):
    probs, y = predict_loader(model, loader, device)
    return metrics_from(y, probs)


def evaluate(model, df, cfg: Config, mask_mode="none", label_col="label_idx"):
    """Metrics on a dataframe. For video frames also reports per-clip majority vote."""
    if len(df) == 0:
        return metrics_from(np.zeros(0, int), np.zeros((0, NUM_CLASSES)))
    probs, y = predict_loader(model, make_loader(df, cfg, False, 64, label_col, mask_mode))
    out = metrics_from(y, probs)
    if (df["modality"] == "video").any():
        out["clip_vote"] = clip_vote_accuracy(df.reset_index(drop=True), probs)
    out["_probs"] = probs
    return out


def clip_vote_accuracy(df, probs):
    """Video voting (feature 6): classify every frame, take the majority per clip."""
    m = (df["modality"] == "video").values
    if not m.any():
        return None
    sub = df[m].copy()
    sub["pred"] = probs[m].argmax(1)
    correct = []
    for _, g in sub.groupby("group"):
        vote = np.bincount(g["pred"], minlength=NUM_CLASSES).argmax()
        correct.append(vote == g["label_idx"].iloc[0])
    return {"clip_accuracy": float(np.mean(correct)), "n_clips": len(correct)}


def save_model(model, path, extra: dict | None = None):
    torch.save({"state_dict": model.state_dict(), "hparams": model.hparams, "classes": CLASSES,
                **(extra or {})}, path)


def load_model(path, device=None):
    device = device or get_device()
    ck = torch.load(path, map_location=device, weights_only=False)
    hp = ck["hparams"]
    model = AircraftNet(hp.get("arch", "resnet18"), len(ck["classes"]), hp.get("dropout", 0.3),
                        hp.get("unfrozen", 2), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.hparams.update(hp)
    return model.to(device).eval()


def split_frames(df, modalities):
    d = df[df.modality.isin(modalities)]
    return (d[(d.split == "train") & d.labelled], d[(d.split == "train") & ~d.labelled],
            d[d.split == "val"], d[d.split == "test"])


def strip_probs(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}
