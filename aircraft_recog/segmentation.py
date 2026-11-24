"""Step 5b: U-Net segmentation (ResNet-34 encoder pretrained on ImageNet).

Loss: Dice + binary cross-entropy. Metrics: IoU and Dice on the test set.
After training, the U-Net predicts a mask for every image and video frame;
the classifier then sees only the aircraft.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import Config
from .datasets import SegDataset, segment_input
from .utils import get_device, read_gray, read_rgb, save_json, seed_everything


def build_unet(cfg: Config):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=cfg.unet_encoder,
                    encoder_weights="imagenet" if cfg.pretrained else None,
                    in_channels=3, classes=1)


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        inter = (p * target).sum((1, 2, 3))
        dice = (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
        return self.bce(logits, target) + (1 - dice).mean()


def iou_dice(pred: torch.Tensor, target: torch.Tensor):
    """IoU = overlap / union ; Dice = 2*overlap / (pred + true). Per-image, then averaged."""
    pred, target = pred.float(), target.float()
    inter = (pred * target).sum((1, 2, 3))
    union = pred.sum((1, 2, 3)) + target.sum((1, 2, 3)) - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    dice = (2 * inter + 1e-6) / (pred.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1e-6)
    return iou, dice


@torch.no_grad()
def evaluate_unet(model, loader, device):
    model.eval()
    ious, dices = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = (torch.sigmoid(model(x)) > 0.5)
        i, d = iou_dice(pred, y)
        ious.append(i.cpu()); dices.append(d.cpu())
    if not ious:
        return {"iou": float("nan"), "dice": float("nan")}
    return {"iou": torch.cat(ious).mean().item(), "dice": torch.cat(dices).mean().item()}


def train_unet(df, cfg: Config):
    """Train on image rows that have a GrabCut mask. Returns (model, metrics)."""
    seed_everything(cfg.seed)
    device = get_device()
    has_mask = (df["modality"] == "image") & (df["mask_path"].astype(str) != "")
    d = df[has_mask]
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    # The U-Net may use all training images, labelled or not: masks come from boxes, not class labels.
    mk = lambda part, train: DataLoader(SegDataset(part, cfg.seg_size, train), batch_size=cfg.unet_batch,
                                        shuffle=train, num_workers=cfg.num_workers, drop_last=train and len(part) > cfg.unet_batch,
                                        persistent_workers=cfg.num_workers > 0)
    tl, vl, tel = mk(tr, True), mk(va, False), mk(te, False)
    model = build_unet(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.unet_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.unet_epochs))
    loss_fn = DiceBCELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, history = -1.0, []
    ckpt = cfg.models / "unet.pt"
    for ep in range(cfg.unet_epochs):
        model.train()
        tot = 0.0
        for x, y in tqdm(tl, desc=f"U-Net epoch {ep + 1}/{cfg.unet_epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(x)
        sched.step()
        m = evaluate_unet(model, vl, device)
        history.append({"epoch": ep + 1, "loss": tot / max(1, len(tr)), **m})
        print(f"epoch {ep + 1}: loss {history[-1]['loss']:.4f}  val IoU {m['iou']:.3f}  Dice {m['dice']:.3f}")
        if m["iou"] > best or np.isnan(m["iou"]):
            best = m["iou"]
            torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test = evaluate_unet(model, tel, device)
    print(f"U-Net TEST  IoU {test['iou']:.3f}  Dice {test['dice']:.3f}  (vs GrabCut masks)")
    metrics = {"test": test, "history": history}
    save_json(metrics, cfg.results / "unet_metrics.json")
    _plot_samples(model, te if len(te) else va, cfg, device)
    return model, metrics


def load_unet(cfg: Config, device=None):
    device = device or get_device()
    from dataclasses import replace
    model = build_unet(replace(cfg, pretrained=False)).to(device)
    model.load_state_dict(torch.load(cfg.models / "unet.pt", map_location=device))
    return model.eval()


@torch.no_grad()
def predict_mask(model, img: np.ndarray, size: int, device=None) -> np.ndarray:
    """RGB image (any size) -> uint8 mask (0/255) at the image's size."""
    device = device or next(model.parameters()).device
    x = segment_input(img, size)[None].to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    m = (prob > 0.5).astype(np.uint8) * 255
    return cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)


def predict_all_masks(model, df, cfg: Config):
    """Run the U-Net on every image and video frame; adds 'pred_mask_path'."""
    out = cfg.processed / "pred_masks"
    out.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    if "pred_mask_path" not in df:
        df["pred_mask_path"] = ""
    df["pred_mask_path"] = df["pred_mask_path"].fillna("").astype(str)
    device = next(model.parameters()).device
    for i in tqdm(df.index[df.modality.isin(["image", "video"])], desc="predict masks"):
        r = df.loc[i]
        m = predict_mask(model, read_rgb(r["path"]), cfg.seg_size, device)
        p = out / f"{r['id']}.png"
        cv2.imwrite(str(p), m)
        df.at[i, "pred_mask_path"] = str(p)
    return df


def _plot_samples(model, d, cfg, device, n=6):
    d = d.sample(min(n, len(d)), random_state=0)
    if d.empty:
        return
    fig, ax = plt.subplots(len(d), 3, figsize=(9, 3 * len(d)))
    ax = np.atleast_2d(ax)
    for k, (_, r) in enumerate(d.iterrows()):
        img = read_rgb(r["path"])
        ax[k, 0].imshow(img); ax[k, 0].set_title("image")
        ax[k, 1].imshow(read_gray(r["mask_path"]), cmap="gray"); ax[k, 1].set_title("GrabCut mask")
        ax[k, 2].imshow(predict_mask(model, img, cfg.seg_size, device), cmap="gray"); ax[k, 2].set_title("U-Net prediction")
        for a in ax[k]:
            a.axis("off")
    plt.tight_layout()
    fig.savefig(cfg.results / "unet_samples.png", dpi=110)
    plt.close(fig)
