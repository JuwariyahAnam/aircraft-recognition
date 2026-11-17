"""Step 5a: generate training masks from bounding boxes with GrabCut.

The image datasets only give boxes, so we let GrabCut separate the aircraft
from the sky inside each box. The masks are approximate (stated as a
limitation) but good enough to train a U-Net, which then segments images and
video frames that have no box at all.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from .utils import read_rgb


def grabcut_mask(img: np.ndarray, box, iters: int = 5) -> np.ndarray:
    """Return a uint8 mask (0 / 255) of the object inside box=(x1, y1, x2, y2)."""
    h, w = img.shape[:2]
    if box is None:   # no box: assume the aircraft fills the middle of the frame
        box = (int(0.05 * w), int(0.05 * h), int(0.95 * w), int(0.95 * h))
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    rect = (x1, y1, max(2, x2 - x1), max(2, y2 - y1))
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
        out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        out = np.zeros((h, w), np.uint8)
    # Clean up: keep the largest blob, fill small holes.
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(out)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.where(lab == keep, 255, 0).astype(np.uint8)
    # If GrabCut failed badly (almost nothing, or the whole box), fall back to an ellipse.
    box_area = rect[2] * rect[3]
    frac = out.sum() / 255 / max(1, box_area)
    if frac < 0.05 or frac > 0.98:
        out = np.zeros((h, w), np.uint8)
        cv2.ellipse(out, ((x1 + x2) // 2, (y1 + y2) // 2), (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 3)),
                    0, 0, 360, 255, -1)
    return out


def generate_masks(df, out_dir, only_with_box: bool = True):
    """Create a GrabCut mask for every image row; returns df with 'mask_path'."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    if "mask_path" not in df:
        df["mask_path"] = ""
    df["mask_path"] = df["mask_path"].fillna("").astype(str)
    rows = df.index[df["modality"] == "image"]
    for i in tqdm(rows, desc="GrabCut masks"):
        r = df.loc[i]
        has_box = not np.isnan(r["x1"])
        if only_with_box and not has_box:
            continue
        box = (r["x1"], r["y1"], r["x2"], r["y2"]) if has_box else None
        m = grabcut_mask(read_rgb(r["path"]), box)
        p = out_dir / f"{r['id']}.png"
        cv2.imwrite(str(p), m)
        df.at[i, "mask_path"] = str(p)
    return df
