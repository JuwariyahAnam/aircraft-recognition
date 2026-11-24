"""Fine-grained recognition: which aircraft model is it (Boeing 737, F-16, ...)?

The main classifier answers "which of the four types". This second, harder
classifier answers "which specific aircraft model", using
  * FGVC-Aircraft's 70 manufacturer families (Boeing 737, A320, ATR-72, ...)
  * the Military Aircraft Detection Dataset's classes (F-16, Su-57, AH-64, ...)
Names that refer to the same aircraft are merged ("F-16" and "F16").

Training images are cropped to their bounding boxes so the network sees the
details that tell close models apart (engines, nose, tail, windows). At
inference the crop comes from the U-Net mask instead.

The prediction is hierarchical: only aircraft models belonging to the type
chosen by the main classifier are considered, so the two answers never
contradict each other. When the recognised model has a realistic 3D model in
static/models/registry.json, the demo shows that exact aircraft.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .class_map import norm
from .config import CLASSES, CLASS_TO_IDX, Config
from .utils import read_rgb, save_json, write_rgb

SPECIAL = {"MIG": "MiG", "SU": "Su", "KA": "Ka", "MI": "Mi", "TU": "Tu", "AN": "An", "IL": "Il", "YAK": "Yak"}


def pretty_name(raw: str) -> str:
    """'F16' -> 'F-16', 'Mig29' -> 'MiG-29', 'AH64' -> 'AH-64'; FGVC names are kept as they are."""
    raw = raw.strip()
    m = re.fullmatch(r"([A-Za-z]+)(\d+[A-Za-z]?)", raw)
    if not m:
        return raw
    letters, digits = m.groups()
    letters = SPECIAL.get(letters.upper(), letters.upper())
    return f"{letters}-{digits.upper()}"


# ----------------------------------------------------------------- dataset
def build_subtype_dataset(cfg: Config, records=None) -> pd.DataFrame:
    """Crop, resize and split the images of every aircraft model with enough examples."""
    from .data_prep import FGVC_BANNER_PX, scan_fgvc, scan_military
    rng = random.Random(cfg.seed)
    if records is None:
        records = scan_fgvc(cfg) + scan_military(cfg)
    groups = defaultdict(list)
    for r in records:
        if r.get("subtype"):
            groups[norm(r["subtype"])].append(r)
    out_dir = cfg.processed / "subtype"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in groups.items() if len(v) >= cfg.subtype_min_per_class}
    print(f"[fine-grained] {len(groups)} aircraft models found, {len(keep)} have at least "
          f"{cfg.subtype_min_per_class} images")
    names, rows = [], []
    for k in sorted(keep, key=lambda k: -len(keep[k])):
        recs = keep[k]
        rng.shuffle(recs)
        recs = recs[: cfg.subtype_max_per_class]
        fgvc = [r for r in recs if r["source"] == "fgvc"]
        display = fgvc[0]["subtype"] if fgvc else pretty_name(recs[0]["subtype"])
        parent = max(CLASSES, key=lambda c: sum(r["label"] == c for r in recs))
        names.append({"name": display, "class": parent, "n": len(recs)})
        idx = len(names) - 1
        n = len(recs)
        n_te, n_va = max(1, int(round(n * 0.1))), max(1, int(round(n * 0.1)))
        for j, r in enumerate(recs):
            split = "test" if j < n_te else "val" if j < n_te + n_va else "train"
            sid = f"sub_{idx:03d}_{j:04d}"
            p = out_dir / f"{sid}.jpg"
            if not p.exists():
                try:
                    img = read_rgb(r["path"])
                except Exception:
                    continue
                if r.get("crop_bottom"):
                    img = img[: img.shape[0] - FGVC_BANNER_PX]
                img = crop_box(img, r.get("box"))
                write_rgb(p, cv2.resize(img, (cfg.seg_size, cfg.seg_size), interpolation=cv2.INTER_AREA))
            rows.append(dict(id=sid, path=str(p), modality="image", subtype=display, label_idx=idx,
                             parent=parent, split=split, labelled=True, pred_mask_path=""))
        if len(rows) >= cfg.n_subtype_max:
            break
    df = pd.DataFrame(rows)
    df.to_csv(cfg.processed / "subtype_manifest.csv", index=False)
    save_json(names, cfg.processed / "subtype_classes.json")
    print(f"[fine-grained] {len(names)} aircraft models, {len(df)} images "
          f"({(df.split == 'train').sum()} train / {(df.split == 'val').sum()} val / {(df.split == 'test').sum()} test)")
    return df


def crop_box(img: np.ndarray, box, margin: float = 0.12) -> np.ndarray:
    """Square crop around a bounding box (whole image if there is no box)."""
    if box is None:
        return img
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    side = int(max(x2 - x1, y2 - y1) * (1 + 2 * margin))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    crop = img[y0:min(h, y0 + side), x0:min(w, x0 + side)]
    return crop if crop.size else img


def load_subtype_data(cfg: Config):
    df = pd.read_csv(cfg.processed / "subtype_manifest.csv")
    df["pred_mask_path"] = ""
    df["labelled"] = True
    names = json.load(open(cfg.processed / "subtype_classes.json"))
    return df, names


# ----------------------------------------------------------------- 3D model lookup
def model_registry() -> dict:
    p = Path(__file__).resolve().parent / "static" / "models" / "registry.json"
    if not p.exists():
        return {}
    reg = json.loads(p.read_text())
    return {k: v for k, v in reg.items() if (p.parent / v.get("file", "")).exists()}


def model_for(name: str | None, cls: str | None = None) -> str | None:
    """Registry key of the realistic 3D model for an aircraft name (or None)."""
    reg = model_registry()
    if name:
        n = norm(name)
        for key, entry in reg.items():
            if any(norm(m) == n or (len(norm(m)) >= 3 and norm(m) in n) for m in entry.get("matches", [])):
                return key
    return None


# ----------------------------------------------------------------- inference
class SubtypeRecogniser:
    """Wraps the fine-grained CNN: top aircraft models within the predicted type."""

    def __init__(self, path, device=None):
        import torch
        from .classifier import load_model
        self.model = load_model(path, device)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.names = ck["classes"]
        self.parents = np.array([CLASS_TO_IDX[p] for p in ck.get("parents", [CLASSES[0]] * len(self.names))])
        self.device = next(self.model.parameters()).device
        self.size = ck.get("size", 224)

    def predict(self, rgb: np.ndarray, cls: str | None = None, k: int = 3):
        import torch
        from .datasets import to_tensor
        with torch.no_grad():
            p = torch.softmax(self.model(to_tensor(rgb, self.size)[None].to(self.device)), 1)[0].cpu().numpy()
        if cls is not None and (self.parents == CLASS_TO_IDX[cls]).any():
            p = np.where(self.parents == CLASS_TO_IDX[cls], p, 0)
            p = p / max(p.sum(), 1e-9)
        top = np.argsort(-p)[:k]
        return [(self.names[i], float(p[i])) for i in top if p[i] > 0]
