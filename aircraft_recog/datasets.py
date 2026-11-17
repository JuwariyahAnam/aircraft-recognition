"""PyTorch datasets, augmentation (data synthesis) and mask application."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import read_gray, read_rgb

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BG_COLOR = (IMAGENET_MEAN * 255).astype(np.uint8)   # becomes 0 after normalisation


# ------------------------------------------------------------- augmentation
def image_augmenter():
    """Flip, small rotation/scale/shift, brightness and contrast (step 3)."""
    import albumentations as A
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), scale=(0.9, 1.1), translate_percent=(-0.06, 0.06), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    ])


def spec_augment(img: np.ndarray, rng: np.random.RandomState, n_time=2, n_freq=2,
                 max_t=0.12, max_f=0.12) -> np.ndarray:
    """SpecAugment: blank out random time and frequency bands (no flips for audio)."""
    img = img.copy()
    h, w = img.shape[:2]
    fill = int(img.mean())
    for _ in range(n_freq):
        f = rng.randint(0, max(1, int(h * max_f)))
        f0 = rng.randint(0, max(1, h - f))
        img[f0:f0 + f, :] = fill
    for _ in range(n_time):
        t = rng.randint(0, max(1, int(w * max_t)))
        t0 = rng.randint(0, max(1, w - t))
        img[:, t0:t0 + t] = fill
    return img


# --------------------------------------------------------------- masking
def apply_mask(img: np.ndarray, mask: np.ndarray | None, mode: str = "mask_crop",
               min_area: float = 0.005, margin: float = 0.12) -> np.ndarray:
    """Keep only the aircraft (step 5, last bullet).

    mode "none": unchanged; "mask": background painted neutral grey;
    "mask_crop": same, then crop to the aircraft with a margin.
    If the mask is (nearly) empty the original image is returned unchanged.
    """
    if mask is None or mode == "none":
        return img
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = mask > 127
    if m.mean() < min_area:
        return img
    out = img.copy()
    out[~m] = BG_COLOR
    if mode == "mask_crop":
        ys, xs = np.where(m)
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        h, w = img.shape[:2]
        side = int(max(y2 - y1, x2 - x1) * (1 + 2 * margin)) + 1
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        side = min(side, max(h, w))
        y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
        y3, x3 = min(h, y0 + side), min(w, x0 + side)
        out = out[y0:y3, x0:x3]
    return out


def to_tensor(img: np.ndarray, size: int) -> torch.Tensor:
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def load_spectrogram_rgb(path) -> np.ndarray:
    g = read_gray(path)
    return np.stack([g] * 3, axis=-1)


# --------------------------------------------------------------- datasets
class AircraftDataset(Dataset):
    """Classification dataset for one model.

    df          rows of the manifest (images + video frames, or audio)
    label_col   column holding the integer label to train on
    train       whether to augment
    mask_mode   how to use the U-Net mask for images/frames
    mask_col    which mask column to read ("pred_mask_path" or "mask_path")
    """

    def __init__(self, df, size=224, train=False, label_col="label_idx",
                 mask_mode="none", mask_col="pred_mask_path", seed=0):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.train = train
        self.label_col = label_col
        self.mask_mode = mask_mode
        self.mask_col = mask_col
        self.aug = image_augmenter() if train else None
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.df)

    def load(self, i) -> np.ndarray:
        r = self.df.iloc[i]
        if r["modality"] == "audio":
            img = load_spectrogram_rgb(r["path"])
            if self.train:
                img = spec_augment(img, self.rng)
            return img
        img = read_rgb(r["path"])
        if self.mask_mode != "none":
            mp = r.get(self.mask_col, "")
            if isinstance(mp, str) and mp:
                img = apply_mask(img, read_gray(mp), self.mask_mode)
        if self.train:
            img = self.aug(image=img)["image"]
        return img

    def __getitem__(self, i):
        x = to_tensor(self.load(i), self.size)
        y = int(self.df.iloc[i][self.label_col])
        return x, y


class SegDataset(Dataset):
    """Image + mask pairs for U-Net (same flip/rotation applied to both)."""

    def __init__(self, df, size=256, train=False):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.train = train
        if train:
            import albumentations as A
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(rotate=(-10, 10), scale=(0.9, 1.1), p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = read_rgb(r["path"])
        mask = read_gray(r["mask_path"])
        if self.train:
            o = self.aug(image=img, mask=mask)
            img, mask = o["image"], o["mask"]
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())
        y = torch.from_numpy((mask > 127).astype(np.float32))[None]
        return x, y


def segment_input(img: np.ndarray, size: int) -> torch.Tensor:
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def denormalize(x: torch.Tensor) -> np.ndarray:
    """Tensor (3,H,W) -> RGB float image in [0,1] (for Grad-CAM overlays)."""
    img = x.detach().cpu().numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)
