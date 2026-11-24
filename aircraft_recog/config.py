"""Central configuration: paths, class names and default training settings.

Every other module imports from here, so changing a path or a setting in one
place changes it everywhere. On Colab, set DATA_ROOT to a folder in Google
Drive so the data and models survive a runtime restart.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# The four target classes (section 1 of the guide). Order matters: it is the
# index the networks predict.
CLASSES = ["passenger_jet", "fighter_jet", "helicopter", "propeller"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

PRETTY = {
    "passenger_jet": "Passenger jet",
    "fighter_jet": "Fighter jet",
    "helicopter": "Helicopter",
    "propeller": "Propeller aircraft",
}


@dataclass
class Config:
    # ---------------------------------------------------------------- paths
    root: Path = Path(os.environ.get("AIRCRAFT_ROOT", "./workdir"))
    # Optional: keep the big raw downloads somewhere else (e.g. Colab's local
    # disk "/content/raw") while processed data and models live in Drive.
    raw_root: str = ""
    video_root: str = ""          # your clips: <video_root>/<class>/*.mp4

    # ------------------------------------------------------- dataset sizes
    # Suggested 5,000-sample mix (section 9).
    n_images: int = 3500          # balanced across the four classes
    n_audio: int = 1000           # 5-second spectrogram windows
    n_video_frames: int = 500     # about 30 frames per clip
    frames_every: int = 10        # keep every Nth video frame
    max_frames_per_clip: int = 30
    audio_window_s: float = 5.0
    audio_sr: int = 22050
    n_mels: int = 128

    # ---------------------------------------------------------------- splits
    train_frac: float = 0.70
    val_frac: float = 0.15        # test gets the rest (0.15)
    unlabelled_frac: float = 0.40  # fraction of TRAIN whose labels are hidden
    seed: int = 42

    # ---------------------------------------------------------- image sizes
    seg_size: int = 256           # U-Net input
    cls_size: int = 224           # CNN input

    # ---------------------------------------------------------------- U-Net
    unet_encoder: str = "resnet34"
    unet_epochs: int = 25
    unet_lr: float = 1e-4
    unet_batch: int = 16

    # ------------------------------------------------------------------ CNN
    cnn_arch: str = "resnet18"    # or "efficientnet_b0"
    cnn_epochs: int = 20
    cnn_lr: float = 1e-3
    cnn_batch: int = 32
    cnn_dropout: float = 0.3
    cnn_unfrozen: int = 2         # number of top backbone blocks trained
    early_stop_patience: int = 4
    mask_mode: str = "mask_crop"  # "none" | "mask" | "mask_crop"

    # ---------------------------------------------- hyper-parameter search
    hpo_budget: int = 30          # training runs per method (same for all)
    hpo_pop: int = 6              # particles / chromosomes / wolves
    hpo_epochs: int = 3           # short training run per evaluation
    hpo_subset: int = 1200        # training samples used during search

    # ------------------------------------------- fine-grained (which model)
    subtype_min_per_class: int = 25     # aircraft models with fewer images are left out
    subtype_max_per_class: int = 100
    n_subtype_max: int = 10000
    subtype_epochs: int = 20

    # ------------------------------------------------------ semi-supervised
    pseudo_threshold: float = 0.90
    n_clusters: int = 4

    # ---------------------------------------------------------------- agent
    reject_threshold: float = 0.55  # below this max-probability -> "not sure / not an aircraft"
    fusion_audio_weight: float = 0.3

    pretrained: bool = True       # set False only for offline smoke tests
    num_workers: int = 2

    # ------------------------------------------------------- derived paths
    @property
    def raw(self) -> Path: return Path(self.raw_root) if self.raw_root else self.root / "raw"
    @property
    def video(self) -> Path: return Path(self.video_root) if self.video_root else self.raw / "video"
    @property
    def processed(self) -> Path: return self.root / "processed"
    @property
    def models(self) -> Path: return self.root / "models"
    @property
    def results(self) -> Path: return self.root / "results"
    @property
    def manifest(self) -> Path: return self.processed / "manifest.csv"

    def make_dirs(self) -> "Config":
        for p in (self.raw, self.processed, self.models, self.results):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def to_dict(self):
        d = asdict(self)
        d["root"] = str(self.root)
        return d

    def save(self):
        import json
        self.models.mkdir(parents=True, exist_ok=True)
        with open(self.models / "config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, root) -> "Config":
        """Re-create the settings used for training (sizes must match the models)."""
        import json
        p = Path(root) / "models" / "config.json"
        if not p.exists():
            return cls(root=Path(root))
        d = json.load(open(p))
        d["root"] = Path(root)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def fast_config(root: str | Path) -> Config:
    """Tiny settings for a quick smoke test on CPU (minutes, not hours)."""
    return Config(
        root=Path(root),
        n_images=160, n_audio=60, n_video_frames=40,
        unet_epochs=2, unet_batch=8,
        cnn_epochs=2, cnn_batch=16,
        hpo_budget=4, hpo_pop=2, hpo_epochs=1, hpo_subset=64,
        subtype_min_per_class=10, subtype_max_per_class=40, subtype_epochs=2,
        seg_size=128, cls_size=112,
        pretrained=False, num_workers=0,
    )
