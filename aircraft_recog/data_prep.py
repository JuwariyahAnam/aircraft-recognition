"""Steps 1-3: collect the data, convert it to pictures, clean it and split it.

Output: processed/manifest.csv, one row per sample, with these columns

    id, modality (image|video|audio), path, label, label_idx, source, group,
    x1, y1, x2, y2 (bounding box in processed-image pixels, images only),
    npy_path (raw dB spectrogram, audio only), split (train|val|test),
    labelled (False for the hidden 40% of train), true_label

`true_label` is kept for every row so you can measure how accurate the
pseudo-labels are, but no training code reads it for unlabelled rows.

Folder layout expected under cfg.raw (downloads go there automatically):

    raw/fgvc/fgvc-aircraft-2013b/...      FGVC-Aircraft (torchvision download)
    raw/military/...                       Kaggle Military Aircraft Detection Dataset
    raw/aerosonicdb/...                    AeroSonicDB zip(s) + sample_meta.csv
    raw/video/<class>/*.mp4                your own 15-20 clips, one folder per class
    raw/extra_images/<class>/*.jpg         optional extra photos, one folder per class
"""
from __future__ import annotations

import csv
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import CLASSES, CLASS_TO_IDX, Config
from . import class_map as cm
from .convert import (audio_to_spectrogram, audio_windows, denoise, resize_with_box,
                      spectrogram_to_image, video_to_frames)
from .utils import read_rgb, seed_everything, write_rgb

FGVC_BANNER_PX = 20   # FGVC images carry a 20-pixel copyright strip at the bottom


# ===================================================================== downloads
def download_fgvc(cfg: Config) -> Path:
    from torchvision.datasets import FGVCAircraft
    root = cfg.raw / "fgvc"
    for split in ("trainval", "test"):
        FGVCAircraft(root=str(root), split=split, annotation_level="family", download=True)
    return root


def download_military(cfg: Config) -> Path:
    """Needs a Kaggle account: kagglehub asks for your token the first time."""
    import kagglehub
    path = Path(kagglehub.dataset_download("a2015003713/militaryaircraftdetectiondataset"))
    dst = cfg.raw / "military"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            dst.symlink_to(path, target_is_directory=True)
        except OSError:
            # Windows without admin rights can't make symlinks: remember where kagglehub put it
            # instead of copying several GB a second time.
            (cfg.raw / "military_path.txt").write_text(str(path))
            return path
    return dst


def military_root(cfg: Config) -> Path:
    p = cfg.raw / "military"
    note = cfg.raw / "military_path.txt"
    if not p.exists() and note.exists():
        return Path(note.read_text().strip())
    return p


def download_aerosonic(cfg: Config, record: str = "8371595") -> Path:
    """Download every file of the Zenodo record and unzip the archives."""
    import requests
    dst = cfg.raw / "aerosonicdb"
    dst.mkdir(parents=True, exist_ok=True)
    meta = requests.get(f"https://zenodo.org/api/records/{record}", timeout=60).json()
    for f in meta["files"]:
        name = f["key"]
        url = f["links"]["self"]
        out = dst / name
        if not out.exists():
            print("downloading", name)
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(out, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
        if name.endswith(".zip") and not (dst / (name[:-4] + ".extracted")).exists():
            with zipfile.ZipFile(out) as z:
                z.extractall(dst)
            (dst / (name[:-4] + ".extracted")).touch()
    return dst


def download_all(cfg: Config, fgvc=True, military=True, aerosonic=True):
    cfg.make_dirs()
    if fgvc:
        download_fgvc(cfg)
    if military:
        download_military(cfg)
    if aerosonic:
        download_aerosonic(cfg)


# ================================================================ raw scanners
def scan_fgvc(cfg: Config) -> list[dict]:
    """Return image records {path, label, box, source, group} from FGVC-Aircraft."""
    hits = list((cfg.raw / "fgvc").rglob("images_box.txt"))
    if not hits:
        print("[fgvc] not found - skipping")
        return []
    data_dir = hits[0].parent
    boxes = {}
    for line in open(data_dir / "images_box.txt"):
        p = line.split()
        if len(p) == 5:
            boxes[p[0]] = tuple(int(v) - 1 for v in p[1:])   # 1-based -> 0-based
    recs, skipped = [], defaultdict(int)
    for split in ("trainval", "test"):
        f = data_dir / f"images_family_{split}.txt"
        if not f.exists():
            continue
        for line in open(f):
            img_id, family = line.strip().split(" ", 1)
            label = cm.map_fgvc(family)
            if label is None:
                skipped[family] += 1
                continue
            recs.append(dict(path=str(data_dir / "images" / f"{img_id}.jpg"), label=label, subtype=family.strip(),
                             box=boxes.get(img_id), source="fgvc", group=f"fgvc_{img_id}",
                             crop_bottom=FGVC_BANNER_PX))
    _report("fgvc", recs, skipped)
    return recs


def scan_military(cfg: Config) -> list[dict]:
    """Military Aircraft Detection Dataset: one CSV per image with boxes.

    Falls back to the dataset's 'crop/<CLASS>/' folders if no box CSVs exist.
    """
    root = military_root(cfg)
    if not root.exists():
        print("[military] not found - skipping")
        return []
    per_image = {}
    skipped = defaultdict(int)
    csvs = [p for p in root.rglob("*.csv") if "crop" not in p.parts]
    for c in tqdm(csvs, desc="military csv", leave=False):
        try:
            with open(c, newline="") as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            continue
        for r in rows:
            if not {"filename", "class", "xmin", "ymin", "xmax", "ymax"} <= set(r):
                break
            img = c.parent / r["filename"]
            if not img.suffix:
                img = img.with_suffix(".jpg")
            if not img.exists():
                continue
            label = cm.map_military(r["class"])
            if label is None:
                skipped[r["class"]] += 1
                continue
            box = tuple(int(float(r[k])) for k in ("xmin", "ymin", "xmax", "ymax"))
            area = (box[2] - box[0]) * (box[3] - box[1])
            best = per_image.get(str(img))
            if best is None or area > best["_area"]:     # keep the largest aircraft
                per_image[str(img)] = dict(path=str(img), label=label, box=box, subtype=r["class"].strip(),
                                           source="military", group=f"mil_{img.stem}",
                                           _area=area)
    recs = [{k: v for k, v in r.items() if k != "_area"} for r in per_image.values()]
    if not recs:   # fall back to pre-cropped folders
        for d in root.rglob("crop"):
            for cls_dir in d.iterdir():
                if not cls_dir.is_dir():
                    continue
                label = cm.map_military(cls_dir.name)
                files = [p for p in cls_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
                if label is None:
                    skipped[cls_dir.name] += len(files)
                    continue
                for p in files:
                    recs.append(dict(path=str(p), label=label, box=None, source="military_crop", subtype=cls_dir.name,
                                     group=f"milc_{cls_dir.name}_{p.stem}"))
    _report("military", recs, skipped)
    return recs


def scan_extra_images(cfg: Config) -> list[dict]:
    root = cfg.raw / "extra_images"
    recs = []
    if not root.exists():
        return recs
    for cls_dir in root.iterdir():
        label = cm.map_video_folder(cls_dir.name) if cls_dir.is_dir() else None
        if label is None:
            continue
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                recs.append(dict(path=str(p), label=label, box=None, source="extra",
                                 group=f"extra_{p.stem}"))
    _report("extra_images", recs, {})
    return recs


def scan_aerosonic(cfg: Config) -> list[dict]:
    """Return audio file records {path, label, group} from AeroSonicDB."""
    root = cfg.raw / "aerosonicdb"
    metas = list(root.rglob("sample_meta.csv")) if root.exists() else []
    if not metas:
        print("[aerosonic] not found - skipping")
        return []
    meta = pd.read_csv(metas[0])
    wavs = {p.name: p for p in root.rglob("*.wav")}
    recs, skipped = [], defaultdict(int)
    for _, r in meta.iterrows():
        name = str(r["filename"])
        if not name.endswith(".wav"):
            name += ".wav"
        p = wavs.get(name)
        label = cm.AEROSONIC_SUBCLASS.get(int(r["subclass"]))
        if label is None or p is None:
            skipped[f"subclass {r['subclass']}" if p is not None else "missing file"] += 1
            continue
        recs.append(dict(path=str(p), label=label, source="aerosonicdb",
                         group=f"aud_{Path(name).stem}"))
    _report("aerosonic", recs, skipped)
    return recs


def scan_videos(cfg: Config) -> list[dict]:
    root = cfg.video
    recs = []
    if not root.exists():
        print(f"[video] {root}/<class>/ not found - no video frames this run")
        return recs
    for cls_dir in sorted(root.iterdir()):
        label = cm.map_video_folder(cls_dir.name) if cls_dir.is_dir() else None
        if label is None:
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
                recs.append(dict(path=str(p), label=label, source="video",
                                 group=f"vid_{label}_{p.stem}"))
    _report("video clips", recs, {})
    return recs


def _report(name, recs, skipped):
    counts = defaultdict(int)
    for r in recs:
        counts[r["label"]] += 1
    print(f"[{name}] kept {len(recs)}: " + ", ".join(f"{c}={counts.get(c, 0)}" for c in CLASSES))
    if skipped:
        top = sorted(skipped.items(), key=lambda kv: -kv[1])[:15]
        print(f"[{name}] skipped (not in class map): " + ", ".join(f"{k}({v})" for k, v in top))


# ======================================================================= build
def _balanced_pick(recs, per_class_target, total_target, rng, key="label"):
    """Take up to per_class_target from each class, then top up to total_target."""
    by = defaultdict(list)
    for r in recs:
        by[r[key]].append(r)
    for v in by.values():
        rng.shuffle(v)
    chosen, rest = [], []
    for c in CLASSES:
        chosen += by[c][:per_class_target]
        rest += by[c][per_class_target:]
    if len(chosen) < total_target:
        rng.shuffle(rest)
        chosen += rest[: total_target - len(chosen)]
    return chosen


def _process_image(rec, out_path, size, do_denoise=True):
    img = read_rgb(rec["path"])
    crop = rec.get("crop_bottom", 0)
    if crop:
        img = img[: img.shape[0] - crop]
    box = rec.get("box")
    if box is not None:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        box = (max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2))
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            box = None
    if do_denoise:
        img = denoise(img)
    img, box = resize_with_box(img, size, box)
    write_rgb(out_path, img)
    return box


def build_dataset(cfg: Config, denoise_images: bool = True) -> pd.DataFrame:
    """Scan raw data, pick the 5,000-sample mix, convert, clean, split, save manifest."""
    seed_everything(cfg.seed)
    rng = random.Random(cfg.seed)
    cfg.make_dirs()
    out_img = cfg.processed / "images"
    out_frm = cfg.processed / "frames"
    out_spec = cfg.processed / "spectrograms"
    for d in (out_img, out_frm, out_spec):
        d.mkdir(parents=True, exist_ok=True)
    rows = []

    # ------------------------------------------------------------ images
    vid_recs = scan_videos(cfg)
    n_images = cfg.n_images
    if not vid_recs:   # no clips: take extra photos instead so the total stays about 5,000
        n_images += cfg.n_video_frames
        print(f"[video] no clips - using {n_images} images instead of {cfg.n_images}")
    img_recs = scan_fgvc(cfg) + scan_military(cfg) + scan_extra_images(cfg)
    picked = _balanced_pick(img_recs, n_images // len(CLASSES), n_images, rng)
    for i, r in enumerate(tqdm(picked, desc="images")):
        sid = f"img_{i:05d}"
        out = out_img / f"{sid}.png"
        try:
            box = _process_image(r, out, cfg.seg_size, denoise_images)
        except Exception as e:  # unreadable file
            print("skip", r["path"], e)
            continue
        rows.append(_row(sid, "image", out, r, box))

    # ------------------------------------------------------------- audio
    aud_recs = scan_aerosonic(cfg)
    if aud_recs:
        rows += _build_audio(cfg, aud_recs, out_spec, rng)

    # ------------------------------------------------------------- video
    if vid_recs:
        per_clip = max(1, min(cfg.max_frames_per_clip, cfg.n_video_frames // len(vid_recs)))
        n = 0
        for r in tqdm(vid_recs, desc="video"):
            frames = video_to_frames(r["path"], every=cfg.frames_every)
            if not frames:
                continue
            idx = np.linspace(0, len(frames) - 1, num=min(per_clip, len(frames))).astype(int)
            for j in idx:
                sid = f"frm_{n:05d}"
                out = out_frm / f"{sid}.png"
                img, _ = resize_with_box(frames[j], cfg.seg_size)
                write_rgb(out, img)
                rows.append(_row(sid, "video", out, r, None))
                n += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No data found. Run download_all() or check cfg.raw.")
    df = assign_splits(df, cfg)
    save_manifest(df, cfg)
    print_summary(df)
    return df


def _build_audio(cfg, aud_recs, out_spec, rng):
    """Cut recordings into 5 s windows, balanced across the classes present."""
    by = defaultdict(list)
    for r in aud_recs:
        by[r["label"]].append(r)
    present = [c for c in CLASSES if by[c]]
    quota = {c: cfg.n_audio // len(present) for c in present}
    for c in present:
        rng.shuffle(by[c])
    rows, n = [], 0
    pbar = tqdm(total=cfg.n_audio, desc="audio")
    # Round-robin over files of each class so one long recording cannot fill a class.
    for c in present:
        files = by[c]
        per_file = max(1, int(np.ceil(quota[c] / max(1, len(files)))) + 1)
        got = 0
        for r in files:
            if got >= quota[c]:
                break
            try:
                segs = audio_windows(r["path"], cfg.audio_sr, cfg.audio_window_s)
            except Exception as e:
                print("skip", r["path"], e)
                continue
            for seg in segs[:per_file]:
                if got >= quota[c]:
                    break
                sid = f"aud_{n:05d}"
                S = audio_to_spectrogram(seg, cfg.audio_sr, cfg.n_mels)
                png = out_spec / f"{sid}.png"
                npy = out_spec / f"{sid}.npy"
                cv2.imwrite(str(png), spectrogram_to_image(S))
                np.save(npy, S.astype(np.float16))
                row = _row(sid, "audio", png, r, None)
                row["npy_path"] = str(npy)
                rows.append(row)
                n += 1
                got += 1
                pbar.update(1)
    pbar.close()
    return rows


def _row(sid, modality, path, rec, box):
    x1, y1, x2, y2 = box if box is not None else (np.nan,) * 4
    return dict(id=sid, modality=modality, path=str(path), label=rec["label"],
                label_idx=CLASS_TO_IDX[rec["label"]], source=rec["source"],
                group=rec["group"], x1=x1, y1=y1, x2=x2, y2=y2, npy_path="",
                orig_path=rec["path"])


# ====================================================================== splits
def assign_splits(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """70/15/15 split by *group* (all frames of a clip or windows of a recording
    stay together), stratified by modality and class. Then hide the labels of
    `unlabelled_frac` of the training groups."""
    rng = np.random.RandomState(cfg.seed)
    df = df.copy()
    df["split"] = ""
    df["labelled"] = True
    df["true_label"] = df["label"]
    for (mod, lab), sub in df.groupby(["modality", "label"]):
        groups = np.array(sub["group"].unique(), dtype=object)
        rng.shuffle(groups)
        n = len(groups)
        n_tr = max(1, int(round(n * cfg.train_frac)))
        n_va = max(1 if n >= 3 else 0, int(round(n * cfg.val_frac)))
        if n_tr + n_va >= n and n >= 3:
            n_tr = n - n_va - 1
        tr, va, te = groups[:n_tr], groups[n_tr:n_tr + n_va], groups[n_tr + n_va:]
        df.loc[sub.index[sub["group"].isin(tr)], "split"] = "train"
        df.loc[sub.index[sub["group"].isin(va)], "split"] = "val"
        df.loc[sub.index[sub["group"].isin(te)], "split"] = "test"
        # hide labels on part of train
        n_hide = int(round(len(tr) * cfg.unlabelled_frac))
        hidden = rng.choice(tr, size=n_hide, replace=False) if n_hide else []
        df.loc[sub.index[sub["group"].isin(hidden)], "labelled"] = False
    return df


def print_summary(df: pd.DataFrame):
    print("\nSamples per modality and split:")
    print(pd.crosstab(df["modality"], df["split"], margins=True))
    print("\nSamples per class and modality:")
    print(pd.crosstab(df["label"], df["modality"], margins=True))
    tr = df[df.split == "train"]
    print(f"\nTraining set: {tr.labelled.sum()} labelled, {(~tr.labelled).sum()} unlabelled "
          f"({(~tr.labelled).mean():.0%} hidden). Total samples: {len(df)}")


PATH_COLS = ("path", "mask_path", "pred_mask_path", "npy_path")


def load_manifest(cfg: Config) -> pd.DataFrame:
    """Paths are stored relative to cfg.root, so the project folder can be moved."""
    df = pd.read_csv(cfg.manifest, keep_default_na=True)
    df["labelled"] = df["labelled"].astype(str).str.lower().isin(["true", "1"])
    for c in PATH_COLS:
        if c not in df:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).map(
            lambda p: str(cfg.root / p) if p and not Path(p).is_absolute() else p)
    return df


def save_manifest(df: pd.DataFrame, cfg: Config):
    out = df.copy()
    root = Path(cfg.root).resolve()
    def rel(p):
        if not p:
            return p
        try:
            return str(Path(p).resolve().relative_to(root))
        except ValueError:
            return p
    for c in PATH_COLS:
        if c in out:
            out[c] = out[c].fillna("").astype(str).map(rel)
    out.to_csv(cfg.manifest, index=False)
