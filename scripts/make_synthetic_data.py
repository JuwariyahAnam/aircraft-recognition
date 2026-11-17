"""Create a tiny FAKE dataset that mimics the real download layouts.

Purpose: check that every step of the pipeline runs (a smoke test) in a few
minutes, before spending GPU hours on the real data. The shapes and sounds
are cartoons - accuracy numbers on this data mean nothing.

    python scripts/make_synthetic_data.py --root ./smoke
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

CLASSES = ["passenger_jet", "fighter_jet", "helicopter", "propeller"]
FGVC_FAMILY = {"passenger_jet": ["Boeing 737", "A320"], "fighter_jet": ["F-16", "Tornado"],
               "propeller": ["ATR-72", "Cessna 172"]}
MIL_CLASS = {"fighter_jet": ["F16", "Su57"], "helicopter": ["AH64", "Mi24"]}


def sky(h, w, rng):
    top, bot = rng.randint(120, 200, 3), rng.randint(180, 255, 3)
    g = np.linspace(0, 1, h)[:, None, None]
    img = (top * (1 - g) + bot * g) * np.ones((h, w, 3))
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_aircraft(img, cls, cx, cy, s, rng, angle=0):
    """Cartoon silhouettes; returns the bounding box."""
    c = tuple(int(v) for v in rng.randint(20, 90, 3))
    layer = np.zeros(img.shape[:2], np.uint8)
    if cls == "passenger_jet":
        cv2.ellipse(layer, (cx, cy), (int(1.6 * s), int(0.18 * s)), 0, 0, 360, 255, -1)
        pts = np.array([[cx - 0.2 * s, cy], [cx + 0.4 * s, cy - 1.0 * s], [cx + 0.6 * s, cy - 1.0 * s],
                        [cx + 0.3 * s, cy], [cx + 0.6 * s, cy + 1.0 * s], [cx + 0.4 * s, cy + 1.0 * s]], np.int32)
        cv2.fillPoly(layer, [pts], 255)
        cv2.fillPoly(layer, [np.array([[cx - 1.5 * s, cy], [cx - 1.6 * s, cy - 0.5 * s], [cx - 1.3 * s, cy]], np.int32)], 255)
    elif cls == "fighter_jet":
        pts = np.array([[cx + 1.2 * s, cy], [cx - 0.8 * s, cy - 0.9 * s], [cx - 0.6 * s, cy],
                        [cx - 0.8 * s, cy + 0.9 * s]], np.int32)
        cv2.fillPoly(layer, [pts], 255)
        cv2.ellipse(layer, (cx, cy), (int(1.2 * s), int(0.12 * s)), 0, 0, 360, 255, -1)
    elif cls == "helicopter":
        cv2.ellipse(layer, (cx, cy), (int(0.7 * s), int(0.45 * s)), 0, 0, 360, 255, -1)
        cv2.line(layer, (cx - int(0.5 * s), cy), (cx - int(1.6 * s), cy - int(0.2 * s)), 255, max(2, s // 8))
        cv2.line(layer, (cx - int(1.5 * s), cy - int(0.6 * s)), (cx + int(1.5 * s), cy - int(0.6 * s)), 255, max(2, s // 12))
        cv2.line(layer, (cx, cy - int(0.45 * s)), (cx, cy - int(0.6 * s)), 255, max(2, s // 10))
    else:  # propeller
        cv2.ellipse(layer, (cx, cy), (int(1.1 * s), int(0.16 * s)), 0, 0, 360, 255, -1)
        cv2.rectangle(layer, (cx - int(0.15 * s), cy - int(1.3 * s)), (cx + int(0.15 * s), cy + int(1.3 * s)), 255, -1)
        cv2.circle(layer, (cx + int(1.15 * s), cy), int(0.3 * s), 255, max(2, s // 12))
    if angle:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        layer = cv2.warpAffine(layer, M, (img.shape[1], img.shape[0]))
    img[layer > 0] = c
    ys, xs = np.where(layer > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def make_image(cls, rng, h=300, w=420):
    img = sky(h, w, rng)
    s = rng.randint(35, 60)
    cx, cy = rng.randint(int(2 * s), w - int(2 * s)), rng.randint(int(1.6 * s), h - int(1.6 * s))
    box = draw_aircraft(img, cls, cx, cy, s, rng, angle=rng.uniform(-20, 20))
    return img, box


def make_sound(cls, rng, sr=22050, dur=15.0):
    t = np.arange(int(sr * dur)) / sr
    noise = rng.normal(0, 1, len(t))
    if cls == "passenger_jet":      # broadband roar, high-passed
        y = noise - np.convolve(noise, np.ones(8) / 8, "same")
    elif cls == "propeller":        # strong harmonics of a blade-pass frequency
        f0 = rng.uniform(80, 120)
        y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8)) + 0.2 * noise
    else:                           # helicopter: low thump modulated at rotor rate
        y = np.convolve(noise, np.ones(40) / 40, "same") * (1 + np.sin(2 * np.pi * rng.uniform(12, 20) * t)) * 3
    y = y / (np.abs(y).max() + 1e-9) * 0.5
    return y.astype(np.float32)


def main(root, n_img=60, n_aud=8, n_vid=2):
    root = Path(root)
    raw = root / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    rng = np.random.RandomState(0)

    # ---- FGVC-like
    d = raw / "fgvc" / "fgvc-aircraft-2013b" / "data"
    (d / "images").mkdir(parents=True)
    boxes, splits = [], {"trainval": [], "test": []}
    k = 1000000
    for cls, fams in FGVC_FAMILY.items():
        for i in range(n_img):
            img, (x1, y1, x2, y2) = make_image(cls, rng)
            banner = np.full((20, img.shape[1], 3), 255, np.uint8)        # copyright strip
            cv2.imwrite(str(d / "images" / f"{k}.jpg"), np.vstack([img, banner])[..., ::-1])
            boxes.append(f"{k} {x1 + 1} {y1 + 1} {x2 + 1} {y2 + 1}")
            splits["trainval" if i % 3 else "test"].append(f"{k} {fams[i % 2]}")
            k += 1
    (d / "images_box.txt").write_text("\n".join(boxes) + "\n")
    for s, lines in splits.items():
        (d / f"images_family_{s}.txt").write_text("\n".join(lines) + "\n")

    # ---- Kaggle military-like (one csv per image)
    m = raw / "military" / "dataset"
    m.mkdir(parents=True)
    for cls, names in MIL_CLASS.items():
        for i in range(n_img):
            img, (x1, y1, x2, y2) = make_image(cls, rng)
            name = f"{cls}_{i:04d}"
            cv2.imwrite(str(m / f"{name}.jpg"), img[..., ::-1])
            (m / f"{name}.csv").write_text("filename,width,height,class,xmin,ymin,xmax,ymax\n"
                                           f"{name},{img.shape[1]},{img.shape[0]},{names[i % 2]},{x1},{y1},{x2},{y2}\n")

    # ---- AeroSonicDB-like
    a = raw / "aerosonicdb" / "audio"
    a.mkdir(parents=True)
    rows = ["train-test,fold,filename,class,subclass"]
    sub = {"propeller": [1, 2], "passenger_jet": [3], "helicopter": [4]}
    j = 0
    for cls, subs in sub.items():
        for i in range(n_aud):
            fn = f"{j:06d}_sample.wav"
            sf.write(a / fn, make_sound(cls, rng), 22050)
            rows.append(f"train,1,{fn},1,{subs[i % len(subs)]}")
            j += 1
    for i in range(3):   # background (subclass 0) is skipped by the loader
        fn = f"000000_bg{i}.wav"
        sf.write(a / fn, (rng.normal(0, 0.01, 22050 * 5)).astype(np.float32), 22050)
        rows.append(f"train,1,{fn},0,0")
    (raw / "aerosonicdb" / "sample_meta.csv").write_text("\n".join(rows) + "\n")

    # ---- self-collected-like video clips (aircraft flying across the sky)
    for cls in CLASSES:
        vd = raw / "video" / cls
        vd.mkdir(parents=True)
        for v in range(n_vid):
            h, w = 240, 320
            bg = sky(h, w, rng)
            out = vd / f"clip{v}.mp4"
            vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
            s = rng.randint(25, 40)
            y = rng.randint(int(1.6 * s), h - int(1.6 * s))
            for f in range(90):
                frame = bg.copy()
                draw_aircraft(frame, cls, int(2 * s + f * (w - 4 * s) / 90), y, s, np.random.RandomState(v))
                vw.write(frame[..., ::-1])
            vw.release()
    # one demo clip with a soundtrack, to exercise image+audio fusion
    demo = root / "demo_files"
    demo.mkdir(exist_ok=True)
    sf.write(demo / "prop_sound.wav", make_sound("propeller", rng, dur=6.0), 22050)
    img, _ = make_image("fighter_jet", rng)
    cv2.imwrite(str(demo / "fighter.jpg"), img[..., ::-1])
    shutil.copy(demo / "fighter.jpg", demo / "fighter_renamed.wav")    # wrong extension on purpose
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw / "video" / "propeller" / "clip0.mp4"),
                        "-i", str(demo / "prop_sound.wav"), "-shortest", "-c:v", "libx264", "-c:a", "aac",
                        str(demo / "prop_with_sound.mp4")], check=False)
    (demo / "notes.txt").write_text("not an aircraft file\n")
    print("synthetic data written to", raw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./smoke")
    ap.add_argument("--n_img", type=int, default=60)
    a = ap.parse_args()
    main(a.root, a.n_img)
