"""Step 2: turn every input into a picture.

* video  -> still frames (every Nth frame)
* audio  -> mel-spectrogram (a picture of frequency over time)
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------ video
def video_to_frames(path, every: int = 10, max_frames: int | None = None):
    """Return a list of RGB frames, keeping one frame in every `every`."""
    cap = cv2.VideoCapture(str(path))
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames


def sample_video_frames(path, n: int = 16):
    """Return up to n frames spread evenly across the whole clip (used at inference)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return video_to_frames(path, every=5, max_frames=n)
    idxs = np.linspace(0, total - 1, num=min(n, total)).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def extract_audio_from_video(path, sr: int = 22050):
    """Pull the soundtrack out of a video with ffmpeg. Returns a wav path or None."""
    if shutil.which("ffmpeg") is None:
        return None
    out = Path(tempfile.mkdtemp()) / "track.wav"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(sr), str(out)]
    try:
        subprocess.run(cmd, check=True, timeout=120)
    except Exception:
        return None
    return out if out.exists() and out.stat().st_size > 1000 else None


# ------------------------------------------------------------------ audio
def load_audio(path, sr: int = 22050, offset: float = 0.0, duration: float | None = None):
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True, offset=offset, duration=duration)
    return y


def audio_to_spectrogram(y: np.ndarray, sr: int = 22050, n_mels: int = 128) -> np.ndarray:
    """Mel-spectrogram in decibels, shape (n_mels, time)."""
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    return librosa.power_to_db(S, ref=np.max)


def audio_windows(path, sr: int = 22050, window_s: float = 5.0, hop_s: float | None = None,
                  min_rms: float = 1e-4):
    """Cut a long recording into fixed windows (more samples from one file)."""
    y = load_audio(path, sr=sr)
    win = int(window_s * sr)
    hop = int((hop_s or window_s) * sr)
    if len(y) < win:                       # pad short clips
        y = np.pad(y, (0, win - len(y)))
    out = []
    for start in range(0, len(y) - win + 1, hop):
        seg = y[start:start + win]
        if np.sqrt(np.mean(seg ** 2)) >= min_rms:   # skip digital silence
            out.append(seg)
    return out


def spectrogram_to_image(S_db: np.ndarray, top_db: float = 80.0) -> np.ndarray:
    """Scale a dB spectrogram to uint8 (0..255), low frequencies at the bottom."""
    S = np.clip(S_db, -top_db, 0.0)
    img = ((S + top_db) / top_db * 255.0).astype(np.uint8)
    return np.flipud(img).copy()


def spectrogram_rgb(img_u8: np.ndarray, size: int | None = None) -> np.ndarray:
    """Gray spectrogram image -> 3-channel RGB (optionally resized) for the CNN."""
    if size:
        img_u8 = cv2.resize(img_u8, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.stack([img_u8] * 3, axis=-1)


def file_to_spectrogram_images(path, sr=22050, window_s=5.0, n_mels=128):
    """Audio file -> list of uint8 spectrogram images, one per 5-second window."""
    return [spectrogram_to_image(audio_to_spectrogram(seg, sr, n_mels))
            for seg in audio_windows(path, sr, window_s)]


# ------------------------------------------------------------------ images
def resize_with_box(img: np.ndarray, size: int, box=None):
    """Resize to size x size and scale a (x1, y1, x2, y2) box to match."""
    h, w = img.shape[:2]
    out = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if box is None:
        return out, None
    x1, y1, x2, y2 = box
    sx, sy = size / w, size / h
    nb = (int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy)))
    nb = tuple(int(np.clip(v, 0, size - 1)) for v in nb)
    return out, nb


def denoise(img: np.ndarray) -> np.ndarray:
    """Light denoising that keeps edges (used during cleaning)."""
    return cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
