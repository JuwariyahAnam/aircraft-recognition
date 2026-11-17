"""Step 4: the router decides which path a file takes.

Two checks, as in "Smarter agent" (feature 5 of the guide):
1. the file's *content* (its first bytes, the "magic number"), and
2. its extension.
If the two disagree the content wins, because a renamed file still has its
real bytes. Files that are neither image, video nor audio are rejected.
"""
from __future__ import annotations

from pathlib import Path

IMAGE_EXT = {"jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"}
VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}
AUDIO_EXT = {"wav", "mp3", "flac", "ogg", "m4a", "aac"}


def route_by_extension(path) -> str | None:
    ext = str(path).lower().rsplit(".", 1)[-1]
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    return None


def route_by_content(path) -> str | None:
    """Look at the first bytes of the file."""
    with open(path, "rb") as f:
        h = f.read(64)
    if len(h) < 12:
        return None
    # images
    if h[:3] == b"\xff\xd8\xff":                 return "image"   # JPEG
    if h[:8] == b"\x89PNG\r\n\x1a\n":           return "image"   # PNG
    if h[:2] == b"BM":                           return "image"   # BMP
    if h[:4] == b"RIFF" and h[8:12] == b"WEBP":  return "image"   # WebP
    if h[:4] in (b"II*\x00", b"MM\x00*"):        return "image"   # TIFF
    # audio
    if h[:4] == b"RIFF" and h[8:12] == b"WAVE":  return "audio"   # WAV
    if h[:4] == b"fLaC":                         return "audio"   # FLAC
    if h[:3] == b"ID3":                          return "audio"   # MP3 with tag
    if h[0] == 0xFF and (h[1] & 0xE0) == 0xE0:   return "audio"   # MP3 frame / AAC
    if h[:4] == b"OggS":                         return "audio"   # Ogg (Vorbis/Opus)
    # video
    if h[:4] == b"RIFF" and h[8:12] == b"AVI ":  return "video"   # AVI
    if h[:4] == b"\x1a\x45\xdf\xa3":             return "video"   # MKV / WebM
    if h[4:8] == b"ftyp":                                          # MP4 / MOV / M4A
        brand = h[8:12]
        if brand in (b"M4A ", b"M4B ", b"M4P "):
            return "audio"
        return "video"
    return None


def route(path) -> str:
    """Return "image", "video" or "audio"; raise ValueError for anything else."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    by_content = route_by_content(path)
    by_ext = route_by_extension(path)
    kind = by_content or by_ext
    if kind is None:
        raise ValueError(f"Unsupported file type: {path.name}. "
                         "Please give an image, a video or an audio file.")
    return kind


def describe_route(path) -> dict:
    """Same as route() but also reports how the decision was made (for the demo)."""
    by_content = route_by_content(path)
    by_ext = route_by_extension(path)
    kind = by_content or by_ext
    note = "content and extension agree"
    if kind is None:
        note = "not an image, video or audio file: rejected"
    elif by_content and by_ext and by_content != by_ext:
        note = f"extension says {by_ext}, content says {by_content}: trusting content"
    elif by_content and not by_ext:
        note = "unknown extension, recognised from content"
    elif by_ext and not by_content:
        note = "recognised from extension only"
    return {"modality": kind, "by_content": by_content, "by_extension": by_ext, "note": note}
