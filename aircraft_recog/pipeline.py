"""The agent: one call takes any file and returns an explained prediction.

    route -> convert -> segment -> classify -> explain

Includes video voting (feature 6), image+audio fusion for videos with sound
(feature 7) and a "not sure / not an aircraft" answer (feature 5).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import CLASSES, PRETTY, Config
from .convert import (extract_audio_from_video, file_to_spectrogram_images, sample_video_frames,
                      spectrogram_rgb)
from .datasets import apply_mask, to_tensor
from .explain import gradcam_image, lime_image, overlay
from .router import describe_route
from .utils import get_device, read_rgb


class AircraftAgent:
    def __init__(self, cfg: Config, device=None):
        import torch
        from .classifier import load_model
        from .segmentation import load_unet
        self.cfg = cfg
        self.device = device or get_device()
        m = cfg.models
        self.unet = load_unet(cfg, self.device) if (m / "unet.pt").exists() else None
        self.image_model = load_model(m / "image_cnn.pt", self.device) if (m / "image_cnn.pt").exists() else None
        self.audio_model = load_model(m / "audio_cnn.pt", self.device) if (m / "audio_cnn.pt").exists() else None
        if self.image_model is None and self.audio_model is None:
            raise FileNotFoundError(f"No trained models in {m}. Run the training steps first.")
        self.torch = torch

    # ------------------------------------------------------------ helpers
    def _probs(self, model, rgb):
        with self.torch.no_grad():
            x = to_tensor(rgb, self.cfg.cls_size)[None].to(self.device)
            return self.torch.softmax(model(x), 1)[0].cpu().numpy()

    def _segment(self, rgb):
        if self.unet is None:
            return None
        from .segmentation import predict_mask
        return predict_mask(self.unet, rgb, self.cfg.seg_size, self.device)

    def _verdict(self, probs, extra_note=""):
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        rejected = conf < self.cfg.reject_threshold
        label = PRETTY[CLASSES[idx]]
        text = (f"Not sure: possibly not an aircraft (best guess {label}, {conf:.0%})"
                if rejected else f"{label} ({conf:.0%} confident)")
        if extra_note:
            text += f". {extra_note}"
        return {"class": CLASSES[idx], "label": label, "confidence": conf, "rejected": rejected,
                "summary": text, "probabilities": {PRETTY[c]: float(p) for c, p in zip(CLASSES, probs)}}

    # ------------------------------------------------------------ per modality
    def analyse_image(self, rgb, explain=True, use_lime=False):
        mask = self._segment(rgb)
        mode = self.cfg.mask_mode if mask is not None else "none"
        cut = apply_mask(rgb, mask, mode)
        probs = self._probs(self.image_model, cut)
        out = {"input": rgb, "mask": mask, "cutout": cut, "probs": probs}
        note = ""
        if mask is not None and (mask > 127).mean() < 0.005:
            note = "No aircraft-shaped object was segmented"
        if explain:
            out["gradcam"], _, _, _ = gradcam_image(self.image_model, cut, self.cfg.cls_size)
        if use_lime:
            out["lime"], _ = lime_image(self.image_model, cut, self.cfg.cls_size)
        out["note"] = note
        return out

    def analyse_audio_file(self, path, explain=True):
        specs = file_to_spectrogram_images(path, self.cfg.audio_sr, self.cfg.audio_window_s, self.cfg.n_mels)
        if not specs:
            raise ValueError("Audio file is empty or silent.")
        rgbs = [spectrogram_rgb(s) for s in specs]
        all_p = np.array([self._probs(self.audio_model, r) for r in rgbs])
        probs = all_p.mean(0)
        best = int(all_p.max(1).argmax())
        shown = cv2.resize(rgbs[best], (self.cfg.cls_size * 2, self.cfg.cls_size))
        shown = cv2.applyColorMap(shown[..., 0], cv2.COLORMAP_MAGMA)[..., ::-1].copy()
        out = {"input": shown, "probs": probs, "n_windows": len(specs), "window_probs": all_p}
        if explain:
            _, cam, _, _ = gradcam_image(self.audio_model, rgbs[best], self.cfg.cls_size)
            out["gradcam"] = overlay(shown, cam)
        return out

    # ------------------------------------------------------------ main entry
    def predict(self, path, explain=True, use_lime=False, n_video_frames=16):
        path = Path(path)
        route = describe_route(path)
        kind = route["modality"]
        if kind is None:
            raise ValueError("Unsupported file: please give an image, a video or an audio file.")
        result = {"route": route, "modality": kind}

        if kind == "image":
            self._need(self.image_model, "image")
            a = self.analyse_image(read_rgb(path), explain, use_lime)
            result.update(a)
            result.update(self._verdict(a["probs"], a["note"]))

        elif kind == "audio":
            self._need(self.audio_model, "audio")
            a = self.analyse_audio_file(path, explain)
            result.update(a)
            result.update(self._verdict(a["probs"], f"Averaged over {a['n_windows']} five-second windows"))

        else:  # video
            self._need(self.image_model, "video")
            frames = sample_video_frames(path, n_video_frames)
            if not frames:
                raise ValueError("Could not read any frames from the video.")
            analyses = [self.analyse_image(f, explain=False) for f in frames]
            fp = np.array([a["probs"] for a in analyses])
            votes = np.bincount(fp.argmax(1), minlength=len(CLASSES))
            image_probs = fp.mean(0)
            best = int(fp.max(1).argmax())
            shown = self.analyse_image(frames[best], explain, use_lime)
            result.update(shown)
            note = f"Majority vote over {len(frames)} frames: " + ", ".join(
                f"{PRETTY[c]} {v}" for c, v in zip(CLASSES, votes) if v)
            probs = image_probs
            # feature 7: fuse with the soundtrack if there is one
            if self.audio_model is not None:
                wav = extract_audio_from_video(path, self.cfg.audio_sr)
                if wav is not None:
                    try:
                        au = self.analyse_audio_file(wav, explain=False)
                        w = self.cfg.fusion_audio_weight
                        probs = (1 - w) * image_probs + w * au["probs"]
                        result["audio_probs"] = au["probs"]
                        note += f". Fused with the soundtrack (audio weight {w})"
                    except ValueError:
                        pass
            result["image_probs"] = image_probs
            result["n_frames"] = len(frames)
            result["frame_probs"] = fp
            result["votes"] = votes
            result["probs"] = probs
            result.update(self._verdict(probs, note))
        return result

    @staticmethod
    def _need(model, what):
        if model is None:
            raise RuntimeError(f"No trained model for {what} input yet.")
