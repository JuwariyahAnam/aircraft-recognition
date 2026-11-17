"""Step 9 + feature 8: explainable AI with Grad-CAM and LIME.

Grad-CAM (written out so each line matches the four steps in the guide):
  1. run the image through the CNN, take the score of the predicted class
  2. gradient of that score w.r.t. the last conv layer's feature maps
  3. average each gradient map -> one weight per feature map
  4. weighted sum of maps, keep positives (ReLU), resize to image size
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .datasets import IMAGENET_MEAN, IMAGENET_STD, to_tensor


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.layer = target_layer or model.target_layer()
        self.acts = self.grads = None
        self.h1 = self.layer.register_forward_hook(self._save_act)
        self.h2 = self.layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self.acts = o

    def _save_grad(self, m, gi, go):
        self.grads = go[0]

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: (1,3,H,W). Returns (heatmap HxW in [0,1], class_idx, probabilities)."""
        self.model.eval()
        # some layers are frozen; gradients must still flow to the feature maps
        x = x.clone().requires_grad_(True)
        logits = self.model(x)                                     # step 1
        probs = torch.softmax(logits, 1)[0].detach().cpu().numpy()
        if class_idx is None:
            class_idx = int(probs.argmax())
        self.model.zero_grad()
        logits[0, class_idx].backward()                            # step 2
        w = self.grads.mean(dim=(2, 3), keepdim=True)              # step 3
        cam = torch.relu((w * self.acts).sum(1))[0]                # step 4
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, probs

    def close(self):
        self.h1.remove(); self.h2.remove()


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Colour heatmap on top of an RGB uint8 image."""
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip((1 - alpha) * rgb + alpha * heat, 0, 255))


def gradcam_image(model, rgb: np.ndarray, size: int, class_idx=None):
    """Convenience: RGB uint8 image -> (overlay image, heatmap, class_idx, probs)."""
    device = next(model.parameters()).device
    x = to_tensor(rgb, size)[None].to(device)
    gc = GradCAM(model)
    try:
        cam, idx, probs = gc(x, class_idx)
    finally:
        gc.close()
    shown = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return overlay(shown, cam), cam, idx, probs


def lime_image(model, rgb: np.ndarray, size: int, num_samples: int = 300, num_features: int = 6):
    """LIME: hide random super-pixels, see how the prediction changes, and
    highlight the super-pixels that support the predicted class."""
    from lime import lime_image as li
    from skimage.segmentation import mark_boundaries
    device = next(model.parameters()).device
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)

    @torch.no_grad()
    def predict(batch):
        x = (batch.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(device)
        model.eval()
        out = []
        for i in range(0, len(x), 64):
            out.append(torch.softmax(model(x[i:i + 64]), 1).cpu().numpy())
        return np.concatenate(out)

    exp = li.LimeImageExplainer(random_state=0).explain_instance(
        img, predict, top_labels=1, hide_color=0, num_samples=num_samples)
    label = exp.top_labels[0]
    temp, mask = exp.get_image_and_mask(label, positive_only=True, num_features=num_features,
                                        hide_rest=False)
    vis = mark_boundaries(temp / 255.0, mask, color=(1, 1, 0))
    return np.uint8(vis * 255), int(label)
