"""Step 11: put the trained models on the project website.

Converts the trained networks to ONNX so the website (GitHub Pages) can run
recognition inside the visitor's browser with ONNX Runtime Web. No server needed.

  unet.onnx      U-Net segmentation
  image.onnx     image CNN: outputs class scores AND the last feature maps,
                 so the browser can draw the CAM heatmap (for a CNN that ends in
                 global average pooling + one linear layer, CAM equals Grad-CAM)
  audio.onnx     spectrogram CNN (same two outputs)
  subtype.onnx   fine-grained "which aircraft model" CNN
  manifest.json  class names, input sizes, thresholds, CAM weights, audio settings

Weights are stored as float16 (half the download) and converted back to float32
when the browser loads the model, so every browser can run it and the results
match the Python app.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import CLASSES, PRETTY, Config
from .datasets import IMAGENET_MEAN, IMAGENET_STD


class CamNet(nn.Module):
    """Wrap AircraftNet so it returns (logits, last conv feature maps)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        b = self.net.backbone
        if self.net.arch == "resnet18":
            x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
            fmap = b.layer4(b.layer3(b.layer2(b.layer1(x))))
        else:
            fmap = b.features(x)
        logits = self.net.head[-1](fmap.mean((2, 3)))
        return logits, fmap


class LogitsOnly(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)


def _export(module, size, path, outputs):
    module.eval()
    x = torch.randn(1, 3, size, size)
    kw = dict(input_names=["input"], output_names=outputs, opset_version=17, do_constant_folding=True)
    try:
        torch.onnx.export(module, x, str(path), dynamo=False, **kw)
    except TypeError:                     # older torch without the dynamo flag
        torch.onnx.export(module, x, str(path), **kw)
    return x


def halve_weights(path):
    """Store float32 weights as float16 + a Cast back to float32 (compute stays float32)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(str(path))
    g = m.graph
    new_inits, casts = [], []
    for init in list(g.initializer):
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= 1024:
            arr = numpy_helper.to_array(init).astype(np.float16)
            h = numpy_helper.from_array(arr, init.name + "__f16")
            new_inits.append(h)
            casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT,
                                          name=init.name + "__cast"))
            g.initializer.remove(init)
    g.initializer.extend(new_inits)
    for i, c in enumerate(casts):
        g.node.insert(i, c)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def _check(module, x, path):
    """The ONNX file must give the same answers as PyTorch."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})
    with torch.no_grad():
        want = module(x)
    want = want if isinstance(want, (tuple, list)) else [want]
    err = max(float(np.abs(g - w.numpy()).max()) for g, w in zip(got, want))
    return err


def export_web_models(cfg: Config, out_dir=None, fp16: bool = True) -> Path:
    from .classifier import load_model
    from .segmentation import load_unet
    out = Path(out_dir or (cfg.root / "web_models"))
    out.mkdir(parents=True, exist_ok=True)
    cpu = torch.device("cpu")
    m = cfg.models
    man = {
        "version": 1, "classes": CLASSES, "pretty": PRETTY,
        "seg_size": cfg.seg_size, "cls_size": cfg.cls_size, "mask_mode": cfg.mask_mode,
        "reject_threshold": cfg.reject_threshold, "fusion_audio_weight": cfg.fusion_audio_weight,
        "mean": IMAGENET_MEAN.tolist(), "std": IMAGENET_STD.tolist(),
        "audio": {"sr": cfg.audio_sr, "n_mels": cfg.n_mels, "window_s": cfg.audio_window_s,
                  "n_fft": 2048, "hop": 512, "top_db": 80.0},
        "models": {},
    }
    report = {}

    def done(name, module, x, path, extra=None):
        err = _check(module, x, path)
        if fp16:
            halve_weights(path)
            err = _check(module, x, path)          # still has to agree after halving the weights
        man["models"][name] = {"file": path.name, **(extra or {})}
        report[name] = {"max_abs_diff_vs_torch": err, "MB": round(path.stat().st_size / 1e6, 1)}
        print(f"[web] {name}: {path.name} {report[name]['MB']} MB (max diff vs PyTorch {err:.2e})")

    if (m / "unet.pt").exists():
        unet = load_unet(cfg, cpu)
        p = out / "unet.onnx"
        x = _export(unet, cfg.seg_size, p, ["logits"])
        done("unet", unet, x, p)
    for name, file, size in (("image", "image_cnn.pt", cfg.cls_size), ("audio", "audio_cnn.pt", cfg.cls_size)):
        if (m / file).exists():
            net = load_model(m / file, cpu)
            mod = CamNet(net).eval()
            p = out / f"{name}.onnx"
            x = _export(mod, size, p, ["logits", "features"])
            lin = net.head[-1]
            done(name, mod, x, p, {"cam_weights": lin.weight.detach().numpy().round(6).tolist()})
    if (m / "subtype_cnn.pt").exists():
        ck = torch.load(m / "subtype_cnn.pt", map_location="cpu", weights_only=False)
        net = load_model(m / "subtype_cnn.pt", cpu)
        mod = LogitsOnly(net).eval()
        p = out / "subtype.onnx"
        x = _export(mod, ck.get("size", cfg.cls_size), p, ["logits"])
        done("subtype", mod, x, p, {"names": ck["classes"], "parents": ck.get("parents", []),
                                    "size": ck.get("size", cfg.cls_size)})
    if not man["models"]:
        raise FileNotFoundError(f"No trained models in {m}. Train first (Steps 5-8).")
    with open(out / "manifest.json", "w") as f:
        json.dump(man, f)
    print(f"[web] wrote {out} ({sum(v['MB'] for v in report.values()):.0f} MB in total)")
    return out


def copy_into_repo(web_dir, repo_dir) -> Path:
    """Copy exported files into <repo>/web_models (the website loads them from there)."""
    dst = Path(repo_dir) / "web_models"
    dst.mkdir(parents=True, exist_ok=True)
    for f in Path(web_dir).iterdir():
        if f.suffix in (".onnx", ".json"):
            shutil.copy2(f, dst / f.name)
    return dst
