"""Step 10 + feature 10: the demo interface.

A dark, cinematic interface: thin oversized type, chamfered glass panels,
HUD rings, and a 3D model of the identified aircraft type in the centre that
you can drag to rotate 360 degrees (three.js, bundled in ./static so it works
offline). Built on Gradio so it runs the trained models inside Colab;
launch(share=True) prints a public link for the viva.
"""
from __future__ import annotations

import base64
import html
import time
from pathlib import Path

import cv2
import numpy as np

from .config import CLASSES, PRETTY, Config

STATIC = Path(__file__).resolve().parent / "static"

CLASS_COLOR = {
    "passenger_jet": "#6EA8FE",
    "fighter_jet": "#FF6B4A",
    "helicopter": "#5EE0A0",
    "propeller": "#C9A2FF",
}
MOD_WORD = {"image": "Photo", "video": "Video", "audio": "Sound"}

ICON = {  # line icons for the orange badge
    "image": '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-8 8"/></svg>',
    "video": '<svg viewBox="0 0 24 24"><rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3z"/></svg>',
    "audio": '<svg viewBox="0 0 24 24"><path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2"/></svg>',
    "none": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><path d="M12 8v5M12 16v.5"/></svg>',
}

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@200;300;400;500&display=swap');

@property --p { syntax: '<integer>'; inherits: true; initial-value: 0; }
@property --x { syntax: '<percentage>'; inherits: true; initial-value: 50%; }

:root, .dark, .gradio-container {
  --bg: #050505;
  --panel: #0d0d0e;
  --line: rgba(255,255,255,.10);
  --line-soft: rgba(255,255,255,.06);
  --text: #EDEDED;
  --muted: #8E8E8E;
  --dim: #575757;
  --accent: #FF4D1F;
  --amber: #FFB020;
  --font: 'Sora', ui-sans-serif, system-ui, sans-serif;

  --body-background-fill: var(--bg);
  --background-fill-primary: transparent;
  --background-fill-secondary: rgba(255,255,255,.03);
  --block-background-fill: transparent;
  --block-border-width: 0px;
  --block-border-color: transparent;
  --block-shadow: none;
  --block-radius: 0px;
  --block-label-background-fill: transparent;
  --block-label-text-color: var(--muted);
  --block-label-border-width: 0px;
  --body-text-color: var(--text);
  --body-text-color-subdued: var(--muted);
  --border-color-primary: var(--line);
  --input-background-fill: transparent;
  --color-accent: var(--accent);
  --checkbox-label-background-fill: transparent;
  --checkbox-label-background-fill-hover: transparent;
  --checkbox-label-background-fill-selected: transparent;
  --checkbox-label-border-width: 0px;
  --checkbox-label-text-color: var(--text);
  --button-secondary-background-fill: rgba(255,255,255,.04);
  --button-secondary-text-color: var(--text);
  --button-secondary-border-color: var(--line);
  color-scheme: dark;
}

html, body { background: var(--bg) !important; }
.gradio-container {
  font-family: var(--font) !important; color: var(--text);
  max-width: 1440px !important; margin: 0 auto !important; padding: 0 28px 60px !important;
  background:
    radial-gradient(ellipse 38% 34% at 50% 34%, rgba(255,77,31,.11), transparent 70%),
    radial-gradient(ellipse 60% 50% at 50% 40%, #191919 0%, transparent 70%),
    var(--bg) !important;
}
.gradio-container * { font-family: inherit; }
footer, .progress-text, .eta-bar, .meta-text, .meta-text-center { display: none !important; }
.gradio-container .form { background: transparent !important; border: none !important; gap: 0 !important; }
.gradio-container .wrap.default, .gradio-container .wrap.translucent { background: transparent !important; }

/* ------------------------------------------------------------ nav */
.nav { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; padding: 22px 6px; border-bottom: 1px solid var(--line-soft); }
.nav .brand { display: flex; align-items: center; gap: 10px; font-weight: 400; letter-spacing: .06em; font-size: 15px; }
.nav .brand i { width: 14px; height: 14px; position: relative; display: inline-block; }
.nav .brand i::before, .nav .brand i::after { content: ""; position: absolute; left: 6px; top: -2px; width: 1.6px; height: 18px; background: var(--text); transform: rotate(45deg); }
.nav .brand i::after { transform: rotate(-45deg); }
.nav .mods { display: flex; gap: 46px; font-size: 13px; color: var(--dim); }
.nav .mods span { transition: color .4s; position: relative; color: var(--dim) !important; }
.nav .mods span.on { color: var(--text) !important; }
.nav .mods span.on::after { content: ""; position: absolute; left: 50%; bottom: -10px; width: 4px; height: 4px; background: var(--accent); transform: translateX(-50%); }
.nav .status { justify-self: end; display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted);
  border: 1px solid var(--line); padding: 7px 14px; border-radius: 999px; }
.nav .status b { width: 6px; height: 6px; border-radius: 50%; background: #3BE07B; box-shadow: 0 0 8px #3BE07B; }

/* ------------------------------------------------------------ hero (left) */
.eyebrow { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--muted); margin: 34px 0 18px; }
.eyebrow::before { content: "\2715"; color: var(--text); font-size: 10px; }
.hero h1 { font-weight: 200; font-size: clamp(42px, 4.6vw, 68px); line-height: 1.02; letter-spacing: -0.035em; margin: 0; color: #F4F4F4; }
.hero h1 .px { display: inline-grid; grid-template-columns: repeat(3, 6px); gap: 2px; margin-left: 12px; vertical-align: middle; }
.hero h1 .px i { width: 6px; height: 6px; background: #CFCFCF; }
.hero h1 .px i:nth-child(2), .hero h1 .px i:nth-child(4), .hero h1 .px i:nth-child(9) { background: transparent; }
.hero p { color: var(--muted); font-size: 15px; line-height: 1.6; max-width: 40ch; margin: 18px 0 26px; font-weight: 300; }

/* ------------------------------------------------------------ chamfered panels */
.panel { position: relative; padding: 22px 22px 20px; margin-bottom: 14px;
  background: var(--line);
  clip-path: polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 0 100%); }
.panel::before { content: ""; position: absolute; inset: 1px; z-index: 0;
  background: linear-gradient(180deg, #131314 0%, var(--panel) 60%);
  clip-path: polygon(0 0, calc(100% - 17.6px) 0, 100% 17.6px, 100% 100%, 0 100%); }
.panel > * { position: relative; z-index: 1; }
.panel h3 { font-weight: 400; font-size: 15px; margin: 0 0 6px; letter-spacing: -0.01em; color: var(--text); }
.panel p { font-size: 13px; line-height: 1.6; color: var(--muted); margin: 0; font-weight: 300; }
.badge { width: 30px; height: 30px; background: var(--accent); display: grid; place-items: center; margin-bottom: 16px; }
.badge svg { width: 17px; height: 17px; fill: none; stroke: #fff; stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round; }
.panel.low .badge { background: var(--amber); }
.tickmark { position: absolute; right: 10px; top: 44px; width: 2px; height: 22px; background: rgba(255,255,255,.35); z-index: 1; }

/* ------------------------------------------------------------ upload column */
#drop { min-height: 168px; background: rgba(255,255,255,.02) !important; border: 1px dashed rgba(255,255,255,.18) !important;
  border-radius: 2px !important; transition: border-color .3s, box-shadow .3s; }
#drop:hover { border-color: var(--accent) !important; box-shadow: 0 0 0 4px rgba(255,77,31,.08), inset 0 0 40px rgba(255,77,31,.06); }
#drop, #drop * { color: var(--muted) !important; font-size: 13px; }
#drop .file-name, #drop td { color: var(--text) !important; }
#drop label, #drop .label-wrap { background: transparent !important; }
#drop table, #drop tr, #drop td, #drop tbody, #drop .file-preview-holder, #drop .file-preview { background: transparent !important; border-color: var(--line) !important; }
#lime { margin: 14px 0 4px; }
#lime label { display: flex !important; align-items: center; gap: 12px; padding: 0 !important; cursor: pointer; }
#lime input[type=checkbox] { appearance: none; -webkit-appearance: none; width: 38px !important; height: 22px !important; border-radius: 11px !important;
  background: #2A2A2B !important; border: 1px solid var(--line) !important; position: relative; flex: none; cursor: pointer; transition: background .25s; margin: 0 !important; }
#lime input[type=checkbox]::after { content: ""; position: absolute; top: 2px; left: 2px; width: 16px; height: 16px; border-radius: 50%; background: #CFCFCF; transition: transform .25s cubic-bezier(.3,1.4,.5,1); }
#lime input[type=checkbox]:checked { background: var(--accent) !important; border-color: var(--accent) !important; background-image: none !important; }
#lime input[type=checkbox]:checked::after { transform: translateX(16px); background: #fff; }
#lime span { font-size: 13px; color: var(--muted) !important; font-weight: 300; }

#identify { position: relative; margin-top: 14px; background: #161617 !important; color: var(--text) !important; border: 1px solid var(--line) !important;
  border-radius: 0 !important; font-weight: 400 !important; font-size: 13px !important; letter-spacing: .14em; padding: 16px 0 !important;
  transition: background .3s, letter-spacing .3s; box-shadow: none !important; overflow: visible !important; }
#identify::before, #identify::after { content: ""; position: absolute; top: -1px; bottom: -1px; width: 7px; border: 2px solid var(--accent); }
#identify::before { left: -1px; border-right: none; }
#identify::after { right: -1px; border-left: none; }
#identify:hover { background: #1F1F21 !important; letter-spacing: .2em; }
button:focus-visible, input:focus-visible, canvas:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 3px; }
.samples { margin-top: 18px; }
.samples button, .samples .gallery-item { background: rgba(255,255,255,.03) !important; border: 1px solid var(--line) !important; color: var(--muted) !important;
  border-radius: 999px !important; font-size: 12px !important; }
.samples button:hover { color: var(--text) !important; border-color: rgba(255,255,255,.3) !important; }

/* ------------------------------------------------------------ stage (centre) */
.stage { position: relative; height: 600px; margin-top: 8px; }
#stage-canvas { position: absolute; inset: 0; cursor: grab; }
#stage-canvas.grabbing { cursor: grabbing; }
#stage-canvas canvas { width: 100% !important; height: 100% !important; display: block; }
#stage-canvas.no-webgl::after { content: "3D preview needs WebGL"; position: absolute; inset: 0; display: grid; place-items: center; color: var(--dim); font-size: 13px; }
.hud { position: absolute; inset: 0; pointer-events: none; }
.hud svg { width: 100%; height: 100%; overflow: visible; }
.hud .orbit { transform-origin: 300px 276px; animation: orbit 60s linear infinite; }
.hud .orbit2 { transform-origin: 300px 276px; animation: orbit 90s linear infinite reverse; }
@keyframes orbit { to { transform: rotate(360deg); } }
.hud circle, .hud line, .hud path { fill: none; stroke: rgba(255,255,255,.16); stroke-width: 1; vector-effect: non-scaling-stroke; }
.hud .x line { stroke: rgba(255,255,255,.55); }
.hud .x circle { stroke: rgba(255,255,255,.4); }
.hud .dot { fill: rgba(255,255,255,.7); stroke: none; }
#stage-canvas[data-state=loading] ~ .hud .orbit { animation-duration: 6s; }
#stage-canvas[data-state=loading] ~ .hud .scan { opacity: 1; }
.hud .scan { opacity: 0; transition: opacity .4s; transform-origin: 300px 276px; animation: orbit 2.2s linear infinite; }
.hud .scan path { stroke: var(--accent); stroke-width: 1.5; }
.stage-caption { text-align: center; margin-top: -54px; position: relative; z-index: 2; pointer-events: none; min-height: 70px; }
.stage-caption .name { font-weight: 200; font-size: 30px; letter-spacing: -0.02em; color: #F2F2F2; animation: rise .7s cubic-bezier(.2,.8,.2,1) both; }
.stage-caption .name.low { color: var(--amber); }
.stage-caption .hint { font-size: 12px; color: var(--dim); margin-top: 6px; letter-spacing: .02em; }
.stage-caption .hint.pulse { animation: blink 1.2s ease-in-out infinite; color: var(--muted); }
@keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
@keyframes blink { 50% { opacity: .35; } }

/* ------------------------------------------------------------ result panel (right) */
.result-row { display: flex; align-items: center; gap: 18px; margin-top: 4px; }
.ring-wrap { position: relative; width: 92px; height: 92px; flex: none; }
.ring { --p: 0; --ring: var(--accent); position: absolute; inset: 0; border-radius: 50%;
  background: conic-gradient(var(--ring) calc(var(--p) * 1%), rgba(255,255,255,.07) 0);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 5px), #000 calc(100% - 4px));
          mask: radial-gradient(farthest-side, transparent calc(100% - 5px), #000 calc(100% - 4px));
  animation: fill 1.5s cubic-bezier(.2,.8,.2,1) .2s forwards; }
.ring-num { position: absolute; inset: 0; display: grid; place-items: center; font-weight: 300; font-size: 24px; letter-spacing: -0.02em;
  --p: 0; animation: fill 1.5s cubic-bezier(.2,.8,.2,1) .2s forwards; }
.ring-num::before { counter-reset: n var(--p); content: counter(n) "%"; }
@keyframes fill { to { --p: var(--target); } }
.panel.low .ring { --ring: var(--amber); }
.type-big { font-weight: 300; font-size: 28px; line-height: 1.1; letter-spacing: -0.02em; color: #F4F4F4; }
.panel.low .type-big { color: var(--amber); }
.small { font-size: 12.5px; color: var(--muted); line-height: 1.55; margin-top: 6px; font-weight: 300; }
.warn { font-size: 12.5px; color: var(--amber); line-height: 1.5; margin-top: 12px; }

.bars { display: grid; gap: 12px; margin-top: 14px; }
.bar { display: grid; grid-template-columns: 1fr auto; gap: 6px 10px; font-size: 13px; font-weight: 300; color: var(--muted); }
.bar.top { color: var(--text); }
.bar .pct { font-variant-numeric: tabular-nums; }
.bar .track { grid-column: 1 / 3; height: 3px; background: rgba(255,255,255,.07); overflow: hidden; }
.bar .fill { height: 100%; transform-origin: left; animation: grow 1s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--i) * 90ms + 150ms); }
@keyframes grow { from { transform: scaleX(0); } }
.note { font-size: 12px; color: var(--dim); margin-top: 12px; line-height: 1.5; }

.skeleton { height: 12px; background: linear-gradient(90deg, rgba(255,255,255,.04), rgba(255,255,255,.11), rgba(255,255,255,.04));
  background-size: 200% 100%; animation: shimmer 1.3s linear infinite; margin-top: 10px; }
@keyframes shimmer { to { background-position: -200% 0; } }

/* ------------------------------------------------------------ evidence */
.section { margin-top: 70px; border-top: 1px solid var(--line-soft); padding-top: 46px; }
.section h2 { font-weight: 200; font-size: clamp(32px, 3.4vw, 48px); letter-spacing: -0.03em; margin: 0 0 10px; color: #F2F2F2; }
.section .lede { color: var(--muted); font-size: 14px; line-height: 1.6; max-width: 62ch; margin: 0 0 28px; font-weight: 300; }
.ev-grid { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr); gap: 22px; align-items: start; }
.cmp { position: relative; overflow: hidden; background: #0b0b0b; border: 1px solid var(--line); user-select: none;
  animation: sweep 1.8s cubic-bezier(.6,0,.2,1) .5s backwards; }
@keyframes sweep { 0% { --x: 100%; } 55% { --x: 18%; } 100% { --x: 50%; } }
.cmp img { display: block; width: 100%; height: auto; }
.cmp .over { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; clip-path: inset(0 0 0 var(--x)); }
.cmp .handle { position: absolute; top: 0; bottom: 0; left: var(--x); width: 1px; background: #fff; pointer-events: none; box-shadow: 0 0 12px rgba(255,255,255,.6); }
.cmp .handle::after { content: ""; position: absolute; top: 50%; left: -16px; width: 30px; height: 30px; margin-top: -15px; border-radius: 50%;
  border: 1px solid #fff; background: rgba(0,0,0,.45) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.6'%3E%3Cpath d='M9 7l-5 5 5 5M15 7l5 5-5 5'/%3E%3C/svg%3E") center/18px no-repeat; backdrop-filter: blur(4px); }
.cmp-range { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; cursor: ew-resize; margin: 0; }
.cmp .tag { position: absolute; bottom: 12px; font-size: 11.5px; padding: 6px 10px; background: rgba(0,0,0,.55); backdrop-filter: blur(6px); color: #ddd; pointer-events: none; }
.cmp .tag.l { left: 12px; } .cmp .tag.r { right: 12px; }
.thumbs { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.thumbs figure { margin: 0; border: 1px solid var(--line); background: #0b0b0b; transition: border-color .3s, transform .4s cubic-bezier(.2,.8,.2,1); }
.thumbs figure:hover { border-color: rgba(255,255,255,.3); transform: translateY(-3px); }
.thumbs img { display: block; width: 100%; aspect-ratio: 1; object-fit: cover; }
.thumbs figcaption { font-size: 12px; color: var(--muted); padding: 10px 12px; font-weight: 300; }
.thumbs figure.wide { grid-column: 1 / 3; }
.thumbs figure.wide img { aspect-ratio: auto; }

.timeline { display: flex; gap: 8px; overflow-x: auto; padding: 4px 2px 14px; scrollbar-width: thin; scrollbar-color: #333 transparent; }
.frame { flex: none; width: 112px; transition: transform .35s cubic-bezier(.2,.8,.2,1); animation: rise .6s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--i) * 45ms); }
.frame:hover { transform: translateY(-6px) scale(1.04); }
.frame img { width: 100%; display: block; border: 1px solid var(--line); }
.frame .cbar { height: 3px; margin-top: 6px; }
.frame .lbl { font-size: 11px; color: var(--muted); margin-top: 6px; display: flex; justify-content: space-between; font-weight: 300; }
.windows { display: flex; gap: 4px; margin-top: 4px; }
.windows div { flex: 1; height: 44px; animation: rise .5s both; animation-delay: calc(var(--i) * 40ms); }
.legend { display: flex; gap: 22px; flex-wrap: wrap; font-size: 12px; color: var(--muted); margin-top: 10px; font-weight: 300; }
.legend i { display: inline-block; width: 8px; height: 8px; margin-right: 8px; }
.fusion { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 12px; }
.fusion .panel { margin: 0; }
.fusion .big { font-weight: 200; font-size: 34px; letter-spacing: -0.02em; margin: 8px 0 2px; color: #F2F2F2; }
.sub { font-size: 14px; color: var(--text); margin: 34px 0 12px; font-weight: 400; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-delay: 0ms !important; transition-duration: .01ms !important; }
}
@media (max-width: 900px) {
  .nav { grid-template-columns: 1fr auto; } .nav .mods { display: none; }
  .stage { height: 420px; }
  .ev-grid, .fusion { grid-template-columns: 1fr; }
}
"""

HUD_SVG = """
<div class="hud" aria-hidden="true"><svg viewBox="0 0 600 600" preserveAspectRatio="xMidYMid meet">
  <g class="orbit">
    <circle cx="300" cy="276" r="232"/>
    <g class="x" transform="translate(126 128)"><circle r="13"/><line x1="-5" y1="-5" x2="5" y2="5"/><line x1="5" y1="-5" x2="-5" y2="5"/></g>
    <g class="x" transform="translate(495 392)"><circle r="15"/><line x1="-5" y1="-5" x2="5" y2="5"/><line x1="5" y1="-5" x2="-5" y2="5"/></g>
    <circle class="dot" cx="300" cy="44" r="2.5"/><circle class="dot" cx="72" cy="300" r="2"/>
  </g>
  <g class="orbit2">
    <circle cx="300" cy="276" r="292" stroke-dasharray="2 10"/>
    <path d="M300 -16 v14 M300 554 v14 M8 276 h14 M578 276 h14"/>
  </g>
  <g class="scan"><path d="M300 44 A232 232 0 0 1 532 276"/></g>
  <path d="M286 276 h28 M300 262 v28" style="stroke: rgba(255,255,255,.25)"/>
</svg></div>
"""


# ------------------------------------------------------------------ helpers
def _e(s) -> str:
    return html.escape(str(s))


def data_uri(img: np.ndarray, max_side: int = 720, png: bool = False) -> str:
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
    ok, buf = cv2.imencode(".png" if png else ".jpg", bgr, [] if png else [cv2.IMWRITE_JPEG_QUALITY, 88])
    return f"data:image/{'png' if png else 'jpeg'};base64,{base64.b64encode(buf).decode()}"


def square(img: np.ndarray, side: int) -> np.ndarray:
    """Letterbox to a square on black, so two images line up in the compare slider."""
    h, w = img.shape[:2]
    s = side / max(h, w)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((side, side, 3), np.uint8)
    y, x = (side - r.shape[0]) // 2, (side - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def tint_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape + (3,), np.uint8)
    out[:] = (13, 13, 14)
    out[mask > 127] = (255, 77, 31)
    return out


def _head_script() -> str:
    d = str(STATIC).replace("\\", "/")
    return f"""<script>
(function () {{
  var dir = {d!r};
  var bases = ["gradio_api/file=" + dir, "/gradio_api/file=" + dir, "file=" + dir, "/file=" + dir];
  function load(src, ok, fail) {{ var s = document.createElement("script"); s.src = src; s.onload = ok; s.onerror = fail; document.head.appendChild(s); }}
  function tryBase(i) {{
    if (i >= bases.length) {{ console.error("Could not load the 3D viewer files"); return; }}
    load(bases[i] + "/three.min.js", function () {{ load(bases[i] + "/viewer3d.js", function () {{}}, function () {{}}); }},
         function () {{ tryBase(i + 1); }});
  }}
  tryBase(0);
}})();
</script>"""


# ------------------------------------------------------------------ HTML pieces
def nav_html(device: str) -> str:
    return (f'<nav class="nav"><div class="brand"><i></i>AERO-ID</div>'
            f'<div class="mods"><span data-mod-item="image">Photo</span><span data-mod-item="video">Video</span>'
            f'<span data-mod-item="audio">Sound</span></div>'
            f'<div class="status"><b></b>Models ready on {device}</div></nav>')


def hero_html() -> str:
    return ('<div class="hero"><div class="eyebrow">Explainable aircraft recognition</div>'
            '<h1>Know what’s<br>flying overhead<span class="px">' + "<i></i>" * 9 + '</span></h1>'
            '<p>Drop in a photo, a video clip or a sound recording. The model names the aircraft type '
            'and shows exactly what it based that on.</p></div>')


def stage_html() -> str:
    return f'<div class="stage"><div id="stage-canvas" data-state="idle"></div>{HUD_SVG}</div>'


def vstate_html(state: str, cls: str = "", mod: str = "") -> str:
    key = f"{state}-{cls}-{time.time():.3f}"
    if state in ("result", "low"):
        cap = (f'<div class="name{" low" if state == "low" else ""}">{_e(PRETTY[cls])}</div>'
               f'<div class="hint">Drag to rotate, scroll to zoom, double-click to reset</div>')
    elif state == "loading":
        cap = '<div class="hint pulse">Analysing…</div>'
    elif state == "error":
        cap = '<div class="hint">No model to show</div>'
    else:
        cap = ('<div class="hint">Recognises passenger jets, fighter jets, helicopters and propeller aircraft. '
               'Drag the hologram to look around.</div>')
    return (f'<div class="vstate" data-key="{key}" data-state="{state}" data-cls="{cls}" data-mod="{mod}" hidden></div>'
            f'<div class="stage-caption">{cap}</div>')


def result_html(r: dict | None = None, state: str = "idle", error: str = "") -> str:
    if state == "loading":
        return ('<div class="panel"><div class="badge">' + ICON["none"] + '</div><h3>Analysing the file</h3>'
                '<div class="skeleton" style="width:70%"></div><div class="skeleton" style="width:45%"></div>'
                '<div class="skeleton" style="width:58%"></div></div>')
    if state == "error":
        return (f'<div class="panel low"><div class="badge">{ICON["none"]}</div><h3>File not analysed</h3>'
                f'<p>{_e(error)}</p></div>')
    if r is None:
        return ('<div class="panel"><div class="tickmark"></div><div class="badge">' + ICON["none"] + '</div>'
                '<h3>Waiting for a file</h3><p>The identified type, how confident the model is, '
                'and a 3D model of the aircraft will appear here.</p></div>')
    kind = r["modality"]
    if kind == "video":
        detail = f'From {r.get("n_frames", 0)} video frames' + (" and the soundtrack." if r.get("audio_probs") is not None else ".")
    elif kind == "audio":
        n = r.get("n_windows", 1)
        detail = f'From {n} five-second {"piece" if n == 1 else "pieces"} of sound.'
    else:
        detail = "From a single photo."
    pct = int(round(r["confidence"] * 100))
    warn = (f'<div class="warn">Below the {int(r["threshold"] * 100)}% confidence limit, so this may not be '
            'an aircraft at all.</div>' if r["rejected"] else "")
    return (f'<div class="panel{" low" if r["rejected"] else ""}"><div class="tickmark"></div>'
            f'<div class="badge">{ICON[kind]}</div><h3>{"Unconfirmed" if r["rejected"] else "Identified"}</h3>'
            f'<div class="result-row"><div class="ring-wrap" role="img" aria-label="{pct} percent confident">'
            f'<div class="ring" style="--target:{pct}"></div><div class="ring-num" style="--target:{pct}"></div></div>'
            f'<div><div class="type-big">{_e(r["label"])}</div><div class="small">{_e(detail)}</div></div></div>'
            f'{warn}</div>')


def bars_html(r: dict | None) -> str:
    if not r:
        return ""
    probs = r["probabilities"]
    top = max(probs, key=probs.get)
    audio = r["modality"] == "audio"
    rows = []
    for i, c in enumerate(CLASSES):
        name = PRETTY[c]
        p = probs[name]
        na = audio and c == "fighter_jet"
        rows.append(f'<div class="bar{" top" if name == top else ""}"><span>{_e(name)}</span>'
                    f'<span class="pct">{"n/a" if na else f"{p:.0%}"}</span>'
                    f'<div class="track"><div class="fill" style="--i:{i};width:{0 if na else p * 100:.1f}%;'
                    f'background:{CLASS_COLOR[c]}"></div></div></div>')
    note = ('<div class="note">The training recordings contain no fighter jets, so the sound model '
            'can’t recognise them.</div>' if audio else "")
    return f'<div class="panel"><h3>Confidence by type</h3><div class="bars">{"".join(rows)}</div>{note}</div>'


def route_html(r: dict | None) -> str:
    if not r:
        return ""
    ro = r["route"]
    how = {"content and extension agree": "Its contents and its file name both say so.",
           "recognised from extension only": "Recognised from its file name; the contents were not conclusive.",
           "unknown extension, recognised from content": "Recognised from its contents; the file name was unfamiliar."
           }.get(ro["note"], f"The file name says {ro['by_extension']}, but its contents say {ro['by_content']}, "
                             "so the contents were trusted.")
    return f'<div class="panel"><h3>Read as {MOD_WORD[r["modality"]].lower()}</h3><p>{_e(how)}</p></div>'


def legend_html() -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{CLASS_COLOR[c]}"></i>{PRETTY[c]}</span>' for c in CLASSES) + "</div>"


def evidence_html(r: dict | None) -> str:
    if not r:
        return ""
    kind = r["modality"]
    if kind == "audio":
        base, over = r["input"], r.get("gradcam")
        tags = ("The sound as a picture", "What it listened to")
        lede = ("The recording is drawn as a spectrogram: time runs left to right and pitch rises upwards. "
                "Drag across it to reveal which sounds the model relied on; red areas mattered most.")
    else:
        side = 640
        # Grad-CAM was computed on the cut-out resized to a square, so stretch it the same way to line up.
        base = cv2.resize(r["cutout"], (side, side), interpolation=cv2.INTER_AREA)
        over = cv2.resize(r["gradcam"], (side, side)) if r.get("gradcam") is not None else None
        tags = ("What the model sees", "Where it looked")
        lede = ("The aircraft is cut out from the sky first, so the model judges the aircraft and not the background. "
                "Drag across the image to reveal where it looked; red areas mattered most to the answer.")
        if kind == "video":
            lede = "Showing the frame the model was most sure about. " + lede
    cmp = ""
    if over is not None:
        if over.shape[:2] != base.shape[:2]:
            over = cv2.resize(over, (base.shape[1], base.shape[0]))
        cmp = (f'<div class="cmp" style="--x:50%"><img src="{data_uri(base)}" alt="{tags[0]}">'
               f'<img class="over" src="{data_uri(over)}" alt="{tags[1]}"><div class="handle"></div>'
               f'<input class="cmp-range" type="range" min="0" max="100" value="50" aria-label="Compare the image with the heatmap" '
               f'oninput="this.parentNode.style.setProperty(\'--x\', this.value + \'%\')">'
               f'<span class="tag l">{tags[0]}</span><span class="tag r">{tags[1]}</span></div>')
    figs = []
    if kind != "audio":
        figs.append(("Original", r["input"], ""))
        if r.get("mask") is not None:
            figs.append(("Where the aircraft is", tint_mask(r["mask"]), ""))
        if r.get("lime") is not None:
            figs.append(("Supporting regions (LIME)", r["lime"], "wide" if len(figs) % 2 == 0 else ""))
    thumbs = "".join(f'<figure class="{c}"><img src="{data_uri(im, 520, png=(t == "Where the aircraft is"))}" alt="{_e(t)}">'
                     f'<figcaption>{_e(t)}</figcaption></figure>' for t, im, c in figs)
    grid = f'<div class="ev-grid">{cmp}<div class="thumbs">{thumbs}</div></div>' if thumbs else cmp
    return f'<section class="section"><h2>Why it decided</h2><p class="lede">{_e(lede)}</p>{grid}</section>'


def breakdown_html(r: dict | None) -> str:
    if not r:
        return ""
    kind = r["modality"]
    if kind == "video":
        votes, fp = r["votes"], r["frame_probs"]
        winner = CLASSES[int(np.argmax(votes))]
        frames = []
        for i, (th, p) in enumerate(zip(r.get("frame_thumbs", []), fp)):
            c = CLASSES[int(np.argmax(p))]
            frames.append(f'<div class="frame" style="--i:{i}" title="Frame {i + 1}: {PRETTY[c]}, {float(np.max(p)):.0%}">'
                          f'<img src="{data_uri(th, 160)}" alt="Frame {i + 1}">'
                          f'<div class="cbar" style="background:{CLASS_COLOR[c]};opacity:{0.35 + 0.65 * float(np.max(p)):.2f}"></div>'
                          f'<div class="lbl"><span>{i + 1}</span><span>{float(np.max(p)):.0%}</span></div></div>')
        out = (f'<section class="section"><h2>Frame by frame</h2><p class="lede">{int(votes.max())} of {int(votes.sum())} '
               f'frames say {PRETTY[winner].lower()}. The bar under each frame shows what the model saw in it; '
               f'paler bars were less certain. Scroll sideways to see them all.</p>'
               f'<div class="timeline">{"".join(frames)}</div>{legend_html()}')
        if r.get("audio_probs") is not None:
            w = r.get("audio_weight", 0.3)
            card = lambda title, p, note: (f'<div class="panel"><h3>{title}</h3><div class="big">{float(np.max(p)):.0%}</div>'
                                           f'<p>{PRETTY[CLASSES[int(np.argmax(p))]]}. {note}</p></div>')
            out += ('<div class="sub">Picture and sound combined</div><div class="fusion">'
                    + card("From the frames", r["image_probs"], f"Counts for {1 - w:.0%} of the answer.")
                    + card("From the soundtrack", r["audio_probs"], f"Counts for {w:.0%} of the answer.")
                    + card("Combined", r["probs"], "This is the answer shown above.") + "</div>")
        return out + "</section>"
    if kind == "audio" and r.get("window_probs") is not None and len(r["window_probs"]) > 1:
        cells = "".join(f'<div style="--i:{i};background:{CLASS_COLOR[CLASSES[int(np.argmax(p))]]};'
                        f'opacity:{0.35 + 0.65 * float(np.max(p)):.2f}" title="Piece {i + 1}: '
                        f'{PRETTY[CLASSES[int(np.argmax(p))]]}"></div>' for i, p in enumerate(r["window_probs"]))
        return (f'<section class="section"><h2>Piece by piece</h2><p class="lede">The recording was cut into '
                f'five-second pieces. Each block is one piece, coloured by the type the model heard.</p>'
                f'<div class="windows">{cells}</div>{legend_html()}</section>')
    return ""


# ------------------------------------------------------------------ app
def _example_files(cfg: Config):
    for d in (cfg.root / "demo_files", Path(__file__).resolve().parents[1] / "demo_files"):
        if d.exists():
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".mp4", ".wav", ".mp3"))
            if files:
                return [[str(p)] for p in files[:8]]
    return []


def build_demo(cfg: Config):
    import gradio as gr
    from .pipeline import AircraftAgent
    agent = AircraftAgent(cfg)
    device = "GPU" if str(agent.device).startswith("cuda") else "CPU"
    gr.set_static_paths(paths=[str(STATIC)])
    theme = gr.themes.Base(font=[gr.themes.GoogleFont("Sora"), "ui-sans-serif", "sans-serif"])

    def idle():
        return (vstate_html("idle"), result_html(), "", "", "", "")

    def loading(file):
        if file is None:
            return idle()
        return (vstate_html("loading"), result_html(state="loading"), "", "", "", "")

    def run(file, use_lime):
        if file is None:
            return idle()
        path = file if isinstance(file, str) else file.name
        try:
            r = agent.predict(path, explain=True, use_lime=bool(use_lime))
        except Exception as e:
            msg = str(e)
            if "Unsupported" in msg:
                msg = ("This isn’t a photo, video or sound recording. "
                       "Try a JPG or PNG image, an MP4 video, or a WAV or MP3 recording.")
            return (vstate_html("error"), result_html(state="error", error=msg), "", "", "", "")
        r["threshold"] = cfg.reject_threshold
        r["audio_weight"] = cfg.fusion_audio_weight
        state = "low" if r["rejected"] else "result"
        return (vstate_html(state, r["class"], r["modality"]), result_html(r),
                bars_html(r), route_html(r), evidence_html(r), breakdown_html(r))

    with gr.Blocks(title="AERO-ID – aircraft recognition", **_blocks_kwargs(gr, theme)) as demo:
        gr.HTML(nav_html(device))
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=300):
                gr.HTML(hero_html())
                f = gr.File(label="Photo, video or sound", type="filepath", elem_id="drop")
                lime = gr.Checkbox(label="Add a second explanation (LIME). Slower; photos and video only.", elem_id="lime")
                btn = gr.Button("IDENTIFY AIRCRAFT", elem_id="identify")
                ex = _example_files(cfg)
                if ex:
                    with gr.Column(elem_classes="samples"):
                        gr.Examples(ex, inputs=[f], label="Or try a sample")
            with gr.Column(scale=5, min_width=320):
                gr.HTML(stage_html())
                vstate = gr.HTML(vstate_html("idle"), elem_id="viewer-state")
            with gr.Column(scale=3, min_width=280):
                gr.HTML('<div style="height:38px"></div>')
                result = gr.HTML(result_html())
                bars = gr.HTML()
                route = gr.HTML()
        evidence = gr.HTML()
        breakdown = gr.HTML()
        outs = [vstate, result, bars, route, evidence, breakdown]
        quiet = dict(show_progress="hidden")
        for trigger in (f.change, btn.click):
            trigger(loading, [f], outs, queue=False, **quiet).then(run, [f, lime], outs, **quiet)
    demo._aircraft_launch = _launch_kwargs(gr, theme)
    return demo


def _major(gr) -> int:
    try:
        return int(gr.__version__.split(".")[0])
    except Exception:
        return 4


def _blocks_kwargs(gr, theme):
    # Gradio 4/5 take theme, css and head on Blocks; Gradio 6 moved them to launch().
    return {"theme": theme, "css": CSS, "head": _head_script()} if _major(gr) < 6 else {}


def _launch_kwargs(gr, theme):
    kw = {"allowed_paths": [str(STATIC)]}
    if _major(gr) >= 6:
        kw.update(theme=theme, css=CSS, head=_head_script())
    return kw


def launch(cfg: Config, share: bool = True, **kw):
    demo = build_demo(cfg)
    return demo.launch(share=share, **{**demo._aircraft_launch, **kw})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./workdir")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    launch(Config.load(Path(a.root)), share=a.share, server_port=a.port)
