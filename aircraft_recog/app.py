"""Step 10 + feature 10: the demo interface.

Visual identity: an aeronautical sectional chart (chart paper, navy ink,
airspace magenta, airway blue), with the answer presented as an air-traffic
control flight strip. Built with Gradio so it runs the trained models inside
Colab; launch(share=True) prints a public link for the viva.
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from .config import CLASSES, PRETTY, Config

# Colour of each class wherever it is drawn (bars, frame ticks, legend).
CLASS_COLOR = {
    "passenger_jet": "#2B5C9E",   # airway blue
    "fighter_jet": "#A8236F",     # airspace magenta
    "helicopter": "#3E7D5A",      # terrain green
    "propeller": "#C98A16",       # caution amber
}

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow+Semi+Condensed:wght@400;500;600&display=swap');

:root, .dark, .gradio-container {
  --paper: #EEF1EA;
  --paper-2: #E4E9DE;
  --ink: #16233F;
  --ink-soft: #4A5670;
  --magenta: #A8236F;
  --blue: #2B5C9E;
  --terrain: #DCE4CF;
  --amber: #C98A16;
  --rule: #B9C3B0;

  --body-background-fill: var(--paper);
  --background-fill-primary: var(--paper);
  --background-fill-secondary: var(--paper-2);
  --block-background-fill: transparent;
  --block-border-color: var(--rule);
  --block-border-width: 1px;
  --block-radius: 3px;
  --block-shadow: none;
  --block-label-background-fill: var(--paper);
  --block-label-text-color: var(--ink-soft);
  --block-label-border-width: 0px;
  --block-title-text-color: var(--ink);
  --body-text-color: var(--ink);
  --body-text-color-subdued: var(--ink-soft);
  --border-color-primary: var(--rule);
  --input-background-fill: #F6F8F3;
  --color-accent: var(--magenta);
  --color-accent-soft: #F2DCE8;
  --checkbox-background-color-selected: var(--magenta);
  --checkbox-label-background-fill: transparent;
  --checkbox-label-background-fill-hover: transparent;
  --checkbox-label-background-fill-selected: transparent;
  --checkbox-label-border-width: 0px;
  --checkbox-label-shadow: none;
  --checkbox-label-text-color: var(--ink);
  --checkbox-border-color-selected: var(--magenta);
  --button-primary-background-fill: var(--ink);
  --button-primary-background-fill-hover: #22345C;
  --button-primary-text-color: var(--paper);
  --button-primary-border-color: var(--ink);
  --font: 'Barlow Semi Condensed', 'Arial Narrow', sans-serif;
  color-scheme: light;
}

body, .gradio-container { background: var(--paper) !important; color: var(--ink); }
.gradio-container { font-family: var(--font) !important; max-width: 1240px !important; margin: 0 auto !important; }
.gradio-container * { font-family: inherit; }
footer { display: none !important; }

/* Faint chart graticule: lines of latitude and longitude every 120px. */
.gradio-container::before {
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    linear-gradient(to right, rgba(43,92,158,.07) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(43,92,158,.07) 1px, transparent 1px);
  background-size: 120px 120px;
}
.gradio-container > * { position: relative; z-index: 1; }

/* ---------- header ---------- */
#masthead { padding: 28px 4px 18px; border-bottom: 2px solid var(--ink); margin-bottom: 8px; }
#masthead .row { display: flex; align-items: flex-end; gap: 28px; flex-wrap: wrap; }
#masthead h1 {
  font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 56px; line-height: .9;
  letter-spacing: -0.01em; margin: 0; color: var(--ink);
}
#masthead p { margin: 0 0 4px; max-width: 58ch; font-size: 17px; line-height: 1.4; color: var(--ink-soft); }
#masthead .airway { height: 0; border-top: 3px solid var(--magenta); width: 96px; margin-top: 14px; }

/* ---------- section captions ---------- */
.caption { font-size: 15px; color: var(--ink-soft); margin: 18px 2px 6px; }
.caption b { color: var(--ink); font-weight: 600; }

/* ---------- input column ---------- */
#drop { min-height: 190px; border: 1.5px dashed var(--blue) !important; background: #F6F8F3 !important; }
#drop .wrap, #drop button { color: var(--blue) !important; font-size: 16px; }
#identify { font-family: 'Barlow Condensed', sans-serif !important; font-size: 21px !important; font-weight: 600 !important;
  letter-spacing: .01em; border-radius: 3px !important; padding: 12px 0 !important; }
#identify:focus-visible, button:focus-visible, input:focus-visible { outline: 3px solid var(--magenta) !important; outline-offset: 2px; }
.hint { font-size: 14px; color: var(--ink-soft); line-height: 1.45; margin: 6px 2px 0; }

/* ---------- the flight strip ---------- */
.strip { display: grid; grid-template-columns: 14px minmax(0, 2.2fr) minmax(0, 1fr) minmax(0, 1.3fr);
  background: #FBFCF8; border: 1.5px solid var(--ink); border-radius: 2px; min-height: 132px;
  box-shadow: 4px 4px 0 var(--terrain); animation: slotIn .35s ease-out; }
@keyframes slotIn { from { transform: translateX(-14px); opacity: 0; } to { transform: none; opacity: 1; } }
@media (prefers-reduced-motion: reduce) { .strip { animation: none; } }
.strip .band { background: var(--magenta); }
.strip.unconfirmed .band { background: repeating-linear-gradient(135deg, var(--amber) 0 8px, #E9C77E 8px 16px); }
.strip.empty .band { background: var(--rule); }
.strip .cell { padding: 14px 16px; border-left: 1px solid var(--rule); display: flex; flex-direction: column; justify-content: space-between; gap: 8px; }
.strip .cell:nth-child(2) { border-left: none; }
.strip .k { font-size: 13px; color: var(--ink-soft); }
.strip .type { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 46px; line-height: .95; color: var(--ink); }
.strip.unconfirmed .type { color: var(--ink-soft); }
.strip .big { font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 40px; line-height: 1; }
.strip .med { font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 22px; line-height: 1.1; }
.strip .small { font-size: 14px; line-height: 1.35; color: var(--ink-soft); }
.strip .flag { font-size: 14px; color: #8A5E0A; font-weight: 600; }
.strip.empty .type { font-size: 30px; color: var(--ink-soft); font-weight: 600; }

/* ---------- confidence bars ---------- */
.bars { margin-top: 14px; display: grid; gap: 7px; }
.bar { display: grid; grid-template-columns: 150px 1fr 52px; align-items: center; gap: 12px; font-size: 16px; }
.bar .track { height: 12px; background: var(--paper-2); border: 1px solid var(--rule); border-radius: 1px; overflow: hidden; }
.bar .fill { height: 100%; transition: width .5s ease-out; }
.bar .pct { text-align: right; font-variant-numeric: tabular-nums; color: var(--ink-soft); }
.bar.top { font-weight: 600; }
.bar.top .pct { color: var(--ink); }
.bar .na { color: var(--ink-soft); font-size: 13px; }

/* ---------- evidence ---------- */
.evidence { border-color: var(--rule) !important; background: #F6F8F3 !important; }
.evidence img { width: 100% !important; height: 100% !important; object-fit: contain !important; }
.evidence .icon-button-wrapper, .evidence .icon-buttons, .evidence button[aria-label="Share"],
.evidence button[title="Share"] { display: none !important; }
.gradio-container .form { background: transparent !important; border: none !important; gap: 0 !important; }
#lime, #lime label, #lime .wrap { background: transparent !important; border: none !important; box-shadow: none !important; }
#lime label span { color: var(--ink); font-size: 15px; }

/* ---------- frame / window ticks ---------- */
.ticks { display: flex; gap: 3px; margin: 6px 0 4px; flex-wrap: wrap; }
.tick { flex: 1 0 18px; max-width: 46px; height: 34px; border-radius: 1px; position: relative; }
.tick span { position: absolute; bottom: 2px; left: 0; right: 0; text-align: center; font-size: 11px; color: #fff; opacity: .9; }
.legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: 14px; color: var(--ink-soft); margin-top: 6px; }
.legend i { display: inline-block; width: 11px; height: 11px; margin-right: 6px; vertical-align: -1px; border-radius: 1px; }
.fusion { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-top: 10px; }
.fusion .src { border-top: 2px solid var(--ink); padding-top: 6px; }
.fusion .src h4 { margin: 0 0 4px; font-family: 'Barlow Condensed', sans-serif; font-size: 19px; font-weight: 600; }
.fusion .src p { margin: 0; font-size: 15px; color: var(--ink-soft); }
.router-note { font-size: 14px; color: var(--ink-soft); margin: 10px 2px 0; line-height: 1.4; }
.router-note b { color: var(--ink); font-weight: 600; }

@media (max-width: 760px) {
  #masthead h1 { font-size: 40px; }
  .strip { grid-template-columns: 10px 1fr 1fr; }
  .strip .cell:nth-child(4) { grid-column: 2 / 4; border-left: none; border-top: 1px solid var(--rule); }
  .strip .type { font-size: 36px; }
  .bar { grid-template-columns: 112px 1fr 44px; font-size: 15px; }
  .fusion { grid-template-columns: 1fr; }
}
"""

MODALITY_WORD = {"image": "Photo", "video": "Video", "audio": "Sound"}


# ------------------------------------------------------------------ HTML pieces
def _e(s) -> str:
    return html.escape(str(s))


def strip_html(r: dict | None = None, error: str | None = None) -> str:
    if error:
        return (f'<div class="strip unconfirmed"><div class="band"></div>'
                f'<div class="cell"><span class="k">Couldn\'t identify</span>'
                f'<span class="type" style="font-size:28px">File not analysed</span></div>'
                f'<div class="cell" style="grid-column: span 2"><span class="small">{_e(error)}</span></div></div>')
    if r is None:
        return ('<div class="strip empty"><div class="band"></div>'
                '<div class="cell"><span class="k">Aircraft type</span><span class="type">Waiting for a file</span></div>'
                '<div class="cell"><span class="k">Confidence</span><span class="big" style="color:var(--rule)">--</span></div>'
                '<div class="cell"><span class="k">Input</span><span class="small">Add a photo, video or sound '
                'recording on the left, then choose Identify aircraft.</span></div></div>')
    cls = "strip unconfirmed" if r["rejected"] else "strip"
    kind = r["modality"]
    if kind == "video":
        detail = f'{r.get("n_frames", 0)} frames checked'
        if r.get("audio_probs") is not None:
            detail += ", with soundtrack"
    elif kind == "audio":
        n = r.get("n_windows", 1)
        detail = f'{n} five-second {"window" if n == 1 else "windows"}'
    else:
        detail = "Single image"
    flag = ('<span class="flag">Unconfirmed: confidence is below the '
            f'{int(r["threshold"] * 100)}% limit, so this may not be an aircraft.</span>'
            if r["rejected"] else '<span class="small">Model\'s best answer</span>')
    return (f'<div class="{cls}"><div class="band"></div>'
            f'<div class="cell"><span class="k">Aircraft type</span><span class="type">{_e(r["label"])}</span>{flag}</div>'
            f'<div class="cell"><span class="k">Confidence</span><span class="big">{r["confidence"]:.0%}</span></div>'
            f'<div class="cell"><span class="k">Input</span><span class="med">{MODALITY_WORD[kind]}</span>'
            f'<span class="small">{_e(detail)}</span></div></div>')


def bars_html(probs: dict | None, audio_only=False) -> str:
    if not probs:
        return ""
    top = max(probs, key=probs.get)
    rows = []
    for c in CLASSES:
        name = PRETTY[c]
        p = probs[name]
        na = audio_only and c == "fighter_jet"
        rows.append(
            f'<div class="bar{" top" if name == top else ""}"><span>{_e(name)}</span>'
            f'<div class="track"><div class="fill" style="width:{0 if na else p * 100:.1f}%;background:{CLASS_COLOR[c]}"></div></div>'
            f'<span class="pct">{"<span class=na>n/a</span>" if na else f"{p:.0%}"}</span></div>')
    note = ('<p class="hint">The sound recordings used for training have no fighter jets, so the sound model '
            'can\'t recognise them.</p>' if audio_only else "")
    return f'<div class="bars">{"".join(rows)}</div>{note}'


def legend_html() -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{CLASS_COLOR[c]}"></i>{PRETTY[c]}</span>' for c in CLASSES) + "</div>"


def ticks_html(per_item_probs: np.ndarray, item: str) -> str:
    cells = []
    for k, p in enumerate(per_item_probs):
        c = CLASSES[int(np.argmax(p))]
        cells.append(f'<div class="tick" style="background:{CLASS_COLOR[c]};opacity:{0.45 + 0.55 * float(np.max(p)):.2f}" '
                     f'title="{item} {k + 1}: {PRETTY[c]} {float(np.max(p)):.0%}"><span>{k + 1}</span></div>')
    return f'<div class="ticks">{"".join(cells)}</div>'


def breakdown_html(r: dict | None) -> str:
    """Extra evidence for video (frame votes, sound fusion) and audio (per-window answers)."""
    if not r:
        return ""
    kind = r["modality"]
    if kind == "video":
        votes = r["votes"]
        winner = CLASSES[int(np.argmax(votes))]
        out = (f'<p class="caption"><b>Frame by frame.</b> Each block is one frame, coloured by the type the model saw in it; '
               f'paler blocks were less certain. {int(votes.max())} of {int(votes.sum())} frames voted '
               f'{PRETTY[winner].lower()}.</p>' + ticks_html(r["frame_probs"], "Frame") + legend_html())
        if r.get("audio_probs") is not None:
            ip, ap, fp = r["image_probs"], r["audio_probs"], r["probs"]
            src = lambda title, p, note: (f'<div class="src"><h4>{title}</h4><p>{PRETTY[CLASSES[int(np.argmax(p))]]}, '
                                          f'{float(np.max(p)):.0%}. {note}</p></div>')
            w = r.get("audio_weight", 0.3)
            out += ('<p class="caption"><b>Picture and sound combined.</b> The clip has a soundtrack, so both models voted.</p>'
                    '<div class="fusion">' + src("From the frames", ip, f"Weight {1 - w:.0%}.") +
                    src("From the sound", ap, f"Weight {w:.0%}.") + src("Combined answer", fp, "Shown in the strip above.") +
                    "</div>")
        return out
    if kind == "audio" and r.get("window_probs") is not None and len(r["window_probs"]) > 1:
        return ('<p class="caption"><b>Window by window.</b> The recording was cut into five-second pieces; '
                'each block is one piece, coloured by the type the model heard.</p>'
                + ticks_html(r["window_probs"], "Window") + legend_html())
    return ""


def tint_mask(mask: np.ndarray) -> np.ndarray:
    """Show the mask in chart colours: airway-blue aircraft on chart paper."""
    out = np.empty(mask.shape + (3,), np.uint8)
    out[:] = (0xEE, 0xF1, 0xEA)
    out[mask > 127] = (0x2B, 0x5C, 0x9E)
    return out


EVIDENCE_TEXT = {
    "image": "The aircraft is separated from the sky, then the heatmap shows where the model looked: "
             "red areas mattered most to the answer.",
    "video": "These panels show the frame the model was most sure about. The aircraft is separated from the sky, "
             "then the heatmap shows where the model looked: red areas mattered most.",
    "audio": "The sound is drawn as a spectrogram: time runs left to right, pitch rises upwards. "
             "The heatmap shows which sounds the model listened to: red areas mattered most.",
}


def evidence_html(kind: str | None = None) -> str:
    return f'<p class="caption"><b>The evidence.</b> {EVIDENCE_TEXT.get(kind or "image")}</p>'


def router_html(r: dict | None) -> str:
    if not r:
        return ""
    ro = r["route"]
    how = {"content and extension agree": "its contents and file name agree",
           "recognised from extension only": "its file name (the contents were not conclusive)",
           "unknown extension, recognised from content": "its contents (the file name was unfamiliar)"}.get(
        ro["note"], ro["note"].replace("trusting content", "the contents were trusted"))
    return (f'<p class="router-note"><b>How the file was read.</b> Treated as '
            f'{MODALITY_WORD[r["modality"]].lower()} because {_e(how)}.</p>')


# ------------------------------------------------------------------ app
def _example_files(cfg: Config):
    for d in (cfg.root / "demo_files", Path(__file__).resolve().parents[1] / "demo_files"):
        if d.exists():
            files = sorted(p for p in d.iterdir()
                           if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".mp4", ".wav", ".mp3"))
            if files:
                return [[str(p)] for p in files[:8]]
    return []


def build_demo(cfg: Config):
    import gradio as gr
    from .pipeline import AircraftAgent
    agent = AircraftAgent(cfg)
    theme = gr.themes.Base(font=[gr.themes.GoogleFont("Barlow Semi Condensed"), "Arial Narrow", "sans-serif"])

    def run(file, use_lime):
        hide = gr.update(value=None, visible=False)
        if file is None:
            return (strip_html(), "", "", "", evidence_html(), gr.update(value=None, visible=True), hide, hide,
                    gr.update(value=None, visible=True), hide)
        path = file if isinstance(file, str) else file.name
        try:
            r = agent.predict(path, explain=True, use_lime=bool(use_lime))
        except Exception as e:  # unreadable or unsupported file
            msg = str(e)
            if "Unsupported" in msg:
                msg = ("This file isn't a photo, video or sound recording. "
                       "Try a JPG or PNG image, an MP4 video, or a WAV or MP3 recording.")
            return (strip_html(error=msg), "", "", "", "", hide, hide, hide, hide, hide)
        r["threshold"] = cfg.reject_threshold
        r["audio_weight"] = cfg.fusion_audio_weight
        is_audio = r["modality"] == "audio"
        mask = r.get("mask")
        return (
            strip_html(r),
            bars_html(r["probabilities"], audio_only=is_audio),
            router_html(r),
            breakdown_html(r),
            evidence_html(r["modality"]),
            gr.update(value=r.get("input"), visible=True,
                      label="The recording as a spectrogram" if is_audio else "What you uploaded"),
            gr.update(value=None if mask is None else tint_mask(mask), visible=mask is not None),
            gr.update(value=r.get("cutout"), visible=r.get("cutout") is not None),
            gr.update(value=r.get("gradcam"), visible=r.get("gradcam") is not None,
                      label="Frequencies it listened to" if is_audio else "Where it looked"),
            gr.update(value=r.get("lime"), visible=r.get("lime") is not None),
        )

    with gr.Blocks(title="Aircraft identification", **_blocks_style_kwargs(gr, theme)) as demo:
        gr.HTML('<header id="masthead"><div class="row"><h1>Aircraft<br>identification</h1>'
                '<p>Add a photo, a video clip or a sound recording. The model names the aircraft type and '
                'shows which parts of the input led to its answer.</p></div><div class="airway"></div></header>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=280):
                # No file_types filter on purpose: the router itself checks and rejects other files.
                f = gr.File(label="Photo, video or sound", type="filepath", elem_id="drop")
                lime_box = gr.Checkbox(label="Add a second explanation (LIME). Slower, photos and video only.",
                                       elem_id="lime")
                btn = gr.Button("Identify aircraft", variant="primary", elem_id="identify")
                gr.HTML('<p class="hint">Recognises passenger jets, fighter jets, helicopters and propeller aircraft. '
                        'Video is checked frame by frame; sound is cut into five-second pieces.</p>')
                ex = _example_files(cfg)
                if ex:
                    gr.Examples(ex, inputs=[f], label="Or try a sample file")
            with gr.Column(scale=7, min_width=320):
                strip = gr.HTML(strip_html())
                bars = gr.HTML()
                route = gr.HTML()
        evidence = gr.HTML(evidence_html())
        with gr.Row():
            inp = gr.Image(label="What you uploaded", interactive=False, elem_classes="evidence", height=230)
            msk = gr.Image(label="Where the aircraft is", interactive=False, elem_classes="evidence", height=230, visible=False)
            cut = gr.Image(label="What the model sees", interactive=False, elem_classes="evidence", height=230, visible=False)
            cam = gr.Image(label="Where it looked", interactive=False, elem_classes="evidence", height=230)
            lim = gr.Image(label="Supporting regions (LIME)", interactive=False, elem_classes="evidence",
                           height=230, visible=False)
        breakdown = gr.HTML()
        outs = [strip, bars, route, breakdown, evidence, inp, msk, cut, cam, lim]
        btn.click(run, [f, lime_box], outs)
    demo._aircraft_style = _launch_style_kwargs(gr, theme)
    return demo


def _major(gr) -> int:
    try:
        return int(gr.__version__.split(".")[0])
    except Exception:
        return 4


def _blocks_style_kwargs(gr, theme):
    # Gradio 4/5 take theme and css on Blocks; Gradio 6 moved them to launch().
    return {"theme": theme, "css": CSS} if _major(gr) < 6 else {}


def _launch_style_kwargs(gr, theme):
    return {"theme": theme, "css": CSS} if _major(gr) >= 6 else {}


def launch(cfg: Config, share: bool = True, **kw):
    demo = build_demo(cfg)
    return demo.launch(share=share, **{**demo._aircraft_style, **kw})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./workdir")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    launch(Config.load(Path(a.root)), share=a.share, server_port=a.port)
