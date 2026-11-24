/* In-browser recognition for the project website.
 *
 * Runs the trained networks (exported to ONNX by export_web.py) with ONNX
 * Runtime Web, following the same steps as the Python agent (pipeline.py):
 *
 *   route  -> image / video / audio (by file contents, then extension)
 *   image  -> U-Net mask -> cut-out -> image CNN -> CAM heatmap -> exact model
 *   video  -> 16 frames, each classified, majority vote; soundtrack fused in
 *   audio  -> 5-second mel-spectrograms (librosa-compatible) -> audio CNN
 *
 * Exposes window.AeroInfer = { init(base, modelsDir), available(), analyse(file, onProgress) }.
 */
(function () {
  "use strict";

  let ort = null, man = null, base = "", dir = "", ready = null;
  const sessions = {};

  // ================================================================ loading
  function init(staticBase, modelsDir) {
    base = staticBase; dir = modelsDir;
    if (ready) return ready;
    ready = (async () => {
      const r = await fetch(`${dir}/manifest.json`, { cache: "no-cache" });
      if (!r.ok) return null;
      man = await r.json();
      await loadScript(`${base}/ort/ort.wasm.min.js`);
      ort = window.ort;
      ort.env.wasm.wasmPaths = new URL(`${base}/ort/`, location.href).href;
      ort.env.wasm.numThreads = 1;          // GitHub Pages has no cross-origin isolation
      return man;
    })().catch((e) => { console.warn("recognition unavailable", e); return null; });
    return ready;
  }
  function loadScript(src) {
    return new Promise((res, rej) => {
      if (window.ort) return res();
      const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = rej; document.head.appendChild(s);
    });
  }
  async function session(name) {
    if (!man.models[name]) return null;
    if (!sessions[name]) sessions[name] = ort.InferenceSession.create(`${dir}/${man.models[name].file}`,
      { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    return sessions[name];
  }

  // ================================================================ routing
  const EXT = { image: ["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"], video: ["mp4", "avi", "mov", "mkv", "webm", "m4v"],
                audio: ["wav", "mp3", "flac", "ogg", "m4a", "aac"] };
  async function route(file) {
    const h = new Uint8Array(await file.slice(0, 64).arrayBuffer());
    const s = (a, b) => String.fromCharCode(...h.slice(a, b));
    let c = null;
    if (h[0] === 0xff && h[1] === 0xd8 && h[2] === 0xff) c = "image";
    else if (s(0, 8) === "\x89PNG\r\n\x1a\n") c = "image";
    else if (s(0, 2) === "BM") c = "image";
    else if (s(0, 4) === "RIFF" && s(8, 12) === "WEBP") c = "image";
    else if (s(0, 4) === "RIFF" && s(8, 12) === "WAVE") c = "audio";
    else if (s(0, 4) === "fLaC" || s(0, 3) === "ID3" || s(0, 4) === "OggS") c = "audio";
    else if (h[0] === 0xff && (h[1] & 0xe0) === 0xe0) c = "audio";
    else if (s(0, 4) === "RIFF" && s(8, 12) === "AVI ") c = "video";
    else if (h[0] === 0x1a && h[1] === 0x45 && h[2] === 0xdf && h[3] === 0xa3) c = "video";
    else if (s(4, 8) === "ftyp") c = ["M4A ", "M4B ", "M4P "].includes(s(8, 12)) ? "audio" : "video";
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    const e = Object.keys(EXT).find((k) => EXT[k].includes(ext)) || null;
    let note = "content and extension agree";
    if (c && e && c !== e) note = `extension says ${e}, content says ${c}: trusting content`;
    else if (c && !e) note = "unknown extension, recognised from content";
    else if (!c && e) note = "recognised from extension only";
    return { modality: c || e, by_content: c, by_extension: e, note };
  }

  // ================================================================ images
  function canvas(w, h) { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; }
  function drawTo(src, w, h, smooth = true) {
    const c = canvas(w, h), g = c.getContext("2d");
    g.imageSmoothingEnabled = smooth; g.imageSmoothingQuality = "high";
    g.drawImage(src, 0, 0, w, h);
    return c;
  }
  function tensor(src, size) {
    const d = drawTo(src, size, size).getContext("2d").getImageData(0, 0, size, size).data;
    const n = size * size, out = new Float32Array(3 * n), M = man.mean, S = man.std;
    for (let i = 0; i < n; i++)
      for (let ch = 0; ch < 3; ch++) out[ch * n + i] = (d[i * 4 + ch] / 255 - M[ch]) / S[ch];
    return new ort.Tensor("float32", out, [1, 3, size, size]);
  }
  async function fileToCanvas(file, maxSide = 1600) {
    const bmp = await createImageBitmap(file);
    const s = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
    return drawTo(bmp, Math.round(bmp.width * s), Math.round(bmp.height * s));
  }

  // U-Net: mask (Uint8Array, 0/255) at the image's size
  async function segment(img) {
    const sess = await session("unet");
    if (!sess) return null;
    const S = man.seg_size;
    const out = await sess.run({ input: tensor(img, S) });
    const logits = out.logits.data;
    const small = canvas(S, S), g = small.getContext("2d"), id = g.createImageData(S, S);
    for (let i = 0; i < S * S; i++) { const v = logits[i] > 0 ? 255 : 0; id.data[i * 4] = id.data[i * 4 + 1] = id.data[i * 4 + 2] = v; id.data[i * 4 + 3] = 255; }
    g.putImageData(id, 0, 0);
    const big = drawTo(small, img.width, img.height, false).getContext("2d").getImageData(0, 0, img.width, img.height).data;
    const m = new Uint8Array(img.width * img.height);
    for (let i = 0; i < m.length; i++) m[i] = big[i * 4] > 127 ? 255 : 0;
    return m;
  }

  // Same rules as datasets.apply_mask: "mask", "mask_crop", or "crop" (keep background)
  function applyMask(img, mask, mode, minArea = 0.005, margin = 0.12) {
    if (!mask || mode === "none") return img;
    const W = img.width, H = img.height;
    let n = 0, x1 = W, y1 = H, x2 = 0, y2 = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (mask[y * W + x]) {
      n++; if (x < x1) x1 = x; if (x > x2) x2 = x; if (y < y1) y1 = y; if (y > y2) y2 = y;
    }
    if (n / (W * H) < minArea) return img;
    let out = drawTo(img, W, H);
    if (mode !== "crop") {
      const g = out.getContext("2d"), d = g.getImageData(0, 0, W, H), bg = man.mean.map((v) => Math.round(v * 255));
      for (let i = 0; i < W * H; i++) if (!mask[i]) { d.data[i * 4] = bg[0]; d.data[i * 4 + 1] = bg[1]; d.data[i * 4 + 2] = bg[2]; }
      g.putImageData(d, 0, 0);
    }
    if (mode === "mask_crop" || mode === "crop") {
      let side = Math.floor(Math.max(y2 - y1, x2 - x1) * (1 + 2 * margin)) + 1;
      side = Math.min(side, Math.max(H, W));
      const cy = Math.floor((y1 + y2) / 2), cx = Math.floor((x1 + x2) / 2);
      const y0 = Math.max(0, cy - Math.floor(side / 2)), x0 = Math.max(0, cx - Math.floor(side / 2));
      const y3 = Math.min(H, y0 + side), x3 = Math.min(W, x0 + side);
      const c = canvas(x3 - x0, y3 - y0);
      c.getContext("2d").drawImage(out, x0, y0, x3 - x0, y3 - y0, 0, 0, x3 - x0, y3 - y0);
      out = c;
    }
    return out;
  }
  function maskCanvas(mask, W, H) {
    const c = canvas(W, H), g = c.getContext("2d"), d = g.createImageData(W, H);
    for (let i = 0; i < W * H; i++) {
      const on = mask[i] > 0;
      d.data[i * 4] = on ? 255 : 13; d.data[i * 4 + 1] = on ? 77 : 13; d.data[i * 4 + 2] = on ? 31 : 14; d.data[i * 4 + 3] = 255;
    }
    g.putImageData(d, 0, 0);
    return c;
  }

  const softmax = (a) => { const m = Math.max(...a), e = a.map((v) => Math.exp(v - m)), s = e.reduce((x, y) => x + y, 0); return e.map((v) => v / s); };

  // CNN with CAM: returns {probs, cam (Float32Array h*w normalised), h, w}
  async function classify(name, img, withCam) {
    const sess = await session(name);
    const out = await sess.run({ input: tensor(img, man.cls_size) });
    const probs = softmax(Array.from(out.logits.data));
    if (!withCam) return { probs };
    const F = out.features, [, K, h, w] = F.dims, f = F.data;
    const c = probs.indexOf(Math.max(...probs)), Wc = man.models[name].cam_weights[c];
    const cam = new Float32Array(h * w);
    for (let k = 0; k < K; k++) { const wk = Wc[k], o = k * h * w; for (let i = 0; i < h * w; i++) cam[i] += wk * f[o + i]; }
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < cam.length; i++) { cam[i] = Math.max(0, cam[i]); mn = Math.min(mn, cam[i]); mx = Math.max(mx, cam[i]); }
    for (let i = 0; i < cam.length; i++) cam[i] = (cam[i] - mn) / (mx - mn + 1e-8);
    return { probs, cam, h, w };
  }
  function jet(v) {   // OpenCV COLORMAP_JET
    const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 3))), g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 2))),
          b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 1)));
    return [r * 255, g * 255, b * 255];
  }
  function overlay(img, cam, h, w, outW, outH, alpha = 0.45) {
    const base = drawTo(img, outW, outH), g = base.getContext("2d"), d = g.getImageData(0, 0, outW, outH);
    const camC = canvas(w, h), cg = camC.getContext("2d"), cd = cg.createImageData(w, h);
    for (let i = 0; i < w * h; i++) { const v = Math.round(cam[i] * 255); cd.data[i * 4] = cd.data[i * 4 + 1] = cd.data[i * 4 + 2] = v; cd.data[i * 4 + 3] = 255; }
    cg.putImageData(cd, 0, 0);
    const up = drawTo(camC, outW, outH).getContext("2d").getImageData(0, 0, outW, outH).data;
    for (let i = 0; i < outW * outH; i++) {
      const [r, gg, b] = jet(up[i * 4] / 255);
      d.data[i * 4] = (1 - alpha) * d.data[i * 4] + alpha * r;
      d.data[i * 4 + 1] = (1 - alpha) * d.data[i * 4 + 1] + alpha * gg;
      d.data[i * 4 + 2] = (1 - alpha) * d.data[i * 4 + 2] + alpha * b;
    }
    g.putImageData(d, 0, 0);
    return base;
  }

  // ================================================================ exact model
  const norm = (s) => String(s).toUpperCase().replace(/[^A-Z0-9]/g, "");
  function modelFor(name) {
    const reg = window.__AERO_MODELS || {}, n = norm(name);
    return Object.keys(reg).find((k) => (reg[k].matches || []).some((m) => norm(m) === n || (norm(m).length >= 3 && n.includes(norm(m))))) || null;
  }
  async function whichModel(img, mask, res) {
    res.subtypes = []; res.subtype = null; res.model_key = null;
    const info = man.models.subtype;
    if (!info || res.rejected) return;
    const sess = await session("subtype");
    const crop = mask ? applyMask(img, mask, "crop") : img;
    const out = await sess.run({ input: tensor(crop, info.size || man.cls_size) });
    let p = softmax(Array.from(out.logits.data));
    const ci = man.classes.indexOf(res.class);
    if (info.parents.some((x) => x === res.class)) {
      p = p.map((v, i) => (info.parents[i] === res.class ? v : 0));
      const s = p.reduce((a, b) => a + b, 0) || 1; p = p.map((v) => v / s);
    }
    res.subtypes = p.map((v, i) => [info.names[i], v]).filter((x) => x[1] > 0).sort((a, b) => b[1] - a[1]).slice(0, 3);
    if (res.subtypes.length && res.subtypes[0][1] >= 0.25) { res.subtype = res.subtypes[0][0]; res.model_key = modelFor(res.subtype); }
    void ci;
  }

  function verdict(res, probs, note) {
    const i = probs.indexOf(Math.max(...probs)), conf = probs[i], cls = man.classes[i], label = man.pretty[cls];
    res.probs = probs; res.class = cls; res.label = label; res.confidence = conf;
    res.rejected = conf < man.reject_threshold;
    res.summary = (res.rejected ? `Not sure: possibly not an aircraft (best guess ${label}, ${Math.round(conf * 100)}%)` : `${label} (${Math.round(conf * 100)}% confident)`) + (note ? `. ${note}` : "");
    res.probabilities = Object.fromEntries(man.classes.map((c, k) => [man.pretty[c], probs[k]]));
  }

  async function analyseImageCanvas(img, withCam) {
    const mask = await segment(img);
    const cut = applyMask(img, mask, mask ? man.mask_mode : "none");
    const r = await classify("image", cut, withCam);
    let note = "";
    if (mask) { let n = 0; for (let i = 0; i < mask.length; i++) if (mask[i]) n++; if (n / mask.length < 0.005) note = "No aircraft-shaped object was segmented"; }
    const out = { input: img, mask: mask ? maskCanvas(mask, img.width, img.height) : null, maskRaw: mask, cutout: cut, probs: r.probs, note };
    if (withCam) { const S = 640; out.gradcam = overlay(cut, r.cam, r.h, r.w, S, S); out.cutoutSquare = drawTo(cut, S, S); }
    return out;
  }

  // ================================================================ audio
  let melCache = null;
  function hzToMel(f) { const fsp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fsp, step = Math.log(6.4) / 27; return f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / step : f / fsp; }
  function melToHz(m) { const fsp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fsp, step = Math.log(6.4) / 27; return m >= minLogMel ? minLogHz * Math.exp(step * (m - minLogMel)) : fsp * m; }
  function melFilters(sr, nfft, nmels) {   // librosa.filters.mel(htk=False, norm="slaney")
    const nb = nfft / 2 + 1, fft = Array.from({ length: nb }, (_, i) => (i * sr) / nfft);
    const lo = hzToMel(0), hi = hzToMel(sr / 2);
    const mels = Array.from({ length: nmels + 2 }, (_, i) => melToHz(lo + ((hi - lo) * i) / (nmels + 1)));
    const W = [];
    for (let i = 0; i < nmels; i++) {
      const row = new Float32Array(nb), d0 = mels[i + 1] - mels[i], d1 = mels[i + 2] - mels[i + 1], en = 2 / (mels[i + 2] - mels[i]);
      for (let j = 0; j < nb; j++) row[j] = Math.max(0, Math.min((fft[j] - mels[i]) / d0, (mels[i + 2] - fft[j]) / d1)) * en;
      W.push(row);
    }
    return W;
  }
  function fftMag2(re, im) {    // in-place radix-2 FFT; returns |X|^2 for bins 0..N/2
    const N = re.length;
    for (let i = 1, j = 0; i < N; i++) { let bit = N >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
    for (let len = 2; len <= N; len <<= 1) {
      const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < N; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2, tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
        }
      }
    }
    const out = new Float32Array(N / 2 + 1);
    for (let k = 0; k <= N / 2; k++) out[k] = re[k] * re[k] + im[k] * im[k];
    return out;
  }
  // librosa.power_to_db(melspectrogram(y, center=True, pad_mode="constant"), ref=np.max) -> uint8 image (flipped)
  function melImage(y) {
    const a = man.audio, N = a.n_fft, hop = a.hop, M = a.n_mels;
    if (!melCache) melCache = { W: melFilters(a.sr, N, M), win: Float32Array.from({ length: N }, (_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N)) };
    const pad = N / 2, x = new Float32Array(y.length + 2 * pad); x.set(y, pad);
    const T = 1 + Math.floor((x.length - N) / hop), S = new Float32Array(M * T);
    const re = new Float32Array(N), im = new Float32Array(N);
    for (let t = 0; t < T; t++) {
      for (let n = 0; n < N; n++) { re[n] = x[t * hop + n] * melCache.win[n]; im[n] = 0; }
      const p = fftMag2(re, im);
      for (let m = 0; m < M; m++) { let s = 0; const w = melCache.W[m]; for (let k = 0; k < p.length; k++) s += w[k] * p[k]; S[m * T + t] = s; }
    }
    let ref = 0; for (let i = 0; i < S.length; i++) ref = Math.max(ref, S[i]);
    const refDb = 10 * Math.log10(Math.max(1e-10, ref)), top = a.top_db;
    const c = canvas(T, M), g = c.getContext("2d"), d = g.createImageData(T, M);
    let mx = -Infinity; const db = new Float32Array(S.length);
    for (let i = 0; i < S.length; i++) { db[i] = 10 * Math.log10(Math.max(1e-10, S[i])) - refDb; mx = Math.max(mx, db[i]); }
    for (let m = 0; m < M; m++) for (let t = 0; t < T; t++) {
      let v = Math.max(db[m * T + t], mx - top);            // top_db
      v = Math.min(0, Math.max(-top, v));                    // spectrogram_to_image clip
      const u = Math.floor(((v + top) / top) * 255), o = ((M - 1 - m) * T + t) * 4;   // flipud
      d.data[o] = d.data[o + 1] = d.data[o + 2] = u; d.data[o + 3] = 255;
    }
    g.putImageData(d, 0, 0);
    return c;
  }
  async function decodeAudio(file) {
    const buf = await file.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    let audio;
    try { audio = await ctx.decodeAudioData(buf.slice(0)); } finally { ctx.close && ctx.close(); }
    const sr = man.audio.sr, len = Math.max(1, Math.round(audio.duration * sr));
    const off = new OfflineAudioContext(1, len, sr), src = off.createBufferSource();
    src.buffer = audio; src.connect(off.destination); src.start();
    return (await off.startRendering()).getChannelData(0);
  }
  function windows(y) {
    const a = man.audio, win = Math.round(a.window_s * a.sr);
    if (y.length < win) { const z = new Float32Array(win); z.set(y); y = z; }
    const out = [];
    for (let s = 0; s + win <= y.length; s += win) {
      const seg = y.subarray(s, s + win); let e = 0; for (let i = 0; i < seg.length; i++) e += seg[i] * seg[i];
      if (Math.sqrt(e / seg.length) >= 1e-4) out.push(seg);
    }
    return out;
  }
  function magma(v) {   // compact magma colormap
    const st = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 253, 191]];
    const x = v * (st.length - 1), i = Math.min(st.length - 2, Math.floor(x)), f = x - i;
    return st[i].map((c, k) => c + f * (st[i + 1][k] - c));
  }
  function colourise(gray, outW, outH) {
    const c = drawTo(gray, outW, outH), g = c.getContext("2d"), d = g.getImageData(0, 0, outW, outH);
    for (let i = 0; i < outW * outH; i++) { const [r, gg, b] = magma(d.data[i * 4] / 255); d.data[i * 4] = r; d.data[i * 4 + 1] = gg; d.data[i * 4 + 2] = b; }
    g.putImageData(d, 0, 0);
    return c;
  }
  async function analyseAudioSamples(y, withCam) {
    const ws = windows(y);
    if (!ws.length) throw new Error("The recording is empty or silent.");
    const specs = ws.map(melImage), all = [];
    for (const s of specs) all.push((await classify("audio", s, false)).probs);
    const probs = man.classes.map((_, k) => all.reduce((a, p) => a + p[k], 0) / all.length);
    let best = 0, bv = -1; all.forEach((p, i) => { const m = Math.max(...p); if (m > bv) { bv = m; best = i; } });
    const S = man.cls_size * 2;
    const shown = colourise(specs[best], S * 2, S);
    const out = { input: shown, probs, n_windows: specs.length, window_probs: all };
    if (withCam) { const r = await classify("audio", specs[best], true); out.gradcam = overlay(shown, r.cam, r.h, r.w, shown.width, shown.height); }
    return out;
  }

  // ================================================================ video
  function videoFrames(file, n, onProgress) {
    return new Promise((resolve, reject) => {
      const v = document.createElement("video"), url = URL.createObjectURL(file);
      v.muted = true; v.playsInline = true; v.preload = "auto"; v.src = url;
      v.onerror = () => { URL.revokeObjectURL(url); reject(new Error("This video format can't be read by the browser. Try an MP4 (H.264).")); };
      v.onloadeddata = async () => {
        const dur = isFinite(v.duration) ? v.duration : 0, frames = [];
        const times = dur > 0 ? Array.from({ length: n }, (_, i) => (dur * (i + 0.5)) / n) : [0];
        const maxSide = 960, s = Math.min(1, maxSide / Math.max(v.videoWidth, v.videoHeight));
        for (let i = 0; i < times.length; i++) {
          await new Promise((ok) => { v.onseeked = ok; v.currentTime = Math.min(times[i], Math.max(0, dur - 0.05)); });
          frames.push(drawTo(v, Math.round(v.videoWidth * s), Math.round(v.videoHeight * s)));
          onProgress && onProgress(`Reading frame ${i + 1} of ${times.length}…`);
        }
        URL.revokeObjectURL(url);
        resolve(frames);
      };
    });
  }

  // ================================================================ main
  async function analyse(file, onProgress = () => {}) {
    if (!(await init(base, dir))) throw new Error("The trained models aren't on the website yet.");
    const ro = await route(file);
    if (!ro.modality) throw new Error("This isn’t a photo, video or sound recording. Try a JPG or PNG image, an MP4 video, or a WAV or MP3 recording.");
    const res = { route: ro, modality: ro.modality };
    onProgress("Loading the recognition models (first time only)…");
    if (ro.modality === "image") {
      if (!man.models.image) throw new Error("No image model was exported.");
      await Promise.all([session("unet"), session("image")]);
      onProgress("Finding the aircraft…");
      const img = await fileToCanvas(file);
      const a = await analyseImageCanvas(img, true);
      Object.assign(res, a);
      verdict(res, a.probs, a.note);
      onProgress("Checking which aircraft model…");
      await whichModel(img, a.maskRaw, res);
    } else if (ro.modality === "audio") {
      if (!man.models.audio) throw new Error("No sound model was exported.");
      await session("audio");
      onProgress("Turning the sound into spectrograms…");
      const y = await decodeAudio(file);
      const a = await analyseAudioSamples(y, true);
      Object.assign(res, a);
      verdict(res, a.probs, `Averaged over ${a.n_windows} five-second windows`);
      res.subtypes = []; res.model_key = null;
    } else {
      if (!man.models.image) throw new Error("No image model was exported.");
      await Promise.all([session("unet"), session("image")]);
      const frames = await videoFrames(file, 16, onProgress);
      if (!frames.length) throw new Error("Could not read any frames from the video.");
      const fp = [];
      for (let i = 0; i < frames.length; i++) { onProgress(`Checking frame ${i + 1} of ${frames.length}…`); fp.push((await analyseImageCanvas(frames[i], false)).probs); }
      const votes = man.classes.map(() => 0); fp.forEach((p) => votes[p.indexOf(Math.max(...p))]++);
      const imageProbs = man.classes.map((_, k) => fp.reduce((a, p) => a + p[k], 0) / fp.length);
      let best = 0, bv = -1; fp.forEach((p, i) => { const m = Math.max(...p); if (m > bv) { bv = m; best = i; } });
      onProgress("Explaining the clearest frame…");
      const shown = await analyseImageCanvas(frames[best], true);
      Object.assign(res, shown);
      let probs = imageProbs;
      let note = `Majority vote over ${frames.length} frames: ` + man.classes.map((c, k) => (votes[k] ? `${man.pretty[c]} ${votes[k]}` : "")).filter(Boolean).join(", ");
      if (man.models.audio) {
        try {
          onProgress("Listening to the soundtrack…");
          const y = await decodeAudio(file);
          const au = await analyseAudioSamples(y, false), w = man.fusion_audio_weight;
          probs = imageProbs.map((p, k) => (1 - w) * p + w * au.probs[k]);
          res.audio_probs = au.probs; note += `. Fused with the soundtrack (audio weight ${w})`;
        } catch (e) { /* no soundtrack */ }
      }
      Object.assign(res, { image_probs: imageProbs, frame_probs: fp, votes, n_frames: frames.length,
        frame_thumbs: frames.map((f) => drawTo(f, 160, Math.round((160 * f.height) / f.width))) });
      verdict(res, probs, note);
      onProgress("Checking which aircraft model…");
      await whichModel(frames[best], shown.maskRaw, res);
    }
    return res;
  }

  window.AeroInfer = {
    init, analyse, route,
    available: async () => !!(await init(base, dir)),
    manifest: () => man,
    _melImage: (y) => melImage(y),   // for testing against librosa
  };
})();
