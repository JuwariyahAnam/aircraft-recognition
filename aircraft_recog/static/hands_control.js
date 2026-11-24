/* AR hand-control simulation.
 *
 * "Enter simulation" opens a full-screen view: the webcam fills the screen,
 * the identified aircraft floats in front of you, and MediaPipe Hands
 * (bundled in ./mediapipe, runs in the browser; no video leaves the
 * computer) tracks your hands so you can control it:
 *
 *   open palm   the aircraft hovers above your hand; tilt the hand to bank
 *   pinch       grab: move the hand to turn the aircraft
 *   two hands   spread apart to zoom in, bring together to zoom out
 *   fist        hold for about a second to reset the view
 *
 * Hand landmarks (21 per hand): 0 wrist, 4 thumb tip, 8 index tip,
 * 12 middle tip, 16 ring tip, 20 pinky tip, 5/9/13/17 finger knuckles.
 */
(function () {
  "use strict";
  const NAMES = { passenger_jet: "Passenger jet", fighter_jet: "Fighter jet", helicopter: "Helicopter", propeller: "Propeller aircraft" };
  const TIPS = [4, 8, 12, 16, 20];

  const CSS = `
  #sim { position: fixed; inset: 0; z-index: 9999; background: #000; color: #EDEDED; font-family: 'Sora', ui-sans-serif, system-ui, sans-serif;
    display: none; overflow: hidden; }
  #sim.on { display: block; animation: simIn .5s cubic-bezier(.2,.8,.2,1); }
  @keyframes simIn { from { opacity: 0; transform: scale(1.03); } to { opacity: 1; transform: none; } }
  #sim video { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1); filter: saturate(.85) contrast(1.05) brightness(.8); }
  #sim .tint { position: absolute; inset: 0; background: radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,.55) 100%); pointer-events: none; }
  #sim .scan { position: absolute; inset: 0; pointer-events: none; opacity: .18;
    background: repeating-linear-gradient(0deg, rgba(255,255,255,.06) 0 1px, transparent 1px 3px); }
  #sim-stage { position: absolute; inset: 0; }
  #sim-stage canvas { width: 100% !important; height: 100% !important; display: block; }
  #sim-dots { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
  #sim .simframe { position: absolute; inset: 18px; pointer-events: none; }
  #sim .simframe i { position: absolute; width: 26px; height: 26px; border: 2px solid rgba(255,255,255,.7); }
  #sim .simframe i:nth-child(1) { left: 0; top: 0; border-right: none; border-bottom: none; }
  #sim .simframe i:nth-child(2) { right: 0; top: 0; border-left: none; border-bottom: none; }
  #sim .simframe i:nth-child(3) { left: 0; bottom: 0; border-right: none; border-top: none; }
  #sim .simframe i:nth-child(4) { right: 0; bottom: 0; border-left: none; border-top: none; }
  #sim .title { position: absolute; left: 40px; top: 34px; font-size: 13px; letter-spacing: .14em; display: flex; gap: 12px; align-items: center; }
  #sim .title b { width: 8px; height: 8px; border-radius: 50%; background: #FF4D1F; box-shadow: 0 0 10px #FF4D1F; animation: rec 1.4s infinite; }
  @keyframes rec { 50% { opacity: .3; } }
  #sim .exit { position: absolute; right: 40px; top: 28px; background: rgba(10,10,10,.6); color: #EDEDED; border: 1px solid rgba(255,255,255,.18);
    padding: 10px 18px; font: inherit; font-size: 12px; letter-spacing: .12em; cursor: pointer; backdrop-filter: blur(8px); }
  #sim .exit:hover { border-color: #FF4D1F; }
  #sim .exit:focus-visible { outline: 2px solid #FF4D1F; outline-offset: 3px; }
  .simpanel { position: absolute; background: rgba(8,8,9,.62); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,.12); padding: 14px 16px;
    clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%); }
  .simpanel h4 { margin: 0 0 10px; font-weight: 400; font-size: 11px; letter-spacing: .14em; color: #8E8E8E; }
  #sim-tag { min-width: 230px; transform: translate(-50%, -100%); transition: opacity .3s; pointer-events: none; }
  #sim-tag .k { font-size: 10.5px; letter-spacing: .14em; color: #8E8E8E; }
  #sim-tag .v { font-size: 15px; margin: 2px 0 8px; }
  #sim-tag .v.hot { color: #FF4D1F; }
  #sim-tag::after { content: ""; position: absolute; left: 50%; bottom: -34px; width: 1px; height: 32px; background: rgba(255,255,255,.35); }
  #sim-tele { right: 40px; bottom: 96px; width: 250px; }
  .tele { display: grid; grid-template-columns: 64px 1fr 52px; gap: 8px; align-items: center; font-size: 12px; margin: 7px 0; color: #BDBDBD; }
  .tele .bar { height: 3px; background: rgba(255,255,255,.1); position: relative; }
  .tele .bar i { position: absolute; top: 0; bottom: 0; background: #FF4D1F; }
  .tele .n { text-align: right; font-variant-numeric: tabular-nums; color: #EDEDED; }
  #sim-log { right: 40px; top: 84px; width: 250px; max-height: 40vh; overflow: hidden; }
  #sim-log div { font-size: 11.5px; color: #BDBDBD; padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,.06); display: flex; gap: 10px; animation: logIn .35s; }
  #sim-log div span { color: #6f6f6f; font-variant-numeric: tabular-nums; }
  @keyframes logIn { from { opacity: 0; transform: translateX(10px); } }
  #sim-help { left: 40px; bottom: 34px; display: flex; gap: 26px; font-size: 12px; color: #BDBDBD; flex-wrap: wrap; max-width: calc(100% - 360px); }
  #sim-help span b { color: #EDEDED; font-weight: 500; margin-right: 6px; }
  #sim-help span.on b { color: #FF4D1F; }
  #sim-msg { left: 50%; top: 50%; transform: translate(-50%, -50%); max-width: 460px; text-align: center; font-size: 14px; line-height: 1.6; display: none; }
  #sim-msg.show { display: block; }
  #sim-msg .spin { width: 34px; height: 34px; border: 2px solid rgba(255,255,255,.15); border-top-color: #FF4D1F; border-radius: 50%; margin: 0 auto 14px; animation: sp 0.9s linear infinite; }
  @keyframes sp { to { transform: rotate(360deg); } }
  #sim-mode { position: absolute; left: 50%; top: 34px; transform: translateX(-50%); font-size: 12px; letter-spacing: .12em; color: #BDBDBD;
    padding: 8px 16px; border: 1px solid rgba(255,255,255,.14); background: rgba(8,8,9,.5); backdrop-filter: blur(8px); transition: color .3s, border-color .3s; }
  #sim-mode.hot { color: #FF4D1F; border-color: rgba(255,77,31,.6); }
  @media (max-width: 800px) { #sim-log { display: none; } #sim-tele { bottom: 120px; right: 20px; width: 210px; } #sim-help { max-width: calc(100% - 60px); gap: 14px; } }
  @media (prefers-reduced-motion: reduce) { #sim *, #sim { animation: none !important; } }
  `;

  const $ = (sel) => document.querySelector(sel);
  let built = false, stream = null, hands = null, running = false, busy = false;
  let home = null, viewer = null, lastFrame = 0, fps = 0;
  const G = { mode: "search", pinchFrom: null, zoomFrom: null, fistSince: 0, didReset: false, palm: null, roll: 0 };

  function build() {
    if (built) return;
    built = true;
    const st = document.createElement("style"); st.textContent = CSS; document.head.appendChild(st);
    const el = document.createElement("div");
    el.id = "sim";
    el.setAttribute("role", "dialog"); el.setAttribute("aria-label", "Hand-control simulation");
    el.innerHTML = `
      <video playsinline muted></video><div class="tint"></div><div class="scan"></div>
      <div id="sim-stage"></div><canvas id="sim-dots"></canvas>
      <div class="simframe"><i></i><i></i><i></i><i></i></div>
      <div class="title"><b></b>AERO-ID SIMULATION</div>
      <div id="sim-mode">SHOW YOUR HAND</div>
      <button class="exit" type="button">EXIT SIMULATION</button>
      <div class="simpanel" id="sim-tag"><div class="k">AIRCRAFT</div><div class="v" id="sim-cls">Not identified yet</div>
        <div class="k">STATUS</div><div class="v hot" id="sim-status">Waiting for a hand</div></div>
      <div class="simpanel" id="sim-log"><h4>GESTURE LOG</h4></div>
      <div class="simpanel" id="sim-tele"><h4>TELEMETRY</h4>
        ${["Yaw", "Pitch", "Bank", "Zoom", "Hands", "Tracking"].map((k) => `<div class="tele"><span>${k}</span><div class="bar"><i id="b-${k}"></i></div><span class="n" id="n-${k}">-</span></div>`).join("")}
      </div>
      <div class="simpanel" id="sim-help">
        <span data-g="carry"><b>Open palm</b>carry and bank</span><span data-g="grab"><b>Pinch</b>grab and turn</span>
        <span data-g="zoom"><b>Two hands</b>zoom</span><span data-g="reset"><b>Fist</b>reset</span></div>
      <div class="simpanel" id="sim-msg"></div>`;
    document.body.appendChild(el);
    el.querySelector(".exit").addEventListener("click", exit);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && el.classList.contains("on")) exit(); });
  }

  function msg(html) { const m = $("#sim-msg"); m.innerHTML = html; m.classList.toggle("show", !!html); }
  function log(text) {
    const box = $("#sim-log"), d = document.createElement("div");
    const t = new Date(); d.innerHTML = `<span>${t.toTimeString().slice(0, 8)}</span>${text}`;
    box.insertBefore(d, box.children[1] || null);
    while (box.children.length > 9) box.removeChild(box.lastChild);
  }
  function setMode(mode, status) {
    if (mode !== G.mode) {
      const label = { carry: "Carrying the aircraft", grab: "Grabbed: turning", zoom: "Two hands: zooming", reset: "View reset", search: "Hand lost" }[mode];
      if (label) log(label);
    }
    G.mode = mode;
    $("#sim-status").textContent = status;
    $("#sim-mode").textContent = { carry: "OPEN PALM: CARRY", grab: "PINCH: TURNING", zoom: "TWO HANDS: ZOOM", reset: "FIST: RESET", search: "SHOW YOUR HAND" }[mode];
    $("#sim-mode").classList.toggle("hot", mode !== "search");
    document.querySelectorAll("#sim-help span").forEach((s) => s.classList.toggle("on", s.dataset.g === mode));
  }

  // ------------------------------------------------------------- open / close
  async function enter() {
    build();
    viewer = window.AeroViewer && window.AeroViewer.get();
    const el = $("#sim");
    el.classList.add("on");
    document.body.style.overflow = "hidden";
    if (viewer) { home = viewer.host; viewer.attach($("#sim-stage"), true); }
    const cls = viewer && viewer.cls && NAMES[viewer.cls];
    $("#sim-cls").textContent = cls || "Not identified yet";
    if (viewer && !viewer.cls) viewer.setState({ state: "result", cls: "passenger_jet" });
    msg('<div class="spin"></div>Starting the camera…');
    log("Simulation started");
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    } catch (err) {
      msg("<b>The camera is blocked.</b><br>Allow camera access from the icon in the browser's address bar, then press Enter simulation again. " +
          "In Colab, open the public <i>gradio.live</i> link in its own tab; the preview inside the notebook can't use the camera.");
      log("Camera unavailable");
      return;
    }
    const video = el.querySelector("video");
    video.srcObject = stream;
    await video.play().catch(() => {});
    msg('<div class="spin"></div>Loading hand tracking…');
    try { await loadHands(); } catch (err) {
      msg("<b>Hand tracking couldn't load.</b><br>Reload the page and try again."); console.error(err); return;
    }
    msg("");
    setMode("search", "Waiting for a hand");
    running = true;
    requestAnimationFrame(tick);
  }

  function exit() {
    running = false;
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    const el = $("#sim");
    el.classList.remove("on");
    document.body.style.overflow = "";
    if (viewer && home) { viewer.handActive = false; viewer.attach(home, false); }
    G.mode = "search"; G.pinchFrom = G.zoomFrom = null;
  }

  function loadHands() {
    if (hands) return Promise.resolve();
    const base = window.__AERO_BASE + "/mediapipe";
    return new Promise((resolve, reject) => {
      const go = () => {
        hands = new window.Hands({ locateFile: (f) => `${base}/${f}` });
        hands.setOptions({ maxNumHands: 2, modelComplexity: 0, minDetectionConfidence: 0.6, minTrackingConfidence: 0.5 });
        hands.onResults(onResults);
        hands.initialize().then(resolve, reject);
      };
      if (window.Hands) return go();
      const s = document.createElement("script");
      s.src = `${base}/hands.js`; s.onload = go; s.onerror = reject;
      document.head.appendChild(s);
    });
  }

  async function tick() {
    if (!running) return;
    const video = $("#sim video");
    if (!busy && video.readyState >= 2) {
      busy = true;
      try { await hands.send({ image: video }); } catch (e) { console.error(e); }
      busy = false;
    }
    requestAnimationFrame(tick);
  }

  // ------------------------------------------------------------- geometry
  // Convert a landmark (normalised to the camera image) into screen pixels,
  // accounting for the mirrored, cover-fitted video.
  function toScreen(p) {
    const v = $("#sim video"), W = innerWidth, H = innerHeight;
    const va = (v.videoWidth || 16) / (v.videoHeight || 9), sa = W / H;
    let x, y;
    if (sa > va) { const h = W / va; x = (1 - p.x) * W; y = p.y * h + (H - h) / 2; }
    else { const w = H * va; x = (1 - p.x) * w + (W - w) / 2; y = p.y * H; }
    return { x, y };
  }
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const ndc = (s) => [(s.x / innerWidth) * 2 - 1, -(s.y / innerHeight) * 2 + 1];

  function analyse(lm) {
    const P = lm.map(toScreen);
    const size = dist(P[0], P[9]) || 1;                       // wrist to middle knuckle
    const palm = [0, 5, 9, 13, 17].reduce((a, i) => ({ x: a.x + P[i].x / 5, y: a.y + P[i].y / 5 }), { x: 0, y: 0 });
    const pinchD = dist(P[4], P[8]) / size;
    const curled = [8, 12, 16, 20].filter((i) => dist(P[i], P[0]) < dist(P[i - 3], P[0]) * 1.12).length;
    const roll = Math.atan2(P[17].y - P[5].y, P[17].x - P[5].x);  // knuckle line angle
    return { P, size, palm, pinchD, fist: curled >= 4, pinch: pinchD < (G.mode === "grab" ? 0.5 : 0.33), roll, pinchAt: { x: (P[4].x + P[8].x) / 2, y: (P[4].y + P[8].y) / 2 } };
  }

  // ------------------------------------------------------------- gestures
  function onResults(res) {
    const now = performance.now();
    if (lastFrame) fps = fps * 0.9 + (1000 / Math.max(1, now - lastFrame)) * 0.1;
    lastFrame = now;
    if (now < G.testUntil) return;                            // test input is being fed
    const list = (res.multiHandLandmarks || []).map(analyse);
    handle(list, now);
    draw(list);
  }

  function handle(list, now) {
    if (!viewer) return;
    viewer.handActive = list.length > 0;
    if (list.length >= 2) {                                    // two hands: zoom
      const a = list[0].palm, b = list[1].palm, d = dist(a, b);
      if (!G.zoomFrom) G.zoomFrom = { d, z: viewer.zoom };
      viewer.setZoom(G.zoomFrom.z * (G.zoomFrom.d / Math.max(40, d)));
      viewer.anchorTo(ndc({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 - 40 }));
      G.pinchFrom = null;
      setMode("zoom", `Zoom ${(1 / viewer.zoom).toFixed(2)}×`);
    } else if (list.length === 1) {
      G.zoomFrom = null;
      const h = list[0];
      G.palm = G.palm ? { x: G.palm.x * 0.6 + h.palm.x * 0.4, y: G.palm.y * 0.6 + h.palm.y * 0.4 } : h.palm;
      viewer.anchorTo(ndc({ x: G.palm.x, y: G.palm.y - h.size * 1.6 }));     // hover above the hand
      if (h.fist && !h.pinch) {
        G.pinchFrom = null;
        if (!G.fistSince) G.fistSince = now;
        if (now - G.fistSince > 850 && !G.didReset) { viewer.resetView(); G.didReset = true; }
        setMode("reset", G.didReset ? "View reset" : "Hold the fist to reset");
      } else {
        G.fistSince = 0; G.didReset = false;
        if (h.pinch) {
          if (G.pinchFrom) viewer.grab((h.pinchAt.x - G.pinchFrom.x) * 0.012, (h.pinchAt.y - G.pinchFrom.y) * 0.006);
          G.pinchFrom = h.pinchAt;
          viewer.setBank(0);
          setMode("grab", "Grabbed: move to turn");
        } else {
          G.pinchFrom = null;
          let r = h.roll; if (r > Math.PI / 2) r -= Math.PI; if (r < -Math.PI / 2) r += Math.PI;
          viewer.setBank(-r * 1.2);
          setMode("carry", "Carrying: tilt to bank");
        }
      }
    } else {
      G.pinchFrom = G.zoomFrom = null; G.palm = null; G.fistSince = 0;
      viewer.anchorTo(null); viewer.setBank(0);
      setMode("search", "Waiting for a hand");
    }
    telemetry(list.length);
  }

  function telemetry(n) {
    const deg = (r) => ((((r * 180) / Math.PI) % 360) + 360) % 360;
    const vals = {
      Yaw: [deg(viewer.yaw), deg(viewer.yaw) / 360, `${deg(viewer.yaw).toFixed(0)}°`],
      Pitch: [viewer.pitch, (viewer.pitch + 0.9) / 2.05, `${((viewer.pitch * 180) / Math.PI).toFixed(0)}°`],
      Bank: [viewer.bank, (viewer.bank + 0.9) / 1.8, `${((viewer.bank * 180) / Math.PI).toFixed(0)}°`],
      Zoom: [viewer.zoom, (1.6 - viewer.zoom) / 1.0, `${(1 / viewer.zoom).toFixed(2)}×`],
      Hands: [n, n / 2, `${n}`],
      Tracking: [fps, Math.min(1, fps / 30), `${fps.toFixed(0)} fps`],
    };
    Object.entries(vals).forEach(([k, [, frac, txt]]) => {
      const f = Math.max(0, Math.min(1, frac));
      const b = document.getElementById("b-" + k);
      if (k === "Bank" || k === "Pitch") { b.style.left = `${Math.min(50, f * 100)}%`; b.style.width = `${Math.abs(f - 0.5) * 100}%`; }
      else { b.style.left = "0"; b.style.width = `${f * 100}%`; }
      document.getElementById("n-" + k).textContent = txt;
    });
    // keep the status tag just above the aircraft
    const v = viewer.anchor.clone().add({ x: 0, y: 1.25, z: 0 }).project(viewer.camera);
    const tag = $("#sim-tag");
    tag.style.left = `${Math.max(140, Math.min(innerWidth - 140, ((v.x + 1) / 2) * innerWidth))}px`;
    tag.style.top = `${Math.max(250, ((1 - v.y) / 2) * innerHeight)}px`;
  }

  // Joint dots only (no lines between the hand and the aircraft).
  function draw(list) {
    const c = $("#sim-dots"), dpr = Math.min(devicePixelRatio || 1, 2);
    if (c.width !== innerWidth * dpr) { c.width = innerWidth * dpr; c.height = innerHeight * dpr; }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, innerWidth, innerHeight);
    list.forEach((h) => {
      h.P.forEach((p, i) => {
        const tip = TIPS.includes(i);
        g.beginPath(); g.arc(p.x, p.y, tip ? 4.5 : 2.5, 0, Math.PI * 2);
        g.fillStyle = tip ? "#FF4D1F" : "rgba(255,255,255,.85)"; g.fill();
        if (tip) { g.beginPath(); g.arc(p.x, p.y, 9, 0, Math.PI * 2); g.strokeStyle = "rgba(255,77,31,.45)"; g.lineWidth = 1; g.stroke(); }
      });
      if (h.pinch) {                                           // pinch ring
        g.beginPath(); g.arc(h.pinchAt.x, h.pinchAt.y, 16, 0, Math.PI * 2);
        g.strokeStyle = "#FF4D1F"; g.lineWidth = 2; g.stroke();
      }
      g.beginPath(); g.arc(h.palm.x, h.palm.y, h.size * 0.45, 0, Math.PI * 2);   // palm reticle
      g.strokeStyle = "rgba(255,255,255,.35)"; g.lineWidth = 1; g.setLineDash([4, 6]); g.stroke(); g.setLineDash([]);
    });
  }

  // Button on the 3D stage.
  document.addEventListener("click", (e) => { if (e.target.closest && e.target.closest("#sim-enter")) enter(); });
  // For testing without a camera: feed fake landmark lists through the same gesture logic.
  window.AeroSim = { enter, exit, _feed: (handsLm) => { G.testUntil = performance.now() + 2500; const list = handsLm.map(analyse); handle(list, performance.now()); draw(list); } };
})();
