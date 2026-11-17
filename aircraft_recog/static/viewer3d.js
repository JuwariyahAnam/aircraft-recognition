/* 3D aircraft viewer for the demo.
 *
 * Four original low-poly aircraft, built from three.js primitives (no
 * downloaded model files): passenger jet, fighter jet, helicopter, propeller
 * aircraft. Drag to rotate 360 degrees, scroll or pinch to zoom, double-click
 * to reset.
 *
 * States (set from Python through a hidden <div class="vstate">):
 *   idle     - faint wireframe "hologram" cycling through the four types
 *   loading  - fast scanning wireframe
 *   result   - the identified aircraft, solid, with a fly-in
 *   low      - identified but unconfirmed: amber wireframe
 *   error    - empty stage
 */
(function () {
  "use strict";
  const T = window.THREE;
  if (!T) { console.error("three.js not loaded"); return; }

  const ACCENT = 0xff4d1f;
  const AMBER = 0xffb020;
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------- materials
  const M = {
    body: () => new T.MeshStandardMaterial({ color: 0x9aa0aa, metalness: 0.45, roughness: 0.34 }),
    dark: () => new T.MeshStandardMaterial({ color: 0x2a2c31, metalness: 0.55, roughness: 0.45 }),
    glass: () => new T.MeshStandardMaterial({ color: 0x0c1118, metalness: 0.95, roughness: 0.06, emissive: 0x0a1a2a, emissiveIntensity: 0.35 }),
    blade: () => new T.MeshStandardMaterial({ color: 0x1d1e22, metalness: 0.4, roughness: 0.6 }),
    glow: (c) => new T.MeshBasicMaterial({ color: c }),
  };

  // ------------------------------------------------------------- helpers
  // Body of revolution along +X (nose at +X). profile: [[radius, x], ...] from tail to nose.
  function lathe(profile, mat, seg = 28) {
    const pts = profile.map(([r, x]) => new T.Vector2(Math.max(r, 0.0001), x));
    const g = new T.LatheGeometry(pts, seg);
    g.rotateZ(-Math.PI / 2);
    return new T.Mesh(g, mat);
  }
  // Flat surface in the X-Z plane from an outline [[x, z], ...].
  function plate(outline, thick, mat) {
    const s = new T.Shape();
    outline.forEach(([x, z], i) => (i ? s.lineTo(x, -z) : s.moveTo(x, -z)));
    const g = new T.ExtrudeGeometry(s, { depth: thick, bevelEnabled: true, bevelThickness: thick * 0.4, bevelSize: thick * 0.4, bevelSegments: 1 });
    g.rotateX(-Math.PI / 2);
    g.translate(0, -thick / 2, 0);
    return new T.Mesh(g, mat);
  }
  // Vertical surface in the X-Y plane from an outline [[x, y], ...].
  function fin(outline, thick, mat) {
    const s = new T.Shape();
    outline.forEach(([x, y], i) => (i ? s.lineTo(x, y) : s.moveTo(x, y)));
    const g = new T.ExtrudeGeometry(s, { depth: thick, bevelEnabled: true, bevelThickness: thick * 0.4, bevelSize: thick * 0.4, bevelSegments: 1 });
    g.translate(0, 0, -thick / 2);
    return new T.Mesh(g, mat);
  }
  function cyl(r1, r2, len, mat, axis = "x", seg = 20) {
    const g = new T.CylinderGeometry(r2, r1, len, seg);
    if (axis === "x") g.rotateZ(-Math.PI / 2);
    if (axis === "z") g.rotateX(Math.PI / 2);
    return new T.Mesh(g, mat);
  }
  function box(x, y, z, mat) { return new T.Mesh(new T.BoxGeometry(x, y, z), mat); }
  function at(o, x, y, z) { o.position.set(x, y, z); return o; }
  function mirrorZ(mesh) { const m = mesh.clone(); m.scale.z *= -1; m.position.z *= -1; return m; }
  function navLights(g, x, y, zTip) {
    const red = at(new T.Mesh(new T.SphereGeometry(0.035, 10, 10), M.glow(0xff2a2a)), x, y, -zTip);
    const green = at(new T.Mesh(new T.SphereGeometry(0.035, 10, 10), M.glow(0x2aff7a)), x, y, zTip);
    red.userData.blink = 0; green.userData.blink = 0;
    g.add(red, green);
  }
  function strobe(g, x, y, z) {
    const s = at(new T.Mesh(new T.SphereGeometry(0.03, 8, 8), M.glow(0xffffff)), x, y, z);
    s.userData.strobe = true;
    g.add(s);
  }

  // ------------------------------------------------------------- models
  function passengerJet() {
    const g = new T.Group(), body = M.body(), dark = M.dark();
    g.add(lathe([[0, -2.15], [0.1, -2.05], [0.26, -1.6], [0.34, -1.0], [0.34, 1.15], [0.31, 1.55], [0.22, 1.86], [0.08, 2.06], [0, 2.12]], body));
    const wing = at(plate([[0.55, 0.25], [-0.35, 0.25], [-0.85, 2.15], [-0.55, 2.15]], 0.05, body), 0, -0.14, 0);
    g.add(wing, mirrorZ(wing));
    [0.8, -0.8].forEach((z) => {
      g.add(at(cyl(0.14, 0.12, 0.62, dark), 0.2, -0.34, z));
      g.add(at(cyl(0.1, 0.1, 0.02, M.glow(0x050505)), 0.52, -0.34, z));
      g.add(at(box(0.36, 0.12, 0.04, body), 0.05, -0.22, z));
    });
    const hs = at(plate([[-1.55, 0.12], [-1.95, 0.12], [-2.15, 0.85], [-1.98, 0.85]], 0.035, body), 0, 0.05, 0);
    g.add(hs, mirrorZ(hs));
    g.add(fin([[-1.45, 0.26], [-2.0, 0.26], [-2.2, 1.12], [-1.98, 1.12]], 0.04, body));
    g.add(at(box(0.18, 0.06, 0.5, M.glass()), 1.78, 0.14, 0));
    navLights(g, -0.7, -0.14, 2.17);
    strobe(g, -2.18, 1.14, 0);
    return { group: g };
  }

  function fighterJet() {
    const g = new T.Group(), body = M.body(), dark = M.dark();
    g.add(lathe([[0.17, -1.85], [0.2, -1.6], [0.24, -0.6], [0.24, 0.4], [0.19, 0.95], [0.11, 1.45], [0.03, 1.88], [0, 1.95]], body));
    const wing = plate([[0.55, 0.18], [-1.45, 0.18], [-1.45, 1.38], [-1.15, 1.38]], 0.035, body);
    g.add(wing, mirrorZ(wing));
    const hs = plate([[-1.35, 0.2], [-1.85, 0.2], [-1.95, 0.75], [-1.75, 0.75]], 0.03, body);
    g.add(hs, mirrorZ(hs));
    [0.26, -0.26].forEach((z) => {
      const f = at(fin([[-1.0, 0.12], [-1.6, 0.12], [-1.78, 0.78], [-1.55, 0.78]], 0.03, body), 0, 0, z);
      f.rotation.x = z > 0 ? -0.32 : 0.32;
      g.add(f);
      g.add(at(box(0.9, 0.16, 0.14, dark), 0.15, -0.1, z * 0.95));
    });
    const canopy = at(new T.Mesh(new T.SphereGeometry(0.5, 24, 16, 0, Math.PI * 2, 0, Math.PI / 2), M.glass()), 0.95, 0.12, 0);
    canopy.scale.set(1.05, 0.38, 0.32);
    g.add(canopy);
    g.add(at(cyl(0.2, 0.17, 0.22, dark), -1.92, 0, 0));
    const burner = at(new T.Mesh(new T.CircleGeometry(0.15, 24), M.glow(ACCENT)), -2.035, 0, 0);
    burner.rotation.y = -Math.PI / 2;
    burner.userData.pulse = true;
    g.add(burner);
    navLights(g, -1.3, 0, 1.4);
    return { group: g };
  }

  function helicopter() {
    const g = new T.Group(), body = M.body(), dark = M.dark();
    const cabin = new T.Mesh(new T.SphereGeometry(0.62, 28, 20), body);
    cabin.scale.set(1.45, 0.92, 0.82);
    g.add(cabin);
    const canopy = at(new T.Mesh(new T.SphereGeometry(0.5, 28, 20, -Math.PI / 2, Math.PI, 0, Math.PI * 0.62), M.glass()), 0.42, 0.06, 0);
    canopy.scale.set(1.0, 0.95, 0.92);
    g.add(canopy);
    g.add(at(cyl(0.2, 0.06, 2.3, body), -1.65, 0.12, 0));
    g.add(fin([[-2.55, 0.1], [-2.85, 0.1], [-2.95, 0.75], [-2.78, 0.75]], 0.04, body));
    const ths = plate([[-2.45, 0.02], [-2.65, 0.02], [-2.68, 0.38], [-2.52, 0.38]], 0.025, body);
    g.add(ths, mirrorZ(ths));
    g.add(at(cyl(0.05, 0.05, 0.32, dark, "y"), 0, 0.66, 0));
    g.add(at(cyl(0.22, 0.16, 0.16, dark, "y"), 0, 0.58, 0));
    const rotor = at(new T.Group(), 0, 0.84, 0);
    rotor.add(cyl(0.09, 0.09, 0.08, dark, "y"));
    for (let i = 0; i < 4; i++) {
      const b = at(box(1.75, 0.018, 0.12, M.blade()), 0.92, 0, 0);
      const arm = new T.Group(); arm.rotation.y = (i * Math.PI) / 2; arm.add(b); rotor.add(arm);
    }
    rotor.userData.spin = { axis: "y", speed: 14 };
    g.add(rotor);
    const tail = at(new T.Group(), -2.82, 0.48, 0.09);
    for (let i = 0; i < 2; i++) {
      const b = box(0.04, 0.5, 0.012, M.blade()); b.rotation.z = (i * Math.PI) / 2; tail.add(b);
    }
    tail.userData.spin = { axis: "z", speed: 24 };
    g.add(tail);
    [0.42, -0.42].forEach((z) => {
      g.add(at(cyl(0.025, 0.025, 1.7, dark), 0.05, -0.78, z));
      [0.45, -0.35].forEach((x) => g.add(at(cyl(0.02, 0.02, 0.32, dark, "y"), x, -0.63, z * 0.9)));
    });
    navLights(g, -0.1, 0.0, 0.52);
    strobe(g, -2.95, 0.78, 0);
    return { group: g };
  }

  function propeller() {
    const g = new T.Group(), body = M.body(), dark = M.dark();
    g.add(lathe([[0, -1.95], [0.07, -1.85], [0.2, -1.15], [0.29, -0.3], [0.3, 0.75], [0.27, 1.05], [0.24, 1.2]], body));
    g.add(at(cyl(0.25, 0.24, 0.18, dark), 1.28, 0, 0));
    const spinner = at(new T.Mesh(new T.ConeGeometry(0.1, 0.22, 20), dark), 1.47, 0, 0);
    spinner.rotation.z = -Math.PI / 2;
    g.add(spinner);
    const prop = at(new T.Group(), 1.4, 0, 0);
    for (let i = 0; i < 3; i++) {
      const b = at(box(0.025, 0.62, 0.09, M.blade()), 0, 0.33, 0);
      b.rotation.y = 0.35;
      const arm = new T.Group(); arm.rotation.x = (i * Math.PI * 2) / 3; arm.add(b); prop.add(arm);
    }
    prop.userData.spin = { axis: "x", speed: 30 };
    g.add(prop);
    const wing = at(plate([[0.55, 0.0], [0.0, 0.0], [0.05, 2.2], [0.42, 2.2]], 0.05, body), 0, 0.3, 0);
    g.add(wing, mirrorZ(wing));
    [0.95, -0.95].forEach((z) => {
      const s = at(cyl(0.018, 0.018, 0.95, dark, "y"), 0.25, 0.02, z * 0.55);
      s.rotation.x = z > 0 ? -0.95 : 0.95;
      g.add(s);
    });
    const hs = at(plate([[-1.5, 0.05], [-1.88, 0.05], [-1.9, 0.82], [-1.66, 0.82]], 0.03, body), 0, 0.08, 0);
    g.add(hs, mirrorZ(hs));
    g.add(fin([[-1.4, 0.2], [-1.92, 0.2], [-2.02, 0.85], [-1.8, 0.85]], 0.035, body));
    g.add(at(box(0.36, 0.08, 0.52, M.glass()), 0.82, 0.22, 0));
    [0.32, -0.32].forEach((z) => g.add(at(cyl(0.09, 0.09, 0.05, dark, "z"), 0.35, -0.42, z)));
    g.add(at(cyl(0.07, 0.07, 0.04, dark, "z"), 1.0, -0.38, 0));
    navLights(g, 0.25, 0.3, 2.25);
    return { group: g };
  }

  const BUILD = { passenger_jet: passengerJet, fighter_jet: fighterJet, helicopter: helicopter, propeller: propeller };
  const ORDER = ["passenger_jet", "fighter_jet", "helicopter", "propeller"];

  // Turn a solid model into a glowing wireframe "hologram".
  function hologram(model, color, opacity) {
    const mat = new T.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false });
    model.group.traverse((o) => {
      if (o.isMesh) {
        const lines = new T.LineSegments(new T.EdgesGeometry(o.geometry, 8), mat);
        lines.position.copy(o.position); lines.rotation.copy(o.rotation); lines.scale.copy(o.scale);
        lines.userData = o.userData;
        o.parent.add(lines);
        o.visible = false;
      }
    });
    return model;
  }

  // ------------------------------------------------------------- viewer
  class Viewer {
    constructor(host) {
      this.host = host;
      this.renderer = new T.WebGLRenderer({ antialias: true, alpha: true });
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      this.renderer.outputColorSpace = T.SRGBColorSpace;
      this.renderer.toneMapping = T.ACESFilmicToneMapping;
      this.renderer.toneMappingExposure = 1.15;
      host.appendChild(this.renderer.domElement);

      this.scene = new T.Scene();
      this.camera = new T.PerspectiveCamera(32, 1, 0.1, 100);
      this.dist = 8.2;
      this.scene.add(new T.HemisphereLight(0x9aa3b5, 0x1a0e08, 1.0));
      const key = new T.DirectionalLight(0xffffff, 2.8); key.position.set(3, 5, 4); this.scene.add(key);
      const rim = new T.DirectionalLight(ACCENT, 3.2); rim.position.set(-4, -2, -5); this.scene.add(rim);
      const fill = new T.DirectionalLight(0x7d94ff, 0.7); fill.position.set(-5, 1, 3); this.scene.add(fill);
      this.under = new T.PointLight(ACCENT, 6, 6, 2); this.under.position.set(0, -1.4, 0); this.scene.add(this.under);

      // Faint ground disc with a glowing ring, like a landing pad under the model.
      const ring = new T.Mesh(new T.RingGeometry(1.95, 1.975, 96), new T.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.35 }));
      ring.rotation.x = -Math.PI / 2; ring.position.y = -1.35; this.scene.add(ring); this.ring = ring;
      const disc = new T.Mesh(new T.CircleGeometry(1.9, 64), new T.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.025 }));
      disc.rotation.x = -Math.PI / 2; disc.position.y = -1.36; this.scene.add(disc);

      this.pivot = new T.Group(); this.scene.add(this.pivot);
      this.yaw = -0.6; this.pitch = 0.22; this.vYaw = 0; this.vPitch = 0;
      this.lastInteract = 0; this.model = null; this.state = "idle"; this.cycle = 0; this.flyT = 1;
      this.clock = new T.Clock();
      this._controls();
      new ResizeObserver(() => this._resize()).observe(host);
      this._resize();
      this.setState({ state: "idle" });
      this._loop();
    }

    _resize() {
      const w = this.host.clientWidth || 600, h = this.host.clientHeight || 480;
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.dist = w < 520 ? 10 : 7.6;
      this.camera.updateProjectionMatrix();
    }

    _controls() {
      const el = this.renderer.domElement;
      el.style.touchAction = "none";
      let drag = null;
      el.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId); this.host.classList.add("grabbing"); });
      el.addEventListener("pointermove", (e) => {
        if (!drag) return;
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        drag = { x: e.clientX, y: e.clientY };
        this.vYaw = dx * 0.008; this.vPitch = dy * 0.006;
        this.yaw += this.vYaw; this.pitch = Math.max(-0.9, Math.min(1.1, this.pitch + this.vPitch));
        this.lastInteract = performance.now();
      });
      const end = () => { drag = null; this.host.classList.remove("grabbing"); };
      el.addEventListener("pointerup", end); el.addEventListener("pointercancel", end);
      el.addEventListener("wheel", (e) => {
        e.preventDefault();
        this.zoom = Math.max(0.65, Math.min(1.6, (this.zoom || 1) * (1 + Math.sign(e.deltaY) * 0.08)));
        this.lastInteract = performance.now();
      }, { passive: false });
      el.addEventListener("dblclick", () => { this.yaw = -0.6; this.pitch = 0.22; this.zoom = 1; this.vYaw = this.vPitch = 0; });
      el.tabIndex = 0;
      el.setAttribute("aria-label", "3D aircraft model. Use arrow keys to rotate.");
      el.addEventListener("keydown", (e) => {
        const k = { ArrowLeft: [-0.15, 0], ArrowRight: [0.15, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1] }[e.key];
        if (k) { e.preventDefault(); this.yaw += k[0]; this.pitch = Math.max(-0.9, Math.min(1.1, this.pitch + k[1])); this.lastInteract = performance.now(); }
      });
    }

    _setModel(model) {
      while (this.pivot.children.length) this.pivot.remove(this.pivot.children[0]);
      this.model = model;
      if (model) this.pivot.add(model.group);
      this.flyT = 0;
    }

    setState(s) {
      this.state = s.state || "idle";
      this.host.dataset.state = this.state;
      if (this.state === "result") this._setModel(BUILD[s.cls]());
      else if (this.state === "low") this._setModel(hologram(BUILD[s.cls](), AMBER, 0.75));
      else if (this.state === "loading") this._setModel(hologram(BUILD[ORDER[this.cycle % 4]](), 0xffffff, 0.5));
      else if (this.state === "error") this._setModel(null);
      else { this.cycle = 0; this._setModel(hologram(BUILD[ORDER[0]](), 0xffffff, 0.22)); this.cycleAt = performance.now(); }
      this.ring.material.color.setHex(this.state === "low" ? AMBER : ACCENT);
    }

    _loop() {
      requestAnimationFrame(() => this._loop());
      const dt = Math.min(this.clock.getDelta(), 0.05), t = this.clock.elapsedTime, now = performance.now();

      // idle: cycle through the four types every 3.2 s
      if (this.state === "idle" && now - this.cycleAt > 3200) {
        this.cycle++; this.cycleAt = now;
        this._setModel(hologram(BUILD[ORDER[this.cycle % 4]](), 0xffffff, 0.22));
      }
      if (this.state === "loading" && Math.floor(t * 2.5) % 4 !== this.cycle % 4) {
        this.cycle = Math.floor(t * 2.5) % 4;
        this._setModel(hologram(BUILD[ORDER[this.cycle]](), 0xffffff, 0.5));
        this.flyT = 1;
      }

      // inertia + gentle auto-rotate when nobody is touching it
      const idleFor = now - this.lastInteract;
      if (idleFor > 60 && idleFor < 1200) { this.yaw += this.vYaw * 0.9; this.vYaw *= 0.92; }
      if (!reduceMotion && idleFor > 2500) this.yaw += dt * (this.state === "loading" ? 1.6 : 0.28);

      // fly-in when a new model appears
      this.flyT = Math.min(1, this.flyT + dt / 0.9);
      const e = 1 - Math.pow(1 - this.flyT, 3);
      const s = 0.55 + 0.45 * e;
      this.pivot.scale.setScalar(s);
      this.pivot.rotation.set(this.pitch * 0.35, this.yaw + (1 - e) * 2.4, 0);
      this.pivot.position.y = (1 - e) * 0.8 + Math.sin(t * 1.2) * 0.04;

      const z = this.dist * (this.zoom || 1);
      this.camera.position.set(0, Math.sin(this.pitch) * z * 0.6, Math.cos(this.pitch) * z);
      this.camera.lookAt(0, -0.1, 0);

      if (this.model) this.model.group.traverse((o) => {
        const sp = o.userData.spin;
        if (sp) o.rotation[sp.axis] += dt * sp.speed * (reduceMotion ? 0.15 : 1);
        if (o.userData.strobe) o.visible = (t % 1.2) < 0.08;
        if (o.userData.pulse && o.material) o.material.color.setHSL(0.04, 1, 0.45 + 0.12 * Math.sin(t * 18));
      });
      this.under.intensity = this.state === "result" ? 6 + Math.sin(t * 2) * 1.2 : 2.5;
      this.ring.rotation.z += dt * 0.15;
      this.renderer.render(this.scene, this.camera);
    }
  }

  // ------------------------------------------------------------- wiring
  // Gradio re-renders HTML blocks, so we watch the page for the stage and the
  // hidden state element and react when they appear or change.
  let viewer = null, lastKey = null;
  function sync() {
    const host = document.getElementById("stage-canvas");
    if (host && !host.dataset.mounted) {
      host.dataset.mounted = "1";
      try { viewer = new Viewer(host); } catch (err) { host.classList.add("no-webgl"); console.error(err); }
    }
    const st = document.querySelector("#viewer-state .vstate");
    if (st && viewer && st.dataset.key !== lastKey) {
      lastKey = st.dataset.key;
      viewer.setState({ state: st.dataset.state, cls: st.dataset.cls });
      document.querySelectorAll("[data-mod-item]").forEach((n) =>
        n.classList.toggle("on", n.dataset.modItem === st.dataset.mod));
    }
  }
  new MutationObserver(sync).observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", sync);
  sync();

  // Drag-to-compare slider in the evidence section.
  document.addEventListener("input", (e) => {
    if (e.target.classList && e.target.classList.contains("cmp-range"))
      e.target.closest(".cmp").style.setProperty("--x", e.target.value + "%");
  });
  window.AeroViewer = { get: () => viewer };
})();
