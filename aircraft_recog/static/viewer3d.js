/* 3D aircraft viewer for the demo.
 *
 * Four original aircraft built in code (no downloaded model files):
 * a twin-engine airliner, a twin-tail fighter, a utility helicopter and a
 * high-wing propeller plane. Fuselages are lofted from smooth cross-sections,
 * wings use NACA airfoil profiles, and the paint (windows, stripes, panel
 * lines) is drawn onto canvas textures. A studio environment map gives the
 * paint and glass realistic reflections.
 *
 * Real model files: put a .glb file in static/models/<class>.glb
 * (passenger_jet, fighter_jet, helicopter, propeller) and the viewer uses it
 * instead. Optional static/models/<class>.json: {"rotateX":0,"rotateY":90,"rotateZ":0}
 * (degrees) to turn the model so its nose points right.
 *
 * Drag to rotate 360 degrees, scroll or pinch to zoom, double-click to reset.
 */
(function () {
  "use strict";
  const T = window.THREE;
  if (!T) { console.error("three.js not loaded"); return; }

  const ACCENT = 0xff4d1f;
  const AMBER = 0xffb020;
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const LEN = 4.4;   // every model is scaled to about this length

  // =============================================================== textures
  function canvasTex(w, h, draw) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    draw(c.getContext("2d"), w, h);
    const t = new T.CanvasTexture(c);
    t.encoding = T.sRGBEncoding;
    t.anisotropy = 8;
    return t;
  }
  // Fuselage textures: x = along the body (tail -> nose), y = around it
  // (0 = belly, 0.25 = right side, 0.5 = top, 0.75 = left side).
  function panelLines(g, w, h, alpha, step) {
    g.strokeStyle = `rgba(0,0,0,${alpha})`;
    g.lineWidth = 1;
    for (let x = step; x < w; x += step) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); }
    for (let y = h / 8; y < h; y += h / 8) { g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); }
  }
  function airlinerTex() {
    return canvasTex(2048, 512, (g, w, h) => {
      g.fillStyle = "#f4f5f7"; g.fillRect(0, 0, w, h);
      g.fillStyle = "#c9ccd2"; g.fillRect(0, 0, w, h * 0.17); g.fillRect(0, h * 0.83, w, h * 0.17);   // grey belly
      panelLines(g, w, h, 0.05, 64);
      [0.262, 0.738].forEach((u) => {                       // cheatline stripes
        g.fillStyle = "#1f4e8c"; g.fillRect(w * 0.1, h * u - 6, w * 0.82, 12);
        g.fillStyle = "#3d8fd6"; g.fillRect(w * 0.1, h * u + (u < 0.5 ? -14 : 8), w * 0.82, 4);
      });
      g.fillStyle = "#1a1f27";                               // cabin windows
      [0.305, 0.695].forEach((u) => {
        for (let t = 0.3; t < 0.875; t += 0.0092) {
          if (Math.abs(t - 0.555) < 0.012) continue;        // over-wing exit gap
          g.beginPath(); g.ellipse(w * t, h * u, 6, 9, 0, 0, Math.PI * 2); g.fill();
        }
        [0.885, 0.325, 0.555].forEach((t) => {               // doors
          g.strokeStyle = "rgba(0,0,0,.28)"; g.lineWidth = 2;
          g.strokeRect(w * t - 14, h * u - 34, 28, 64);
        });
      });
      g.fillStyle = "#0f141b";                               // cockpit windscreen
      [0.33, 0.67].forEach((u) => {
        g.beginPath();
        g.moveTo(w * 0.948, h * u - 22 * Math.sign(u - 0.5));
        g.lineTo(w * 0.968, h * u - 30 * Math.sign(u - 0.5));
        g.lineTo(w * 0.972, h * (u + (0.5 - u) * 0.9));
        g.lineTo(w * 0.952, h * (u + (0.5 - u) * 0.9));
        g.closePath(); g.fill();
      });
      g.fillStyle = "#2c3036"; g.fillRect(w * 0.992, 0, w * 0.008, h);   // radome tip
    });
  }
  function fighterTex() {
    return canvasTex(1024, 512, (g, w, h) => {
      g.fillStyle = "#858d97"; g.fillRect(0, 0, w, h);
      g.fillStyle = "#737b85"; g.fillRect(0, 0, w, h * 0.2); g.fillRect(0, h * 0.8, w, h * 0.2);
      panelLines(g, w, h, 0.14, 38);
      g.strokeStyle = "rgba(0,0,0,.12)";
      let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
      for (let i = 0; i < 40; i++) g.strokeRect(rnd() * w, rnd() * h, 20 + rnd() * 40, 12 + rnd() * 20);
      g.fillStyle = "#4a5058"; g.fillRect(w * 0.935, 0, w * 0.065, h);       // dark radome
      g.fillStyle = "#3a3f46"; g.fillRect(0, 0, w * 0.06, h);                // engine section
    });
  }
  function heliTex() {
    return canvasTex(1024, 512, (g, w, h) => {
      g.fillStyle = "#3a403b"; g.fillRect(0, 0, w, h);
      panelLines(g, w, h, 0.18, 46);
      g.fillStyle = "#11161c";                                // canopy glass + cabin windows
      g.beginPath(); g.moveTo(w * 0.86, h * 0.3); g.lineTo(w * 0.995, h * 0.24); g.lineTo(w * 0.995, h * 0.76); g.lineTo(w * 0.86, h * 0.7); g.closePath(); g.fill();
      [0.3, 0.7].forEach((u) => { g.fillRect(w * 0.72, h * u - 30, w * 0.08, 60); g.fillRect(w * 0.62, h * u - 26, w * 0.07, 48); });
      g.strokeStyle = "rgba(0,0,0,.4)"; g.lineWidth = 3;
      [0.3, 0.7].forEach((u) => g.strokeRect(w * 0.6, h * u - 60, w * 0.21, 110));   // sliding door
      g.fillStyle = "#d9d9d9"; g.fillRect(w * 0.3, h * 0.29, w * 0.1, 8); g.fillRect(w * 0.3, h * 0.71 - 8, w * 0.1, 8);
    });
  }
  function propTex() {
    return canvasTex(1024, 512, (g, w, h) => {
      g.fillStyle = "#f2f2f0"; g.fillRect(0, 0, w, h);
      panelLines(g, w, h, 0.05, 52);
      [0.36, 0.64].forEach((u) => {
        const s = u < 0.5 ? 1 : -1;
        g.fillStyle = "#1c3f73"; g.fillRect(0, h * u - s * 18, w, 14 * s);
        g.fillStyle = "#c8202f"; g.fillRect(0, h * u - s * 30, w, 6 * s);
      });
      g.fillStyle = "#121820";
      [0.3, 0.7].forEach((u) => {
        g.fillRect(w * 0.64, h * u - 24, w * 0.07, 34);
        g.fillRect(w * 0.725, h * u - 26, w * 0.075, 38);
      });
      g.beginPath(); g.moveTo(w * 0.8, h * 0.36); g.lineTo(w * 0.87, h * 0.4); g.lineTo(w * 0.87, h * 0.6); g.lineTo(w * 0.8, h * 0.64); g.closePath(); g.fill();
    });
  }
  function fanTex() {
    return canvasTex(256, 256, (g, w, h) => {
      g.fillStyle = "#15171a"; g.fillRect(0, 0, w, h);
      g.translate(w / 2, h / 2);
      for (let i = 0; i < 24; i++) {
        g.rotate((Math.PI * 2) / 24);
        const gr = g.createLinearGradient(0, 0, 0, w / 2);
        gr.addColorStop(0, "#666b73"); gr.addColorStop(1, "#2b2e33");
        g.fillStyle = gr; g.beginPath(); g.moveTo(0, 18); g.quadraticCurveTo(14, 60, 10, w / 2); g.lineTo(-2, w / 2); g.quadraticCurveTo(4, 60, -6, 18); g.fill();
      }
    });
  }
  function shadowTex() {
    return canvasTex(256, 256, (g, w, h) => {
      const gr = g.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
      gr.addColorStop(0, "rgba(0,0,0,.75)"); gr.addColorStop(0.5, "rgba(0,0,0,.35)"); gr.addColorStop(1, "rgba(0,0,0,0)");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
    });
  }

  // =============================================================== materials
  const paint = (map, color = 0xffffff, rough = 0.32) =>
    new T.MeshPhysicalMaterial({ map, color, metalness: 0.12, roughness: rough, clearcoat: 0.9, clearcoatRoughness: 0.12, side: T.DoubleSide });
  const matte = (color, rough = 0.6, metal = 0.3) => new T.MeshStandardMaterial({ color, metalness: metal, roughness: rough, side: T.DoubleSide });
  const metal = (color = 0xb8bcc2, rough = 0.28) => new T.MeshStandardMaterial({ color, metalness: 1.0, roughness: rough, side: T.DoubleSide });
  const glass = (tint = 0x0b1220) => new T.MeshPhysicalMaterial({ color: tint, metalness: 0.2, roughness: 0.03, clearcoat: 1, clearcoatRoughness: 0.02, reflectivity: 1, side: T.DoubleSide });
  const glow = (c) => new T.MeshBasicMaterial({ color: c });

  // =============================================================== geometry
  // Fuselage lofted from cross-sections. stations: [t, halfWidth, halfHeight, centreY]
  // with t from 0 (tail) to 1 (nose); n > 2 makes the section squarer (superellipse).
  function fuselage(stations, length, mat, n = 2.0, radial = 48, samples = 120) {
    const A = new T.CatmullRomCurve3(stations.map((s) => new T.Vector3(s[0], s[1], s[2])), false, "centripetal");
    const B = new T.CatmullRomCurve3(stations.map((s) => new T.Vector3(s[0], s[3], 0)), false, "centripetal");
    const pos = [], uv = [], idx = [];
    const f = (v) => Math.sign(v) * Math.pow(Math.abs(v), 2 / n);
    for (let i = 0; i <= samples; i++) {
      const s = i / samples, a = A.getPoint(s), b = B.getPoint(s);
      const t = a.x, rz = Math.max(a.y, 0.0005), ry = Math.max(a.z, 0.0005), yc = b.y;
      for (let j = 0; j <= radial; j++) {
        const th = (j / radial) * Math.PI * 2;
        pos.push((t - 0.5) * length, yc - ry * f(Math.cos(th)), rz * f(Math.sin(th)));
        uv.push(t, j / radial);
      }
    }
    for (let i = 0; i < samples; i++)
      for (let j = 0; j < radial; j++) {
        const a = i * (radial + 1) + j, b = a + radial + 1;
        idx.push(a, b, a + 1, b, b + 1, a + 1);
      }
    const g = new T.BufferGeometry();
    g.setAttribute("position", new T.Float32BufferAttribute(pos, 3));
    g.setAttribute("uv", new T.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    return new T.Mesh(g, mat);
  }

  // Wing / tail surface with a NACA-style airfoil section.
  // stations: [{x: leading-edge x, s: span position, d: offset (dihedral), c: chord, t: thickness ratio}]
  // vertical=false: span along +z, thickness along y.  vertical=true: span along +y, thickness along z.
  function surface(stations, mat, vertical = false, camber = 0.02, n = 22) {
    const pos = [], idx = [];
    const xs = []; for (let i = 0; i <= n; i++) xs.push(0.5 * (1 - Math.cos((Math.PI * i) / n)));
    const ring = [];
    stations.forEach((st) => {
      const pts = [];
      const half = (x) => 5 * st.t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
      const cam = (x) => (x < 0.4 ? (camber / 0.16) * (0.8 * x - x * x) : (camber / 0.36) * (0.2 + 0.8 * x - x * x));
      for (let i = 0; i <= n; i++) pts.push([xs[i], cam(xs[i]) + half(xs[i])]);
      for (let i = n - 1; i >= 1; i--) pts.push([xs[i], cam(xs[i]) - half(xs[i])]);
      ring.push(pts.map(([x, h]) => {
        const X = st.x - x * st.c, H = h * st.c;
        return vertical ? [X, st.s, st.d + H] : [X, st.d + H, st.s];
      }));
    });
    const m = ring[0].length;
    ring.forEach((r) => r.forEach((p) => pos.push(...p)));
    for (let k = 0; k < ring.length - 1; k++)
      for (let i = 0; i < m; i++) {
        const a = k * m + i, b = k * m + ((i + 1) % m), c = a + m, d = b + m;
        idx.push(a, c, b, b, c, d);
      }
    [0, ring.length - 1].forEach((k) => {                  // end caps
      const base = pos.length / 3;
      const cx = ring[k].reduce((s, p) => s + p[0], 0) / m, cy = ring[k].reduce((s, p) => s + p[1], 0) / m, cz = ring[k].reduce((s, p) => s + p[2], 0) / m;
      pos.push(cx, cy, cz);
      for (let i = 0; i < m; i++) idx.push(base, k * m + i, k * m + ((i + 1) % m));
    });
    const g = new T.BufferGeometry();
    g.setAttribute("position", new T.Float32BufferAttribute(pos, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    return new T.Mesh(g, mat);
  }

  function lathe(profile, mat, seg = 40) {   // body of revolution along +x; profile [[r, x]]
    const g = new T.LatheGeometry(profile.map(([r, x]) => new T.Vector2(Math.max(r, 0.0005), x)), seg);
    g.rotateZ(-Math.PI / 2);
    return new T.Mesh(g, mat);
  }
  function tube(points, r, mat) {
    return new T.Mesh(new T.TubeGeometry(new T.CatmullRomCurve3(points.map((p) => new T.Vector3(...p))), 40, r, 10), mat);
  }
  const at = (o, x, y, z) => { o.position.set(x, y, z); return o; };
  const mirrorZ = (o) => { const m = o.clone(); m.scale.z *= -1; m.position.z *= -1; m.rotation.x *= -1; return m; };
  const both = (g, o) => { g.add(o, mirrorZ(o)); };
  function lights(g, x, y, z, strobeAt) {
    const s = (c) => new T.Mesh(new T.SphereGeometry(0.025, 10, 10), glow(c));
    g.add(at(s(0xff2a2a), x, y, -z), at(s(0x2aff7a), x, y, z));
    if (strobeAt) { const w = at(s(0xffffff), ...strobeAt); w.userData.strobe = true; g.add(w); }
  }

  // =============================================================== aircraft
  function airliner() {
    const g = new T.Group();
    g.add(fuselage([[0, .008, .015, .17], [.05, .06, .075, .15], [.14, .15, .17, .085], [.25, .228, .245, .015], [.33, .24, .255, 0],
      [.86, .24, .255, 0], [.93, .215, .232, -.012], [.968, .15, .165, -.035], [.99, .06, .065, -.052], [1, .004, .004, -.056]], LEN, paint(airlinerTex())));
    const white = paint(null, 0xeef0f3), alu = metal(0xc4c8ce, 0.22);
    both(g, surface([{ x: .5, s: .18, d: -.13, c: 1.12, t: .13 }, { x: .02, s: .8, d: -.08, c: .62, t: .11 }, { x: -.78, s: 2.06, d: .07, c: .26, t: .09 }], white));
    both(g, surface([{ x: -.76, s: .065, d: 2.06, c: .27, t: .08 }, { x: -.98, s: .38, d: 2.13, c: .12, t: .08 }], white, true));
    const z = 0.74, eng = new T.Group();
    eng.add(lathe([[.1, -.48], [.135, -.36], [.18, -.12], [.19, .12], [.185, .28], [.168, .36], [.158, .37], [.15, .33]], white));
    eng.add(lathe([[.0, -.66], [.05, -.6], [.09, -.48]], alu));
    const fan = new T.Mesh(new T.CircleGeometry(.15, 40), new T.MeshStandardMaterial({ map: fanTex(), metalness: .7, roughness: .4 }));
    fan.rotation.y = Math.PI / 2;
    const fanG = at(new T.Group(), .3, 0, 0); fanG.add(fan); fanG.userData.spin = { axis: "x", speed: 6 }; eng.add(fanG);
    eng.add(lathe([[.0, .3], [.05, .27], [.06, .23]], alu));
    const lip = at(new T.Mesh(new T.TorusGeometry(.163, .012, 10, 40), alu), .365, 0, 0); lip.rotation.y = Math.PI / 2; eng.add(lip);
    at(eng, .32, -.36, z);
    g.add(eng, mirrorZ(eng));
    both(g, surface([{ x: .45, s: -.27, d: z, c: .7, t: .16 }, { x: .3, s: -.12, d: z, c: .75, t: .16 }], white, true));
    both(g, surface([{ x: -1.62, s: .1, d: .12, c: .55, t: .1 }, { x: -2.05, s: .9, d: .2, c: .2, t: .09 }], white));
    g.add(surface([{ x: -1.38, s: .18, d: 0, c: .82, t: .11 }, { x: -2.06, s: 1.12, d: 0, c: .3, t: .1 }], paint(null, 0x1f4e8c), true, 0));
    lights(g, -.8, .07, 2.12, [-2.18, 1.14, 0]);
    return { group: g };
  }

  function fighter() {
    const g = new T.Group();
    const body = paint(fighterTex(), 0xffffff, 0.55); body.clearcoat = 0.2;
    g.add(fuselage([[0, .13, .13, 0], [.06, .2, .155, .0], [.25, .29, .17, .0], [.5, .27, .17, .015], [.68, .19, .17, .03],
      [.8, .12, .13, .01], [.92, .07, .075, -.005], [.985, .02, .02, -.015], [1, .003, .003, -.016]], 4.1, body, 2.7));
    const skin = paint(null, 0x858d97, 0.55); skin.clearcoat = 0.2;
    both(g, surface([{ x: 1.05, s: .14, d: 0, c: 2.15, t: .035 }, { x: .25, s: .42, d: 0, c: 1.38, t: .04 }, { x: -.76, s: 1.22, d: -.03, c: .36, t: .04 }], skin, false, 0.005));
    both(g, surface([{ x: -1.2, s: .18, d: -.01, c: .66, t: .04 }, { x: -1.62, s: .82, d: -.03, c: .26, t: .04 }], skin, false, 0));
    const tail = surface([{ x: -.82, s: .12, d: .2, c: .82, t: .05 }, { x: -1.42, s: .82, d: .25, c: .3, t: .05 }], skin, true, 0);
    tail.rotation.x = -.26;
    both(g, tail);
    both(g, at(fuselage([[0, .1, .11, 0], [.5, .12, .12, 0], [1, .11, .1, 0]], .95, skin, 4), .2, -.08, .26));
    const hole = at(new T.Mesh(new T.CircleGeometry(.09, 24), glow(0x050505)), .68, -.08, .26); hole.rotation.y = Math.PI / 2; both(g, hole);
    const canopy = at(new T.Mesh(new T.SphereGeometry(.5, 40, 20, 0, Math.PI * 2, 0, Math.PI / 2), glass(0x2a2210)), .78, .12, 0);
    canopy.scale.set(1.1, .3, .26);
    g.add(canopy);
    g.add(at(lathe([[.105, -.18], [.12, 0], [.13, .1]], metal(0x5a5e66, .5)), -2.12, 0, 0));
    const burner = at(new T.Mesh(new T.CircleGeometry(.1, 32), glow(ACCENT)), -2.3, 0, 0);
    burner.rotation.y = -Math.PI / 2; burner.userData.pulse = true; g.add(burner);
    lights(g, -.78, -.02, 1.23);
    return { group: g };
  }

  function helicopter() {
    const g = new T.Group();
    const body = paint(heliTex(), 0xffffff, 0.5); body.clearcoat = 0.3;
    g.add(fuselage([[0, .03, .05, .36], [.3, .055, .075, .29], [.5, .1, .13, .2], [.58, .3, .32, .07], [.7, .4, .42, .01],
      [.84, .39, .4, -.02], [.94, .3, .29, -.08], [.99, .12, .12, -.13], [1, .01, .01, -.14]], 4.7, body, 2.3));
    const dark = matte(0x22262a, .5, .5), skin = paint(null, 0x3a403b, 0.5);
    g.add(at(fuselage([[0, .05, .05, 0], [.3, .2, .14, 0], [.8, .19, .14, 0], [1, .08, .07, -.02]], 1.2, skin, 2.6), .05, .44, 0));
    g.add(at(new T.Mesh(new T.CylinderGeometry(.05, .06, .22, 16), dark), .15, .66, 0));
    const rotor = at(new T.Group(), .15, .78, 0);
    rotor.add(new T.Mesh(new T.CylinderGeometry(.11, .11, .06, 20), dark));
    for (let i = 0; i < 4; i++) {
      const blade = surface([{ x: .07, s: .1, d: 0, c: .14, t: .1 }, { x: .07, s: 2.25, d: -.04, c: .12, t: .1 }], matte(0x1b1d20, .45, .3), false, 0.01);
      blade.rotation.y = Math.PI / 2;
      const arm = new T.Group(); arm.rotation.y = (i * Math.PI) / 2; arm.add(blade); rotor.add(arm);
    }
    rotor.userData.spin = { axis: "y", speed: 13 };
    g.add(rotor);
    g.add(surface([{ x: -1.95, s: .32, d: 0, c: .42, t: .12 }, { x: -2.3, s: 1.0, d: 0, c: .22, t: .1 }], skin, true, 0));
    both(g, surface([{ x: -1.75, s: .04, d: .27, c: .26, t: .1 }, { x: -1.82, s: .42, d: .27, c: .18, t: .1 }], skin, false, 0));
    const tr = at(new T.Group(), -2.2, .72, .09);
    for (let i = 0; i < 2; i++) { const b = new T.Mesh(new T.BoxGeometry(.05, .62, .014), matte(0x1b1d20)); b.rotation.z = (i * Math.PI) / 2; tr.add(b); }
    tr.userData.spin = { axis: "z", speed: 26 };
    g.add(tr);
    const z = 0.44;
    both(g, tube([[-.95, -.72, z], [.55, -.72, z], [.78, -.64, z], [.86, -.52, z]], .025, dark));
    [.45, -.45].forEach((x) => both(g, tube([[x, -.72, z], [x, -.6, z * .95], [x, -.38, z * .55]], .02, dark)));
    lights(g, -.15, .02, .45, [-2.32, 1.02, 0]);
    return { group: g };
  }

  function propeller() {
    const g = new T.Group();
    g.add(fuselage([[0, .02, .03, .2], [.18, .08, .1, .15], [.42, .19, .24, .06], [.6, .25, .32, .03], [.8, .25, .29, .01],
      [.9, .22, .22, -.03], [.97, .19, .17, -.04], [1, .17, .15, -.045]], 3.7, paint(propTex()), 2.4));
    const white = paint(null, 0xf2f2f0), dark = matte(0x202328, .45, .5), alu = metal(0xb5b9bf, .3);
    both(g, surface([{ x: .62, s: .05, d: .34, c: .66, t: .14 }, { x: .6, s: 1.4, d: .38, c: .62, t: .13 }, { x: .5, s: 2.35, d: .43, c: .42, t: .12 }], white));
    both(g, tube([[.15, -.18, .22], [.3, .33, 1.25]], .016, alu));
    both(g, surface([{ x: -1.25, s: .06, d: .2, c: .5, t: .1 }, { x: -1.32, s: .82, d: .2, c: .32, t: .1 }], white, false, 0));
    g.add(surface([{ x: -1.05, s: .2, d: 0, c: .72, t: .1 }, { x: -1.5, s: .82, d: 0, c: .34, t: .1 }], paint(null, 0x1c3f73), true, 0));
    g.add(at(lathe([[.13, 0], [.12, .08], [.08, .17], [.0, .24]], alu), 1.85, -.045, 0));
    const prop = at(new T.Group(), 1.9, -.045, 0);
    for (let i = 0; i < 2; i++) {
      const blade = surface([{ x: .06, s: .08, d: 0, c: .12, t: .12 }, { x: .05, s: .62, d: 0, c: .07, t: .1 }], dark, true, 0.03);
      blade.rotation.y = Math.PI / 2 - 0.35;
      const arm = new T.Group(); arm.rotation.x = i * Math.PI; arm.add(blade); prop.add(arm);
    }
    prop.userData.spin = { axis: "x", speed: 32 };
    g.add(prop);
    [[.25, .34], [1.35, 0]].forEach(([x, z]) => {
      const wheel = at(new T.Mesh(new T.TorusGeometry(.07, .035, 12, 24), dark), x, -.56, z);
      const fair = at(new T.Mesh(new T.SphereGeometry(.12, 20, 14), white), x + .02, -.52, z);
      fair.scale.set(1.5, .7, .55);
      const strut = tube([[x, -.5, z], [x - .02, -.28, z * .6]], .018, alu);
      [wheel, fair, strut].forEach((o) => (z ? both(g, o) : g.add(o)));
    });
    lights(g, .48, .43, 2.36);
    return { group: g };
  }

  const BUILD = { passenger_jet: airliner, fighter_jet: fighter, helicopter: helicopter, propeller: propeller };
  const ORDER = ["passenger_jet", "fighter_jet", "helicopter", "propeller"];

  // Glowing wire-mesh "hologram" version of any model.
  function hologram(group, color, opacity) {
    const mat = new T.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false });
    const meshes = [];
    group.traverse((o) => { if (o.isMesh) meshes.push(o); });
    meshes.forEach((o) => {
      const lines = new T.LineSegments(new T.EdgesGeometry(o.geometry, 12), mat);
      lines.position.copy(o.position); lines.rotation.copy(o.rotation); lines.scale.copy(o.scale);
      lines.userData = o.userData;
      o.parent.add(lines);
      o.visible = false;
    });
    return group;
  }

  // ------------------------------------------------------------ real model files
  const glbCache = {};
  function loadGLB(cls) {
    if (glbCache[cls]) return glbCache[cls];
    const base = window.__AERO_BASE;
    glbCache[cls] = new Promise((resolve) => {
      const listed = window.__AERO_GLB || [];
      if (!base || !T.GLTFLoader || listed.indexOf(cls) < 0) return resolve(null);
      const url = `${base}/models/${cls}.glb`;
      fetch(url, { method: "HEAD" }).then((r) => {
        if (!r.ok) return resolve(null);
        fetch(`${base}/models/${cls}.json`).then((j) => (j.ok ? j.json() : {})).catch(() => ({})).then((cfg) => {
          new T.GLTFLoader().load(url, (gltf) => resolve(normalise(gltf.scene, cfg || {})), undefined, () => resolve(null));
        });
      }).catch(() => resolve(null));
    });
    return glbCache[cls];
  }
  function normalise(obj, cfg) {
    const d = Math.PI / 180;
    obj.rotation.set((cfg.rotateX || 0) * d, (cfg.rotateY || 0) * d, (cfg.rotateZ || 0) * d);
    const wrap = new T.Group(); wrap.add(obj);
    const box = new T.Box3().setFromObject(wrap), size = box.getSize(new T.Vector3()), c = box.getCenter(new T.Vector3());
    obj.position.sub(c);
    wrap.scale.setScalar(LEN / Math.max(size.x, size.y, size.z));
    const outer = new T.Group(); outer.add(wrap);
    obj.traverse((o) => { if (o.isMesh && o.material) o.material.side = T.DoubleSide; });
    return outer;
  }

  // ------------------------------------------------------------ studio lighting
  function studio(renderer) {
    const s = new T.Scene();
    const sky = new T.SphereGeometry(40, 32, 16), col = [], p = sky.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const y = p.getY(i) / 40, v = y > 0 ? 0.08 + 0.5 * y : 0.08 * (1 + y);
      col.push(v, v, v * 1.04);
    }
    sky.setAttribute("color", new T.Float32BufferAttribute(col, 3));
    s.add(new T.Mesh(sky, new T.MeshBasicMaterial({ vertexColors: true, side: T.BackSide })));
    const box = (w, h, pos, c) => { const m = new T.Mesh(new T.PlaneGeometry(w, h), new T.MeshBasicMaterial({ color: c, side: T.DoubleSide })); m.position.set(...pos); m.lookAt(0, 0, 0); s.add(m); };
    box(30, 6, [0, 24, 8], new T.Color(5, 5, 5));          // overhead strip light
    box(10, 18, [-26, 6, 14], new T.Color(2.2, 2.2, 2.4));  // left softbox
    box(10, 18, [26, 4, 12], new T.Color(1.4, 1.4, 1.5));   // right softbox
    box(30, 5, [0, -6, -26], new T.Color(3, .9, .35));      // warm orange back light
    const pm = new T.PMREMGenerator(renderer);
    const tex = pm.fromScene(s, 0.03).texture;
    pm.dispose();
    return tex;
  }

  // ============================================================== viewer
  class Viewer {
    constructor(host) {
      this.host = host;
      const r = this.renderer = new T.WebGLRenderer({ antialias: true, alpha: true });
      r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      r.outputEncoding = T.sRGBEncoding;
      r.physicallyCorrectLights = true;
      r.toneMapping = T.ACESFilmicToneMapping;
      r.toneMappingExposure = 1.05;
      host.appendChild(r.domElement);

      this.scene = new T.Scene();
      this.scene.environment = studio(r);
      this.camera = new T.PerspectiveCamera(30, 1, 0.1, 100);
      const key = new T.DirectionalLight(0xffffff, 2.2); key.position.set(3, 6, 5); this.scene.add(key);
      const rim = new T.DirectionalLight(ACCENT, 2.0); rim.position.set(-4, -1, -6); this.scene.add(rim);
      this.under = new T.PointLight(ACCENT, 8, 5, 2); this.under.position.set(0, -1.15, 0); this.scene.add(this.under);

      const shadow = new T.Mesh(new T.PlaneGeometry(5.5, 3.2), new T.MeshBasicMaterial({ map: shadowTex(), transparent: true, depthWrite: false }));
      shadow.rotation.x = -Math.PI / 2; shadow.position.y = -1.0; this.scene.add(shadow); this.shadow = shadow;
      const ring = new T.Mesh(new T.RingGeometry(2.0, 2.022, 128), new T.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: .35 }));
      ring.rotation.x = -Math.PI / 2; ring.position.y = -1.01; this.scene.add(ring); this.ring = ring;

      this.pivot = new T.Group(); this.scene.add(this.pivot);
      this.yaw = -0.65; this.pitch = 0.22; this.vYaw = 0; this.zoom = 1;
      this.lastInteract = 0; this.model = null; this.state = "idle"; this.cycle = 0; this.flyT = 1; this.token = 0;
      this.anchor = new T.Vector3(); this.anchorTarget = new T.Vector3(); this.bank = 0; this.bankTarget = 0; this.ar = false;
      this.ray = new T.Raycaster(); this.cls = "";
      this.clock = new T.Clock();
      this._controls();
      this._ro = new ResizeObserver(() => this._resize()); this._ro.observe(host);
      this._resize();
      this.setState({ state: "idle" });
      this._loop();
    }

    _resize() {
      const w = this.host.clientWidth || 600, h = this.host.clientHeight || 480;
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.dist = w < 520 ? 11 : 8.4;
      this.camera.updateProjectionMatrix();
    }

    _controls() {
      const el = this.renderer.domElement;
      el.style.touchAction = "none";
      let drag = null;
      const clampP = (p) => Math.max(-0.9, Math.min(1.15, p));
      el.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId); this.host.classList.add("grabbing"); });
      el.addEventListener("pointermove", (e) => {
        if (!drag) return;
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        drag = { x: e.clientX, y: e.clientY };
        this.vYaw = dx * 0.008;
        this.yaw += this.vYaw; this.pitch = clampP(this.pitch + dy * 0.006);
        this.lastInteract = performance.now();
      });
      const end = () => { drag = null; this.host.classList.remove("grabbing"); };
      el.addEventListener("pointerup", end); el.addEventListener("pointercancel", end);
      el.addEventListener("wheel", (e) => {
        e.preventDefault();
        this.zoom = Math.max(0.6, Math.min(1.6, this.zoom * (1 + Math.sign(e.deltaY) * 0.08)));
        this.lastInteract = performance.now();
      }, { passive: false });
      el.addEventListener("dblclick", () => { this.yaw = -0.65; this.pitch = 0.22; this.zoom = 1; this.vYaw = 0; });
      el.tabIndex = 0;
      el.setAttribute("aria-label", "3D aircraft model. Drag or use the arrow keys to rotate.");
      el.addEventListener("keydown", (e) => {
        const k = { ArrowLeft: [-0.15, 0], ArrowRight: [0.15, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1] }[e.key];
        if (k) { e.preventDefault(); this.yaw += k[0]; this.pitch = clampP(this.pitch + k[1]); this.lastInteract = performance.now(); }
      });
    }

    // ---- hooks used by hand-gesture control (hands_control.js)
    grab(dx, dy) {
      this.vYaw = dx; this.yaw += dx;
      this.pitch = Math.max(-0.9, Math.min(1.15, this.pitch + dy));
      this.lastInteract = performance.now();
    }
    setZoom(z) { this.zoom = Math.max(0.6, Math.min(1.6, z)); this.lastInteract = performance.now(); }
    resetView() { this.yaw = -0.65; this.pitch = 0.22; this.zoom = 1; this.vYaw = 0; this.bankTarget = 0; }
    setBank(a) { this.bankTarget = Math.max(-0.9, Math.min(0.9, a)); }
    // Place the aircraft under a point on the screen (normalised -1..1 device coordinates), or back in the centre.
    anchorTo(ndc) {
      if (!ndc) { this.anchorTarget.set(0, 0, 0); return; }
      this.ray.setFromCamera(new T.Vector2(ndc[0], ndc[1]), this.camera);
      const n = this.camera.getWorldDirection(new T.Vector3()).negate();
      const plane = new T.Plane(n, 0), hit = new T.Vector3();
      if (this.ray.ray.intersectPlane(plane, hit)) this.anchorTarget.copy(hit);
    }
    // Move the canvas into another element (the full-screen simulation) and back.
    attach(host, ar) {
      this.host = host; host.appendChild(this.renderer.domElement);
      this.ar = !!ar; this.ring.visible = !ar; this.shadow.visible = !ar;
      if (!ar) { this.anchorTarget.set(0, 0, 0); this.bankTarget = 0; }
      this._ro && this._ro.disconnect();
      this._ro = new ResizeObserver(() => this._resize()); this._ro.observe(host);
      this._resize();
    }

    _show(group) {
      while (this.pivot.children.length) this.pivot.remove(this.pivot.children[0]);
      this.model = group;
      if (group) this.pivot.add(group);
      this.flyT = 0;
    }

    _holo(cls, color, opacity) { return hologram(BUILD[cls]().group, color, opacity); }

    setState(s) {
      const token = ++this.token;
      this.state = s.state || "idle";
      this.cls = s.cls || "";
      if (!this.ar) this.host.dataset.state = this.state;
      this.ring.material.color.setHex(this.state === "low" ? AMBER : ACCENT);
      if (this.state === "result") {
        this.pitch = 0.22; this.zoom = 1; this.yaw = -0.65;
        this._show(BUILD[s.cls]().group);
        loadGLB(s.cls).then((obj) => { if (obj && token === this.token) this._show(obj.clone()); });
      } else if (this.state === "low") {
        this._show(this._holo(s.cls, AMBER, 0.55));
      } else if (this.state === "loading") {
        this._show(this._holo(ORDER[this.cycle % 4], 0xffffff, 0.4));
      } else if (this.state === "error") {
        this._show(null);
      } else {
        this.cycle = 0; this.cycleAt = performance.now();
        this._show(this._holo(ORDER[0], 0xffffff, 0.16));
      }
    }

    _loop() {
      requestAnimationFrame(() => this._loop());
      const dt = Math.min(this.clock.getDelta(), 0.05), t = this.clock.elapsedTime, now = performance.now();
      if (this.state === "idle" && now - this.cycleAt > 3400) {
        this.cycle++; this.cycleAt = now;
        this._show(this._holo(ORDER[this.cycle % 4], 0xffffff, 0.16));
      }
      if (this.state === "loading" && Math.floor(t * 2.5) % 4 !== this.cycle % 4) {
        this.cycle = Math.floor(t * 2.5) % 4;
        this._show(this._holo(ORDER[this.cycle], 0xffffff, 0.4));
        this.flyT = 1;
      }
      const idleFor = now - this.lastInteract;
      if (idleFor > 60 && idleFor < 1200) { this.yaw += this.vYaw * 0.9; this.vYaw *= 0.92; }
      if (!reduceMotion && idleFor > 2500 && !this.handActive) this.yaw += dt * (this.state === "loading" ? 1.6 : 0.25);

      this.flyT = Math.min(1, this.flyT + dt / 0.9);
      const e = 1 - Math.pow(1 - this.flyT, 3);
      this.pivot.scale.setScalar(0.6 + 0.4 * e);
      this.anchor.lerp(this.anchorTarget, 0.22);
      this.bank += (this.bankTarget - this.bank) * 0.15;
      this.pivot.rotation.set(0, this.yaw + (1 - e) * 2.2, 0);
      this.pivot.rotateOnWorldAxis(new T.Vector3(0, 0, 1), this.bank);
      this.pivot.position.set(this.anchor.x, this.anchor.y + (1 - e) * 0.7 + Math.sin(t * 1.1) * 0.035, this.anchor.z);

      const z = this.dist * this.zoom;
      this.camera.position.set(0, Math.sin(this.pitch) * z, Math.cos(this.pitch) * z);
      this.camera.lookAt(0, -0.15, 0);

      if (this.model) this.model.traverse((o) => {
        const sp = o.userData.spin;
        if (sp) o.rotation[sp.axis] += dt * sp.speed * (reduceMotion ? 0.15 : 1);
        if (o.userData.strobe) o.visible = (t % 1.2) < 0.08;
        if (o.userData.pulse && o.material) o.material.color.setHSL(0.04, 1, 0.45 + 0.12 * Math.sin(t * 18));
      });
      this.under.intensity = this.state === "result" ? 8 + Math.sin(t * 2) * 1.5 : 3;
      this.shadow.material.opacity = this.state === "result" ? 0.9 : 0.25;
      this.ring.rotation.z += dt * 0.15;
      this.renderer.render(this.scene, this.camera);
    }
  }

  // ============================================================== wiring
  // Gradio re-renders HTML blocks, so watch the page for the stage and the
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
      document.querySelectorAll("[data-mod-item]").forEach((n) => n.classList.toggle("on", n.dataset.modItem === st.dataset.mod));
    }
  }
  new MutationObserver(sync).observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", sync);
  sync();

  document.addEventListener("input", (e) => {
    if (e.target.classList && e.target.classList.contains("cmp-range"))
      e.target.closest(".cmp").style.setProperty("--x", e.target.value + "%");
  });
  window.AeroViewer = { get: () => viewer, show: (cls) => viewer && viewer.setState({ state: "result", cls }) };
})();
