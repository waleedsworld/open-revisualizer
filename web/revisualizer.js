// Open Revisualizer: a game-style reality re-render for driving logs.
// Renders a Gaussian-splat reconstruction of a drive from a chase camera above
// and behind the car, with a planned-path ribbon, turn chevrons in the sky,
// and tap-to-go. MIT licence. See docs/SDK.md.
import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const UP = new THREE.Vector3(0, 0, 1);
const BLUE = new THREE.Color("#3e7bff");

export const VIEWS = {
  reality: { back: 10.5, up: 5.6, ahead: 14, fov: 52, lookUp: 0.6 },
  close: { back: 6.0, up: 2.6, ahead: 9, fov: 58, lookUp: 1.1 },
  top: { back: 3.0, up: 32, ahead: 6, fov: 50, lookUp: 0 },
  hood: { back: -1.7, up: 1.65, ahead: 25, fov: 70, lookUp: 1.4 },
};

/**
 * Create a viewer inside `container`.
 * @param {object} o
 * @param {HTMLElement} o.container
 * @param {string|null} [o.scene] scene folder URL (e.g. "scenes/av2-miami-turn"); null = scenes/index.json default
 * @param {boolean} [o.autoplay=true]
 * @param {number} [o.speed=1] playback speed multiplier
 * @param {keyof VIEWS} [o.view="reality"]
 * @param {boolean} [o.night=false]
 * @param {boolean} [o.tapToGo=true]
 * @param {boolean} [o.skySplat=false] load the scene's learned sky dome (sky.ply) instead of the gradient sky
 * @param {(p:number)=>void} [o.onProgress]
 * @returns {Promise<Revisualizer>}
 */
export async function createRevisualizer(o) {
  const rv = new Revisualizer(o);
  await rv.load(o.scene ?? null);
  return rv;
}

export class Revisualizer extends EventTarget {
  constructor(o) {
    super();
    this.opts = { autoplay: true, speed: 1, view: "reality", night: false, tapToGo: true, ...o };
    this.container = o.container;
    this.t = 0;
    this.playing = false;
    this.speed = this.opts.speed;
    this.mode = "replay"; // "replay" | "goto" | "parked"
    this.viewName = this.opts.view in VIEWS ? this.opts.view : "reality";
    this.orbit = { yaw: 0, pitch: 0, zoom: 1 };
    this._camPos = null;
    this._camLook = null;

    const r = (this.renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true, powerPreference: "high-performance" }));
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.setClearColor(0x000000, 1);
    this.container.appendChild(r.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 2000);
    this.camera.up.copy(UP);
    this.spark = new SparkRenderer({ renderer: r });
    this.scene.add(this.spark);

    this.scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x30302c, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 2.4);
    sun.position.set(-30, 20, 60);
    this.scene.add(sun);

    this._buildSky();
    this._buildOverlays();
    const pmrem = new THREE.PMREMGenerator(r);
    this.envMap = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.ego = buildCar(this.envMap);
    this.scene.add(this.ego);
    if (this.opts.carModel !== false) loadCarModel(this.opts.carModel || new URL("assets/car.glb", import.meta.url).href, this.envMap)
      .then((model) => { this.ego.userData.setBody(model); this.ego.userData.setLights(this.night); })
      .catch((e) => console.warn("car model not loaded, using the built-in car", e));

    this._resize = () => {
      const w = this.container.clientWidth || innerWidth, h = this.container.clientHeight || innerHeight;
      r.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    };
    addEventListener("resize", this._resize);
    this._resize();
    this._bindPointer();
    this.clock = new THREE.Clock();
    r.setAnimationLoop(() => this._tick());
  }

  // ---------------------------------------------------------------- loading
  async load(sceneUrl) {
    if (!sceneUrl) {
      const idx = await fetch(new URL("scenes/index.json", location.href)).then((x) => x.json());
      sceneUrl = "scenes/" + idx.default;
    }
    this.sceneUrl = sceneUrl.replace(/\/$/, "");
    const base = new URL(this.sceneUrl + "/", location.href);
    const [tracks, meta] = await Promise.all([
      fetch(new URL("tracks.json", base)).then((x) => x.json()),
      fetch(new URL("meta.json", base)).then((x) => x.json()).catch(() => ({})),
    ]);
    this.tracks = tracks;
    this.meta = meta;
    this.fps = tracks.fps || 10;
    this.frames = tracks.frames;
    this.duration = (this.frames.length - 1) / this.fps;
    // the ego pose is the rear axle; drop the path to the road surface so the car, ribbon and pins sit on the road
    this.zoff = meta.road_z_below_ego ?? 0;
    this._prepTrack();

    const prog = { bg: 0, actors: 0 };
    const report = () => this.opts.onProgress?.(prog.bg * 0.85 + prog.actors * 0.15);
    this.background = new SplatMesh({
      url: new URL("background.ply", base).href,
      raycastable: this.opts.tapToGo,
      onProgress: (e) => { if (e.total) { prog.bg = e.loaded / e.total; report(); } },
    });
    this.scene.add(this.background);

    // optional learned sky dome
    if (this.opts.skySplat) fetch(new URL("sky.ply", base), { method: "HEAD" }).then((r) => {
      if (!r.ok) return;
      this.skySplat = new SplatMesh({ url: new URL("sky.ply", base).href });
      this.scene.add(this.skySplat);
    }).catch(() => {});

    const ids = new Set();
    for (const f of this.frames) for (const id in f.actors || {}) ids.add(id);
    this.actors = new Map();
    let done = 0;
    const actorLoads = [...ids].map(async (id) => {
      const head = await fetch(new URL(`actors/${id}.ply`, base), { method: "HEAD" }).catch(() => null);
      if (!head || !head.ok) return;
      const m = new SplatMesh({ url: new URL(`actors/${id}.ply`, base).href });
      m.visible = false;
      this.scene.add(m);
      this.actors.set(id, m);
      await m.initialized;
      prog.actors = ++done / ids.size;
      report();
    });
    await Promise.all([this.background.initialized, ...actorLoads]);
    this.setNight(this.opts.night);
    this.setView(this.viewName);
    this.seek(0);
    if (this.opts.autoplay) this.play();
    this._emit("loaded", { meta, duration: this.duration, cameras: tracks.cameras || [] });
    return this;
  }

  _prepTrack() {
    const F = this.frames;
    this.egoPts = F.map((f) => new THREE.Vector3(f.ego.p[0], f.ego.p[1], f.ego.p[2] + this.zoff));
    this.egoQs = F.map((f) => new THREE.Quaternion(f.ego.q[1], f.ego.q[2], f.ego.q[3], f.ego.q[0]));
    this.cum = [0];
    for (let i = 1; i < F.length; i++) this.cum.push(this.cum[i - 1] + this.egoPts[i].distanceTo(this.egoPts[i - 1]));
    this.totalDist = this.cum[this.cum.length - 1];
    // unwrapped yaw per frame
    let prev = 0;
    this.yaws = this.egoQs.map((q, i) => {
      let y = yawOf(q);
      if (i) { while (y - prev > Math.PI) y -= 2 * Math.PI; while (y - prev < -Math.PI) y += 2 * Math.PI; }
      return (prev = y);
    });
    // turns: stretches where heading changes faster than 8 deg per 10 m, total > 25 deg
    const turns = [];
    const rate = (i) => {
      const a = this._idxAtDist(this.cum[i] - 5), b = this._idxAtDist(this.cum[i] + 5);
      return (this.yaws[b] - this.yaws[a]) / Math.max(this.cum[b] - this.cum[a], 1) * 10;
    };
    const TH = (8 * Math.PI) / 180;
    let cur = null;
    for (let i = 0; i < F.length; i++) {
      const r = rate(i);
      if (Math.abs(r) > TH) {
        if (cur && this.cum[i] - cur.end < 10) cur.end = this.cum[i];
        else { if (cur) turns.push(cur); cur = { start: this.cum[i], end: this.cum[i], i0: i, i1: i }; }
        cur.i1 = i;
      }
    }
    if (cur) turns.push(cur);
    for (const t of turns) t.dir = Math.sign(this.yaws[t.i1] - this.yaws[t.i0]);
    const keep = turns.filter((t) => Math.abs(this.yaws[t.i1] - this.yaws[t.i0]) > (25 * Math.PI) / 180);
    turns.length = 0; turns.push(...keep);
    this.turns = turns.map((t) => ({ ...t, dir: t.dir > 0 ? "left" : "right" }));
  }

  _idxAtDist(d) {
    const c = this.cum;
    let lo = 0, hi = c.length - 1;
    if (d >= c[hi]) return hi;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (c[m] < d) lo = m; else hi = m; }
    return hi;
  }

  /** world position on the recorded ego path at arc length d */
  pointAtDist(d, out = new THREE.Vector3()) {
    d = Math.max(0, Math.min(this.totalDist, d));
    const i = Math.max(1, this._idxAtDist(d));
    const span = this.cum[i] - this.cum[i - 1] || 1;
    return out.lerpVectors(this.egoPts[i - 1], this.egoPts[i], (d - this.cum[i - 1]) / span);
  }

  /** interpolated recorded state at time t */
  trackAt(t) {
    const x = Math.max(0, Math.min(this.frames.length - 1, t * this.fps));
    const i = Math.floor(x), j = Math.min(i + 1, this.frames.length - 1), a = x - i;
    const p = new THREE.Vector3().lerpVectors(this.egoPts[i], this.egoPts[j], a);
    const q = new THREE.Quaternion().slerpQuaternions(this.egoQs[i], this.egoQs[j], a);
    const fi = this.frames[i], fj = this.frames[j];
    const speed = (fi.speed_mps ?? 0) * (1 - a) + (fj.speed_mps ?? 0) * a;
    const dist = this.cum[i] * (1 - a) + this.cum[j] * a;
    return { p, q, speed, dist, frame: Math.round(x), a, i, j };
  }

  // ---------------------------------------------------------------- playback
  play() { this.playing = true; this._emit("play"); }
  pause() { this.playing = false; this._emit("pause"); }
  toggle() { this.playing ? this.pause() : this.play(); }
  seek(t) { this.t = Math.max(0, Math.min(this.duration, t)); if (this.mode !== "replay") this.resumeRoute(); this._snapCam = true; }
  setSpeed(s) { this.speed = s; }
  setView(name) { if (!VIEWS[name]) return; this.viewName = name; this.orbit = { yaw: 0, pitch: 0, zoom: 1 }; this._snapCam = true; this._emit("view", { view: name }); }
  cycleView() { const k = Object.keys(VIEWS); this.setView(k[(k.indexOf(this.viewName) + 1) % k.length]); }

  setNight(on) {
    this.night = !!on;
    const c = on ? new THREE.Color(0.42, 0.47, 0.66) : new THREE.Color(1, 1, 1);
    if (this.background) this.background.recolor = c;
    if (this.skySplat) this.skySplat.visible = !on;
    for (const m of this.actors?.values() || []) m.recolor = c;
    this.sky.material.uniforms.night.value = on ? 1 : 0;
    this.ground.material.uniforms.night.value = on ? 1 : 0;
    this.ego.userData.setLights(on);
    this._emit("night", { night: this.night });
  }

  // ---------------------------------------------------------------- tap to go
  /** Drive to a world point (x, y). The car leaves the recorded path and drives there. */
  goTo(x, y) {
    const cur = this.state;
    const p0 = cur.ego.p.clone();
    const p3 = new THREE.Vector3(x, y, p0.z);
    const fwd = new THREE.Vector3(Math.cos(cur.ego.yaw), Math.sin(cur.ego.yaw), 0);
    const L = p0.distanceTo(p3);
    if (L < 2) return false;
    const toT = p3.clone().sub(p0).normalize();
    const p1 = p0.clone().addScaledVector(fwd, L * 0.45);
    const p2 = p3.clone().addScaledVector(toT, -L * 0.3);
    const curve = new THREE.CubicBezierCurve3(p0, p1, p2, p3);
    const pts = curve.getSpacedPoints(Math.max(20, Math.ceil(L)));
    // follow ground height from the recorded path near each point
    for (const q of pts) q.z = this._groundZNear(q);
    const cum = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + pts[i].distanceTo(pts[i - 1]));
    this.goto = { pts, cum, len: cum[cum.length - 1], s: 0, v: Math.max(cur.speedMps, 2), target: p3 };
    this.mode = "goto";
    this.marker.position.copy(p3);
    this.marker.position.z = this._groundZNear(p3) + 0.04;
    this.marker.visible = true;
    this._emit("goto", { x, y, distance: this.goto.len });
    return true;
  }

  /** Tap at canvas client coordinates (what a finger tap does). */
  tapAt(clientX, clientY) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    let hit = null;
    if (this.background?.raycastable) {
      const hits = [];
      try { this.background.raycast(ray, hits); } catch { /* raycast index may not be ready */ }
      hits.sort((a, b) => a.distance - b.distance);
      const egoZ = this.state.ego.p.z;
      hit = hits.find((h) => h.point.z < egoZ + 1.2)?.point ?? null;
    }
    if (!hit) {
      const plane = new THREE.Plane(UP, -this.state.ego.p.z);
      hit = ray.ray.intersectPlane(plane, new THREE.Vector3());
    }
    if (!hit) return false;
    const d = hit.distanceTo(this.state.ego.p);
    if (d < 3 || d > 150) return false;
    return this.goTo(hit.x, hit.y);
  }

  resumeRoute() {
    this.mode = "replay";
    this.goto = null;
    this.marker.visible = false;
    this._snapCam = true;
    this._emit("resume");
  }

  _groundZNear(p) {
    // nearest recorded ego point gives a good local road height
    let best = Infinity, z = p.z;
    for (let i = 0; i < this.egoPts.length; i += 2) {
      const e = this.egoPts[i];
      const d = (e.x - p.x) ** 2 + (e.y - p.y) ** 2;
      if (d < best) { best = d; z = e.z; }
    }
    return z;
  }

  // ---------------------------------------------------------------- state
  /** Current state: time, ego pose, speed, next turn, remaining distance. */
  get state() { return this._state || this._computeState(); }

  _computeState() {
    const tr = this.trackAt(this.t);
    let p = tr.p, yaw = yawOf(tr.q), speed = tr.speed, dist = tr.dist, q = tr.q;
    if (this.mode !== "replay" && this.goto) {
      const g = this.goto;
      const i = Math.max(1, lowerBound(g.cum, g.s));
      const a = (g.s - g.cum[i - 1]) / (g.cum[i] - g.cum[i - 1] || 1);
      p = new THREE.Vector3().lerpVectors(g.pts[i - 1], g.pts[i], a);
      const dir = g.pts[Math.min(i, g.pts.length - 1)].clone().sub(g.pts[i - 1]);
      yaw = Math.atan2(dir.y, dir.x);
      q = new THREE.Quaternion().setFromAxisAngle(UP, yaw);
      speed = this.mode === "parked" ? 0 : g.v;
    }
    let nextTurn = null;
    if (this.mode === "replay") {
      const t = this.turns.find((t) => t.end > dist + 2);
      if (t) nextTurn = { dir: t.dir, distM: Math.max(0, t.start - dist), start: t.start, end: t.end };
    }
    const remainingM = this.mode === "replay" ? this.totalDist - dist : this.goto ? this.goto.len - this.goto.s : 0;
    return (this._state = { t: this.t, frame: tr.frame, mode: this.mode, ego: { p, q, yaw }, speedMps: speed, dist, nextTurn, remainingM, duration: this.duration });
  }

  // ---------------------------------------------------------------- frame loop
  _tick() {
    const dt = Math.min(this.clock.getDelta(), 0.1);
    if (!this.frames) { this.renderer.render(this.scene, this.camera); return; }
    if (this.playing) {
      this.t += dt * this.speed;
      if (this.t > this.duration) { this.t = 0; if (this.mode === "replay") this._snapCam = true; }
    }
    if (this.mode === "goto" && this.goto && this.playing) {
      const g = this.goto, rem = g.len - g.s;
      const vmax = 9, acc = 2.5;
      const vTarget = Math.min(vmax, Math.sqrt(2 * acc * Math.max(rem, 0)) + 0.3);
      g.v += Math.sign(vTarget - g.v) * Math.min(Math.abs(vTarget - g.v), acc * dt * 2);
      g.s = Math.min(g.len, g.s + g.v * dt * this.speed);
      if (g.len - g.s < 0.05) { this.mode = "parked"; g.v = 0; this._emit("arrived", { x: g.target.x, y: g.target.y }); }
    }
    this._state = null;
    const st = this._computeState();

    // ego car
    this.ego.position.copy(st.ego.p);
    this.ego.quaternion.copy(st.ego.q);
    this.ego.userData.spinWheels(st.speedMps * dt * this.speed);

    // actors
    const f = this.frames[st.frame];
    const nf = this.frames[Math.min(st.frame + 1, this.frames.length - 1)];
    for (const [id, m] of this.actors) {
      const a = f.actors?.[id];
      if (!a) { m.visible = false; continue; }
      const b = nf.actors?.[id] || a;
      const x = Math.max(0, Math.min(1, this.t * this.fps - st.frame));
      m.visible = true;
      m.position.set(...a.p).lerp(new THREE.Vector3(...b.p), x);
      const qa = new THREE.Quaternion(a.q[1], a.q[2], a.q[3], a.q[0]);
      const qb = new THREE.Quaternion(b.q[1], b.q[2], b.q[3], b.q[0]);
      m.quaternion.slerpQuaternions(qa, qb, x);
    }

    this._updateRibbon(st);
    this._updateChevrons(st, dt);
    this._updateMarker(dt);
    this._updateCamera(st, dt);
    this.sky.position.copy(this.camera.position);
    this.ground.position.set(this.camera.position.x, this.camera.position.y, st.ego.p.z - 1.2);
    this.renderer.render(this.scene, this.camera);
    this._emit("frame", st);
  }

  _updateCamera(st, dt) {
    const v = VIEWS[this.viewName];
    const yaw = st.ego.yaw + this.orbit.yaw;
    const fwd = new THREE.Vector3(Math.cos(yaw), Math.sin(yaw), 0);
    const z = this.orbit.zoom;
    const pitchLift = Math.tan(this.orbit.pitch) * v.back;
    const base = st.ego.p.clone();
    base.z -= this.zoff || 0; // camera offsets are defined from the raw ego pose
    const pos = base.clone().addScaledVector(fwd, -v.back * z).addScaledVector(UP, v.up * z + pitchLift);
    const hfwd = new THREE.Vector3(Math.cos(st.ego.yaw), Math.sin(st.ego.yaw), 0);
    const look = base.clone().addScaledVector(this.viewName === "hood" ? hfwd : fwd, v.ahead).addScaledVector(UP, v.lookUp);
    const k = this._snapCam || !this._camPos ? 1 : 1 - Math.exp(-dt * 5);
    this._camPos = (this._camPos || pos.clone()).lerp(pos, k);
    this._camLook = (this._camLook || look.clone()).lerp(look, k);
    this._snapCam = false;
    this.camera.position.copy(this._camPos);
    this.camera.lookAt(this._camLook);
    if (this.camera.fov !== v.fov) { this.camera.fov = v.fov; this.camera.updateProjectionMatrix(); }
  }

  // ---------------------------------------------------------------- overlays
  _buildSky() {
    const mat = new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false,
      uniforms: { night: { value: 0 } },
      vertexShader: `varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `uniform float night; varying vec3 vDir;
        void main(){ float h = clamp(vDir.z, -0.2, 1.0);
          vec3 dayZ = vec3(0.27,0.47,0.80), dayH = vec3(0.80,0.86,0.93), dayG = vec3(0.36,0.37,0.38);
          vec3 nZ = vec3(0.01,0.015,0.04), nH = vec3(0.09,0.11,0.18), nG = vec3(0.03,0.03,0.04);
          vec3 z = mix(dayZ, nZ, night), hz = mix(dayH, nH, night), g = mix(dayG, nG, night);
          vec3 c = h > 0.0 ? mix(hz, z, pow(h, 0.55)) : mix(hz, g, clamp(-h*6.0,0.0,1.0));
          gl_FragColor = vec4(c, 1.0); }`,
    });
    this.sky = new THREE.Mesh(new THREE.SphereGeometry(1500, 32, 16), mat);
    this.sky.renderOrder = -10;
    this.scene.add(this.sky);
    const gmat = new THREE.ShaderMaterial({
      uniforms: { night: { value: 0 } },
      vertexShader: `varying vec2 vP; void main(){ vP = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `uniform float night; varying vec2 vP;
        void main(){ float r = length(vP) / 900.0;
          vec3 near = mix(vec3(0.20,0.205,0.21), vec3(0.025,0.027,0.032), night);
          vec3 far = mix(vec3(0.80,0.86,0.93), vec3(0.09,0.11,0.18), night);
          gl_FragColor = vec4(mix(near, far, smoothstep(0.02, 0.6, r)), 1.0); }`,
    });
    this.ground = new THREE.Mesh(new THREE.CircleGeometry(1400, 48), gmat);
    this.ground.renderOrder = -9;
    this.scene.add(this.ground);
  }

  _buildOverlays() {
    // planned path ribbon
    const N = 90;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 2 * 3), 3));
    g.setAttribute("uv", new THREE.BufferAttribute(new Float32Array(N * 2 * 2), 2));
    const idx = [];
    for (let i = 0; i < N - 1; i++) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    g.setIndex(idx);
    this.ribbon = new THREE.Mesh(g, new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, depthTest: true, blending: THREE.NormalBlending,
      uniforms: { color: { value: BLUE }, time: { value: 0 } },
      vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `uniform vec3 color; uniform float time; varying vec2 vUv;
        void main(){ float across = abs(vUv.y*2.0-1.0);
          float edge = 1.0 - smoothstep(0.82, 1.0, across);
          float fadeIn = smoothstep(0.0, 0.04, vUv.x);
          float fadeOut = 1.0 - smoothstep(0.55, 1.0, vUv.x);
          float rim = smoothstep(0.7, 0.95, across) * edge * 0.35;
          vec3 c = color * (0.85 + rim) + vec3(0.05,0.08,0.15);
          gl_FragColor = vec4(c, 0.78 * edge * fadeIn * fadeOut); }`,
    }));
    this.ribbon.frustumCulled = false;
    this.ribbon.renderOrder = 5;
    this.scene.add(this.ribbon);
    this._ribbonN = N;

    // chevrons floating in the sky ahead of turns
    this.chevTex = { right: chevronTexture(false), left: chevronTexture(true) };
    this.chevrons = [];
    for (let i = 0; i < 10; i++) {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.chevTex.right, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending }));
      s.renderOrder = 6;
      s.visible = false;
      this.scene.add(s);
      this.chevrons.push(s);
    }

    // tap-to-go destination marker
    const m = new THREE.Group();
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1.25, 48), new THREE.MeshBasicMaterial({ color: BLUE, transparent: true, opacity: 0.9, depthTest: false, depthWrite: false }));
    const disc = new THREE.Mesh(new THREE.CircleGeometry(0.55, 32), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95, depthTest: false, depthWrite: false }));
    const pulse = new THREE.Mesh(new THREE.RingGeometry(1.2, 1.4, 48), new THREE.MeshBasicMaterial({ color: BLUE, transparent: true, opacity: 0.6, depthTest: false, depthWrite: false }));
    const beamMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
      uniforms: { color: { value: BLUE } },
      vertexShader: `varying float vH; void main(){ vH = uv.y; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `uniform vec3 color; varying float vH; void main(){ gl_FragColor = vec4(color*1.4, (1.0-vH)*0.7); }`,
    });
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 6, 12, 1, true), beamMat);
    beam.rotation.x = Math.PI / 2;
    beam.position.z = 3;
    for (const o of [ring, disc, pulse, beam]) { o.renderOrder = 7; m.add(o); }
    m.userData = { pulse };
    m.visible = false;
    this.marker = m;
    this.scene.add(m);
  }

  _updateRibbon(st) {
    const N = this._ribbonN;
    const pos = this.ribbon.geometry.attributes.position.array;
    const uv = this.ribbon.geometry.attributes.uv.array;
    const pts = [];
    if (this.mode === "replay") {
      const L = Math.min(48, this.totalDist - st.dist);
      if (L < 1) { this.ribbon.visible = false; return; }
      for (let i = 0; i < N; i++) pts.push(this.pointAtDist(st.dist + 0.8 + (L * i) / (N - 1)));
    } else if (this.goto && this.mode === "goto") {
      const g = this.goto, L = g.len - g.s;
      if (L < 0.5) { this.ribbon.visible = false; return; }
      for (let i = 0; i < N; i++) {
        const s = g.s + 0.8 + (L * i) / (N - 1);
        const k = Math.max(1, lowerBound(g.cum, s));
        const a = Math.min(1, (s - g.cum[k - 1]) / (g.cum[k] - g.cum[k - 1] || 1));
        pts.push(new THREE.Vector3().lerpVectors(g.pts[k - 1], g.pts[Math.min(k, g.pts.length - 1)], a));
      }
    } else { this.ribbon.visible = false; return; }
    this.ribbon.visible = true;
    const W = 1.0;
    for (let i = 0; i < N; i++) {
      const p = pts[i], q = pts[Math.min(i + 1, N - 1)], o = pts[Math.max(i - 1, 0)];
      const d = new THREE.Vector3().subVectors(q, o).setZ(0).normalize();
      const n = new THREE.Vector3(-d.y, d.x, 0);
      const z = p.z + 0.06;
      pos.set([p.x + n.x * W, p.y + n.y * W, z, p.x - n.x * W, p.y - n.y * W, z], i * 6);
      uv.set([i / (N - 1), 0, i / (N - 1), 1], i * 4);
    }
    this.ribbon.geometry.attributes.position.needsUpdate = true;
    this.ribbon.geometry.attributes.uv.needsUpdate = true;
  }

  _updateChevrons(st, dt) {
    const t = st.nextTurn;
    const show = t && t.distM < 120 && this.mode === "replay";
    this._chevT = (this._chevT || 0) + dt;
    const n = this.chevrons.length;
    for (let i = 0; i < n; i++) {
      const s = this.chevrons[i];
      if (!show) { s.visible = false; continue; }
      const d = t.start + 2 + i * Math.max(5, (t.end - t.start + 20) / n);
      if (d < st.dist + 4) { s.visible = false; continue; }
      const p = this.pointAtDist(d);
      s.visible = true;
      s.material.map = this.chevTex[t.dir];
      s.position.set(p.x, p.y, p.z + 3.4);
      const wave = 0.55 + 0.45 * Math.max(0, Math.sin(this._chevT * 5 - i * 0.7));
      const near = Math.min(1, Math.max(0, (120 - t.distM) / 30));
      s.material.opacity = wave * near;
      s.scale.set(2.4, 2.4, 1);
    }
  }

  _updateMarker(dt) {
    if (!this.marker.visible) return;
    const p = this.marker.userData.pulse;
    this._mk = ((this._mk || 0) + dt * 0.9) % 1;
    p.scale.setScalar(1 + this._mk * 1.8);
    p.material.opacity = 0.6 * (1 - this._mk);
  }

  // ---------------------------------------------------------------- input
  _bindPointer() {
    const el = this.renderer.domElement;
    let down = null;
    el.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY, yaw: this.orbit.yaw, pitch: this.orbit.pitch, moved: false }; el.setPointerCapture(e.pointerId); });
    el.addEventListener("pointermove", (e) => {
      if (!down) return;
      const dx = e.clientX - down.x, dy = e.clientY - down.y;
      if (Math.hypot(dx, dy) > 6) down.moved = true;
      if (down.moved) {
        this.orbit.yaw = down.yaw - dx * 0.006;
        this.orbit.pitch = Math.max(-0.35, Math.min(0.9, down.pitch + dy * 0.004));
      }
    });
    el.addEventListener("pointerup", (e) => {
      if (down && !down.moved && this.opts.tapToGo) this.tapAt(e.clientX, e.clientY);
      down = null;
    });
    el.addEventListener("dblclick", () => { this.orbit = { yaw: 0, pitch: 0, zoom: 1 }; });
    el.addEventListener("wheel", (e) => { e.preventDefault(); this.orbit.zoom = Math.max(0.4, Math.min(3, this.orbit.zoom * Math.exp(e.deltaY * 0.001))); }, { passive: false });
  }

  /** PNG data URL of the current frame */
  screenshot() { return this.renderer.domElement.toDataURL("image/png"); }

  dispose() {
    this.renderer.setAnimationLoop(null);
    removeEventListener("resize", this._resize);
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  on(type, fn) { const h = (e) => fn(e.detail); this.addEventListener(type, h); return () => this.removeEventListener(type, h); }
  _emit(type, detail) { this.dispatchEvent(new CustomEvent(type, { detail })); }
}

// ------------------------------------------------------------------ helpers
function yawOf(q) { return Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)); }
function lowerBound(a, v) { let lo = 0, hi = a.length - 1; if (v >= a[hi]) return hi; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (a[m] < v) lo = m; else hi = m; } return hi; }

function chevronTexture(left) {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  g.translate(128, 128);
  if (left) g.scale(-1, 1);
  const path = () => { g.beginPath(); g.moveTo(-58, -92); g.lineTo(42, 0); g.lineTo(-58, 92); };
  g.lineJoin = "miter"; g.lineCap = "butt";
  g.shadowColor = "rgba(80,150,255,1)";
  for (const [w, col, blur] of [[46, "rgba(60,120,255,0.55)", 40], [30, "rgba(150,200,255,0.9)", 18], [14, "rgba(255,255,255,1)", 6]]) {
    g.shadowBlur = blur; g.strokeStyle = col; g.lineWidth = w; path(); g.stroke();
  }
  // hollow centre stripe like the in-car chevrons
  g.shadowBlur = 0; g.globalCompositeOperation = "destination-out"; g.lineWidth = 5; path(); g.stroke();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** Load a glTF car, turn it z-up with +x forward, scale to 4.75 m, origin at the rear axle. */
async function loadCarModel(url, env) {
  const gltf = await new GLTFLoader().loadAsync(url);
  const root = gltf.scene;
  const mats = await gltf.parser.getDependencies("material");
  const byName = Object.fromEntries(mats.filter((m) => m.name).map((m) => [m.name, m]));
  root.traverse((o) => {
    if (!o.isMesh) return;
    const n = o.material.name || "";
    if (n.includes("Carmine")) o.material = byName[n.replace("Carmine", "Graphite")] || o.material;
    o.material.envMap = env;
    if (n.startsWith("Paint")) { o.material.envMapIntensity = 1.2; }
  });
  const holder = new THREE.Group();
  root.rotation.x = Math.PI / 2; // glTF is y-up
  holder.add(root);
  holder.updateMatrixWorld(true);
  let box = new THREE.Box3().setFromObject(holder);
  const size = box.getSize(new THREE.Vector3());
  // find the front from the headlights
  const head = new THREE.Box3();
  root.traverse((o) => { if (o.isMesh && /Headlight/i.test(o.material.name || "")) head.union(new THREE.Box3().setFromObject(o)); });
  const c = box.getCenter(new THREE.Vector3());
  const hc = head.isEmpty() ? c.clone().add(new THREE.Vector3(size.x > size.y ? 1 : 0, size.y >= size.x ? 1 : 0, 0)) : head.getCenter(new THREE.Vector3());
  const dir = hc.sub(c).setZ(0);
  holder.rotation.z = -Math.atan2(dir.y, dir.x);
  holder.updateMatrixWorld(true);
  box = new THREE.Box3().setFromObject(holder);
  const len = box.max.x - box.min.x;
  const k = 4.75 / len;
  holder.scale.setScalar(k);
  holder.updateMatrixWorld(true);
  box = new THREE.Box3().setFromObject(holder);
  const out = new THREE.Group();
  out.add(holder);
  // rear bumper sits 1.05 m behind the rear axle; wheels on the ground
  holder.position.set(-box.min.x - 1.05, -(box.min.y + box.max.y) / 2, -box.min.z);
  return out;
}

function taper(geo) {
  // narrow the cabin and round the shoulders so the car does not read as a box
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const z = p.getZ(i), x = p.getX(i);
    const k = z > 0.95 ? 1 - Math.min(1, (z - 0.95) / 0.5) * 0.22 : 1 - Math.max(0, 0.45 - z) * 0.12;
    const ends = 1 - Math.max(0, Math.abs(x - 1.37) - 1.9) * 0.35;
    p.setY(i, p.getY(i) * k * ends);
  }
  geo.computeVertexNormals();
}

/** A sleek low-poly sedan, origin at the rear axle on the ground, +x forward. */
function buildCar(env) {
  const car = new THREE.Group();
  const prof = [[-1.02, 0.3], [-1.1, 0.56], [-1.04, 0.84], [-0.7, 0.96], [1.9, 1.0], [2.95, 0.93], [3.66, 0.78], [3.80, 0.52], [3.72, 0.3]];
  const sh = new THREE.Shape(prof.map(([x, z]) => new THREE.Vector2(x, z)));
  const body = new THREE.ExtrudeGeometry(sh, { depth: 1.78, bevelEnabled: true, bevelThickness: 0.08, bevelSize: 0.08, bevelSegments: 4, curveSegments: 4 });
  body.translate(0, 0, -0.89);
  body.rotateX(Math.PI / 2);
  taper(body);
  const paint = new THREE.MeshPhysicalMaterial({ color: 0x2a2d33, metalness: 0.55, roughness: 0.28, clearcoat: 1, clearcoatRoughness: 0.08, envMap: env, envMapIntensity: 1.1 });
  car.add(new THREE.Mesh(body, paint));
  const glassShape = new THREE.Shape([[-0.72, 0.95], [0.1, 1.40], [1.2, 1.46], [1.98, 1.0], [-0.72, 0.95]].map(([x, z]) => new THREE.Vector2(x, z)));
  const glass = new THREE.ExtrudeGeometry(glassShape, { depth: 1.6, bevelEnabled: true, bevelThickness: 0.06, bevelSize: 0.06, bevelSegments: 3 });
  glass.translate(0, 0, -0.8);
  glass.rotateX(Math.PI / 2);
  taper(glass);
  car.add(new THREE.Mesh(glass, new THREE.MeshPhysicalMaterial({ color: 0x07090c, metalness: 0.1, roughness: 0.04, clearcoat: 1, envMap: env, envMapIntensity: 1.6 })));
  // light bars
  const tail = new THREE.Mesh(new THREE.BoxGeometry(0.06, 1.72, 0.07), new THREE.MeshBasicMaterial({ color: 0xff2a2a }));
  tail.position.set(-1.13, 0, 0.8);
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.05, 1.5, 0.05), new THREE.MeshBasicMaterial({ color: 0xdfe9ff }));
  head.position.set(3.82, 0, 0.7);
  car.add(tail, head);
  // wheels
  const wheels = [];
  const tire = new THREE.CylinderGeometry(0.35, 0.35, 0.26, 24);
  const tmat = new THREE.MeshStandardMaterial({ color: 0x0b0b0c, roughness: 0.9 });
  const rim = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.24, 0.27, 10), new THREE.MeshStandardMaterial({ color: 0x8a8d93, metalness: 0.9, roughness: 0.3, envMap: env }));
  for (const [x, y] of [[0, 0.86], [0, -0.86], [2.9, 0.86], [2.9, -0.86]]) {
    const w = new THREE.Group();
    w.add(new THREE.Mesh(tire, tmat), rim.clone());
    w.position.set(x, y, 0.35);
    car.add(w);
    wheels.push(w);
  }
  // soft contact shadow
  const sc = document.createElement("canvas"); sc.width = sc.height = 64;
  const sg = sc.getContext("2d"); const gr = sg.createRadialGradient(32, 32, 4, 32, 32, 32);
  gr.addColorStop(0, "rgba(0,0,0,0.75)"); gr.addColorStop(1, "rgba(0,0,0,0)"); sg.fillStyle = gr; sg.fillRect(0, 0, 64, 64);
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(5.6, 2.6), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(sc), transparent: true, depthWrite: false }));
  shadow.position.set(1.35, 0, 0.03);
  car.add(shadow);
  const beam = new THREE.SpotLight(0xdfe9ff, 0, 40, 0.45, 0.6, 1.2);
  beam.position.set(3.8, 0, 0.7);
  beam.target.position.set(15, 0, 0);
  car.add(beam, beam.target);
  car.userData.spinWheels = (d) => { for (const w of wheels) w.rotation.y += d / 0.35; };
  const builtin = car.children.filter((c) => c !== shadow && c !== beam && c !== beam.target);
  car.userData.setBody = (model) => {
    for (const c of builtin) car.remove(c);
    wheels.length = 0;
    car.add(model);
  };
  car.userData.setLights = (on) => {
    beam.intensity = on ? 60 : 0;
    car.traverse((o) => {
      if (!o.isMesh || !o.material.envMap) return;
      o.material.userData.env ??= o.material.envMapIntensity;
      o.material.envMapIntensity = o.material.userData.env * (on ? 0.3 : 1);
    });
  };
  return car;
}
