// Car-display HUD bound to a Revisualizer instance.
const MPH = 2.23694;
const MI = 1 / 1609.34;

const NICE = {
  ring_front_center: "Front", ring_front_left: "Front Left", ring_front_right: "Front Right",
  ring_side_left: "Left Pillar", ring_side_right: "Right Pillar", ring_rear_left: "Left Rear", ring_rear_right: "Right Rear",
  CAM_FRONT: "Front", CAM_BACK: "Rear", CAM_FRONT_LEFT: "Front Left", CAM_FRONT_RIGHT: "Front Right", CAM_BACK_LEFT: "Left Rear", CAM_BACK_RIGHT: "Right Rear",
};
const nice = (c) => NICE[c] || c.replace(/_/g, " ");
const fmtMi = (m) => { const mi = m * MI; return mi < 0.1 ? `${Math.round((m * 3.28084) / 50) * 50} ft` : `${mi.toFixed(1)} mi`; };
const fmtClock = (d) => d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" }).toLowerCase();

export function bindHud(rv, root, { hud = true, t = 0 } = {}) {
  const $ = (id) => root.querySelector("#" + id);
  if (!hud) root.classList.add("nohud");
  if (t) rv.seek(t);
  $("loader").classList.add("done");
  const meta = rv.meta || {};
  $("credit").textContent = meta.source ? `Scene: ${meta.source}` : "";
  if (meta.street) $("navstreet").dataset.street = meta.street;
  if (meta.speed_limit_mph) $("limit").textContent = meta.speed_limit_mph;
  $("dest").textContent = meta.destination || meta.name || "Route end";

  const gears = [...root.querySelectorAll(".prnd span")];
  let streak = 0, lastT = 0, lastSecond = -1;
  rv.on("frame", (st) => {
    $("speed").textContent = Math.round(st.speedMps * MPH);
    const g = st.speedMps > 0.3 || st.mode === "goto" ? "D" : "P";
    gears.forEach((s) => s.classList.toggle("on", s.dataset.g === g));
    const dt = st.t - lastT; lastT = st.t;
    if (dt > 0 && dt < 0.5) streak += st.speedMps * dt;
    // nav
    const nt = st.nextTurn;
    const icon = $("turnicon");
    if (st.mode !== "replay") {
      $("navdist").textContent = fmtMi(st.remainingM);
      $("navstreet").textContent = st.mode === "parked" ? "Arrived at pin" : "Pinned location";
      icon.classList.remove("right", "straight");
    } else if (nt) {
      $("navdist").textContent = nt.distM < 3 ? "Now" : fmtMi(nt.distM);
      $("navstreet").textContent = (nt.dir === "left" ? "Turn left" : "Turn right") + ($("navstreet").dataset.street ? ` onto ${$("navstreet").dataset.street}` : "");
      icon.classList.toggle("right", nt.dir === "right");
      icon.classList.remove("straight");
    } else {
      $("navdist").textContent = fmtMi(st.remainingM);
      $("navstreet").textContent = "Continue straight";
      icon.classList.add("straight");
    }
    // once a second is enough for text that barely changes
    const sec = Math.floor(performance.now() / 1000);
    if (sec !== lastSecond) {
      lastSecond = sec;
      const now = new Date();
      $("clock").textContent = fmtClock(now);
      $("streak").textContent = `${(streak * MI).toFixed(1)} mi`;
      const v = Math.max(st.speedMps, 4);
      const remS = st.remainingM / v;
      $("eta").textContent = fmtClock(new Date(now.getTime() + remS * 1000));
      $("remtime").textContent = `${Math.max(1, Math.round(remS / 60))} min`;
      $("remdist").textContent = fmtMi(st.remainingM);
    }
    $("mprog").style.width = `${(20 + (st.t / st.duration) * 60).toFixed(1)}%`;
    if (!seeking) $("seek").value = String(Math.round((st.t / st.duration) * 1000));
    $("tlabel").textContent = `${st.t.toFixed(1)} s`;
    if (camMode) updateCamView(st.frame);
    if (renderMode !== "splat") updateGen(st.frame);
  });

  // replay controls
  let seeking = false;
  const seek = $("seek");
  seek.addEventListener("input", () => { seeking = true; rv.seek((seek.value / 1000) * rv.duration); });
  seek.addEventListener("change", () => { seeking = false; });
  $("play").addEventListener("click", () => rv.toggle());
  const syncPlay = () => root.classList.toggle("paused", !rv.playing);
  rv.on("play", syncPlay); rv.on("pause", syncPlay); syncPlay();

  // tap to go
  rv.on("goto", (e) => { $("toast").classList.remove("hidden"); $("toasttext").textContent = `Going to pin · ${Math.round(e.distance)} m`; });
  rv.on("arrived", () => { $("toasttext").textContent = "Arrived"; });
  rv.on("resume", () => $("toast").classList.add("hidden"));
  $("resume").addEventListener("click", () => rv.resumeRoute());
  $("endtrip").addEventListener("click", () => rv.resumeRoute());

  // night + views
  const nightBtn = $("nightbtn");
  nightBtn.classList.toggle("on", rv.night);
  nightBtn.addEventListener("click", () => { rv.setNight(!rv.night); nightBtn.classList.toggle("on", rv.night); });
  $("viewbtn").addEventListener("click", () => rv.cycleView());

  // camera strip, like the in-car dashcam viewer: Reality | Grid | each camera
  const cams = rv.tracks.cameras || [];
  const strip = $("strip"), camview = $("camview");
  let camMode = null; // null = reality, "grid", or a camera name
  const base = rv.sceneUrl + "/cams/";
  const pad = (n) => String(n).padStart(4, "0");
  const tiles = [["reality", "Reality"], ["grid", "Grid"], ...cams.map((c) => [c, nice(c)])];
  for (const [key, label] of tiles) {
    const el = document.createElement("div");
    el.className = "tile" + (key === "reality" ? " reality on" : key === "grid" ? " grid" : "");
    if (key === "grid") el.innerHTML = "<i></i>".repeat(6);
    else if (key !== "reality") el.style.backgroundImage = `url(${base}${key}/0000.jpg)`;
    el.insertAdjacentHTML("beforeend", `<span>${label}</span>`);
    el.addEventListener("click", () => setCam(key === "reality" ? null : key));
    el.dataset.key = key;
    strip.appendChild(el);
  }
  let lastCamFrame = -1;
  function setCam(m) {
    camMode = m;
    lastCamFrame = -1;
    strip.querySelectorAll(".tile").forEach((t) => t.classList.toggle("on", (t.dataset.key === "reality" && !m) || t.dataset.key === m));
    camview.classList.toggle("hidden", !m);
    camview.classList.toggle("grid", m === "grid");
    camview.innerHTML = "";
    if (m === "grid") camview.innerHTML = cams.map((c) => `<figure><img data-cam="${c}"><figcaption>${nice(c)}</figcaption></figure>`).join("");
    else if (m) camview.innerHTML = `<img data-cam="${m}">`;
    if (m) updateCamView(rv.state.frame);
  }
  function updateCamView(frame) {
    if (frame === lastCamFrame) return;
    lastCamFrame = frame;
    camview.querySelectorAll("img").forEach((img) => { img.src = `${base}${img.dataset.cam}/${pad(frame)}.jpg`; });
  }
  $("cambtn").addEventListener("click", () => { strip.classList.toggle("hidden"); $("cambtn").classList.toggle("on", !strip.classList.contains("hidden")); if (strip.classList.contains("hidden")) setCam(null); });

  // render mode: per-scene splat vs AI (pre-generated live-mode frames) vs split
  const genview = $("genview"), genimg = $("genimg"), splitbar = $("splitbar"), seg = $("modeseg");
  let renderMode = "splat", lastGen = -1, splitX = 0.5;
  // prefer the splat + I2I diffusion frames, fall back to the lidar-lift + I2I frames
  let genBase = rv.sceneUrl + "/gen/splat_i2i/";
  (async () => {
    for (const dir of ["decoder", "dggt_fixer", "splat_i2i", "live"]) {
      const r = await fetch(`${rv.sceneUrl}/gen/${dir}/0000.jpg`, { method: "HEAD" }).catch(() => null);
      if (r?.ok) { genBase = `${rv.sceneUrl}/gen/${dir}/`; seg.classList.remove("hidden"); return; }
    }
  })();
  function updateGen(frame) {
    if (frame === lastGen) return;
    lastGen = frame;
    genimg.src = genBase + pad(frame) + ".jpg";
  }
  function setMode(m) {
    renderMode = m;
    lastGen = -1;
    seg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.m === m));
    genview.classList.toggle("hidden", m === "splat");
    genview.classList.toggle("split", m === "split");
    splitbar.classList.toggle("hidden", m !== "split");
    if (m !== "splat") { rv.setView("reality"); updateGen(rv.state.frame); }
    rv.setAiMode(m === "splat" ? "off" : m === "gen" ? "full" : "split", splitX);
    applySplit();
  }
  function applySplit() {
    splitbar.style.left = `${splitX * 100}%`;
    if (renderMode === "split") rv.setAiMode("split", splitX);
  }
  seg.addEventListener("click", (e) => { const m = e.target.dataset?.m; if (m) setMode(m); });
  splitbar.addEventListener("pointerdown", (e) => {
    splitbar.setPointerCapture(e.pointerId);
    const move = (ev) => { splitX = Math.min(0.95, Math.max(0.05, ev.clientX / innerWidth)); applySplit(); };
    splitbar.addEventListener("pointermove", move);
    splitbar.addEventListener("pointerup", () => splitbar.removeEventListener("pointermove", move), { once: true });
  });
  rv.setRenderMode = setMode;

  addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); rv.toggle(); }
    else if (e.key === "ArrowRight") rv.seek(rv.t + 2);
    else if (e.key === "ArrowLeft") rv.seek(rv.t - 2);
    else if (e.key === "v" || e.key === "V") rv.cycleView();
    else if (e.key === "c" || e.key === "C") $("cambtn").click();
    else if (e.key === "n" || e.key === "N") nightBtn.click();
    else if (e.key === "h" || e.key === "H") root.classList.toggle("nohud");
    else if (e.key === "Escape") { rv.resumeRoute(); setCam(null); }
    else if (e.key === "g" || e.key === "G") setMode(renderMode === "splat" ? "gen" : renderMode === "gen" ? "split" : "splat");
    else if ("1234".includes(e.key)) rv.setView(["reality", "close", "top", "hood"][+e.key - 1]);
  });
}
