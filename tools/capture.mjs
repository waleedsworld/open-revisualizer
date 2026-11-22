// Capture stills or a frame sequence from the viewer with a real GPU browser.
//   node tools/capture.mjs shot  <url> out.png [--w 1600 --h 1000 --t 3]
//   node tools/capture.mjs frames <url> outdir  [--w 1280 --h 800 --fps 15 --seconds 8 --t 0 --script steps.json]
// steps.json: [{"at": 2.0, "tap": [0.62, 0.55]}, {"at": 5, "view": "top"}, {"at": 6, "night": true}, {"at": 7, "cams": true}]
// Frames are captured deterministically: playback is paused and stepped by 1/fps per frame.
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";

const [mode, url, out, ...rest] = process.argv.slice(2);
const arg = (k, d) => { const i = rest.indexOf("--" + k); return i >= 0 ? rest[i + 1] : d; };
const W = +arg("w", mode === "shot" ? 1600 : 1280), H = +arg("h", mode === "shot" ? 1000 : 800);
const headed = rest.includes("--headed");

const browser = await chromium.launch({
  headless: !headed,
  args: ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-webgl", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
page.on("console", (m) => { if (m.type() === "error") console.error("[page]", m.text()); });
page.on("pageerror", (e) => console.error("[pageerror]", e.message));
await page.goto(url);
await page.waitForFunction(() => window.revis && window.revis.frames, null, { timeout: 180000 });
await page.waitForTimeout(1500);

if (mode === "shot") {
  const t = +arg("t", 0);
  await page.evaluate((t) => { revis.pause(); revis.seek(t); }, t);
  await page.waitForTimeout(1200);
  await page.screenshot({ path: out });
} else {
  const fps = +arg("fps", 15), secs = +arg("seconds", 8), t0 = +arg("t", 0);
  const steps = arg("script") ? JSON.parse(fs.readFileSync(arg("script"), "utf8")) : [];
  fs.mkdirSync(out, { recursive: true });
  await page.evaluate((t0) => { revis.pause(); revis.seek(t0); revis.clock.getDelta(); }, t0);
  await page.waitForTimeout(800);
  const n = Math.round(fps * secs);
  for (let i = 0; i < n; i++) {
    const tt = i / fps;
    for (const s of steps.filter((s) => !s.done && s.at <= tt)) {
      s.done = true;
      await page.evaluate((s) => {
        if (s.tap) { const c = revis.renderer.domElement.getBoundingClientRect(); revis.tapAt(c.left + s.tap[0] * c.width, c.top + s.tap[1] * c.height); }
        if (s.goto) revis.goTo(...s.goto);
        if (s.view) revis.setView(s.view);
        if (s.night !== undefined) document.getElementById("nightbtn").click();
        if (s.cams !== undefined) document.getElementById("cambtn").click();
        if (s.cam) document.querySelector(`.strip .tile[data-key="${s.cam}"]`)?.click();
        if (s.resume) revis.resumeRoute();
        if (s.orbit) Object.assign(revis.orbit, s.orbit);
        if (s.mode) revis.setRenderMode(s.mode);
        if (s.split !== undefined) { revis.setAiMode("split", s.split); document.getElementById("splitbar").style.left = s.split * 100 + "%"; }
      }, s);
    }
    // advance simulation by exactly one frame, in small sub-steps for smooth camera
    await page.evaluate((dt) => new Promise((res) => {
      let k = 0;
      revis.playing = true;
      const sub = 4;
      revis.clock.getDelta = () => dt / sub;
      const step = () => { if (++k >= sub) { revis.playing = false; requestAnimationFrame(() => res()); } else requestAnimationFrame(step); };
      requestAnimationFrame(step);
    }), 1 / fps);
    await page.screenshot({ path: path.join(out, `f_${String(i).padStart(4, "0")}.png`) });
  }
}
await browser.close();
