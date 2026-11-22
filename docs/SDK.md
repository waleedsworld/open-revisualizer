# Open Revisualizer SDK

The viewer is one ES module, `web/revisualizer.js`, plus an optional car-display HUD, `web/hud.js`. It runs on three.js and [Spark](https://sparkjs.dev). There is no build step.

- [Install](#install)
- [Quick start](#quick-start)
- [createRevisualizer(options)](#createrevisualizeroptions)
- [Methods](#methods)
- [Properties](#properties)
- [Events](#events)
- [URL parameters](#url-parameters-indexhtml)
- [Keyboard and pointer](#keyboard-and-pointer)
- [Recipes](#recipes)
- [Recording video and GIFs](#recording-video-and-gifs)

## Install

Copy the `web/` folder, or load it from the CDN. The module imports `three` and `@sparkjsdev/spark`, so map them with an import map:

```html
<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/",
  "@sparkjsdev/spark": "https://cdn.jsdelivr.net/npm/@sparkjsdev/spark@2.3.1/dist/spark.module.js"
} }
</script>
```

The repo ships these files in `web/vendor/`, so the demo also works offline.

## Quick start

```html
<div id="view" style="position:fixed;inset:0"></div>
<script type="module">
  import { createRevisualizer } from "./revisualizer.js";

  const rv = await createRevisualizer({
    container: document.getElementById("view"),
    scene: "scenes/av2-miami-turn",   // a folder in the scene format
  });

  rv.on("frame", (s) => console.log(s.speedMps, s.nextTurn));
  rv.goTo(40, 12);                     // drive to a world point
</script>
```

To get the full car-display UI as well, add the HUD:

```js
import { bindHud } from "./hud.js";
bindHud(rv, document.getElementById("screen"));   // needs the markup from index.html
```

## `createRevisualizer(options)`

Returns `Promise<Revisualizer>`, which resolves once the splats have loaded.

| option | type | default | |
|---|---|---|---|
| `container` | `HTMLElement` | required | The canvas fills this element. |
| `scene` | `string \| null` | `null` | Scene folder URL. `null` reads `scenes/index.json` and uses its `default`. |
| `autoplay` | `boolean` | `true` | Start playing after load. |
| `speed` | `number` | `1` | Playback speed multiplier. |
| `view` | `"reality" \| "close" \| "top" \| "hood"` | `"reality"` | Starting camera preset. |
| `night` | `boolean` | `false` | Night grade: darker splats, night sky, headlights on. |
| `tapToGo` | `boolean` | `true` | A click or tap on the scene drives the car to that point. |
| `skySplat` | `boolean` | `false` | Load the scene's learned `sky.ply` dome instead of the gradient sky. |
| `carModel` | `string \| false` | `assets/car.glb` | glTF car URL. `false` uses the built-in low-poly car. |
| `onProgress` | `(p: number) => void` | | Load progress from 0 to 1. |

## Methods

| method | |
|---|---|
| `play()` / `pause()` / `toggle()` | Replay control. |
| `seek(t)` | Jump to `t` seconds. This also cancels tap-to-go. |
| `setSpeed(x)` | Playback speed. |
| `setView(name)` / `cycleView()` | Camera presets from `VIEWS`. |
| `setNight(bool)` | Night mode. |
| `goTo(x, y)` | Leave the recorded route and drive to world `(x, y)`. Returns `false` if the point is under 2 m away. |
| `tapAt(clientX, clientY)` | Same as a finger tap. Raycasts the splats (falls back to the ground plane), then calls `goTo`. |
| `resumeRoute()` | Go back to the recorded drive. |
| `pointAtDist(d)` | World point on the recorded path at arc length `d` metres. |
| `trackAt(t)` | Interpolated recorded state at time `t`. |
| `screenshot()` | PNG data URL of the current frame. |
| `on(type, fn)` | Subscribe to an event. Returns an unsubscribe function. |
| `dispose()` | Stop rendering and free the GPU. |

## Properties

| property | |
|---|---|
| `state` | `{ t, frame, mode, ego: {p, q, yaw}, speedMps, dist, nextTurn: {dir, distM} \| null, remainingM, duration }` |
| `t`, `duration`, `playing`, `speed` | Playback. |
| `mode` | `"replay"`, `"goto"` (driving to a pin) or `"parked"` (arrived). |
| `turns` | Detected turns: `[{ start, end, dir: "left" \| "right" }]` in metres along the path. |
| `orbit` | `{ yaw, pitch, zoom }` offsets on top of the active view. Drag and wheel change it. |
| `camera`, `scene`, `renderer`, `spark` | three.js objects, for adding your own meshes. |
| `background`, `actors` | The `SplatMesh` objects (`actors` is a `Map` keyed by track id). |
| `tracks`, `meta` | The parsed scene files. |

`VIEWS` is exported and can be changed:

```js
import { VIEWS } from "./revisualizer.js";
VIEWS.drone = { back: 25, up: 18, ahead: 20, fov: 50, lookUp: 0 };
rv.setView("drone");
```

## Events

| event | detail |
|---|---|
| `loaded` | `{ meta, duration, cameras }` |
| `frame` | the `state` object, every rendered frame |
| `play`, `pause` | none |
| `view` | `{ view }` |
| `night` | `{ night }` |
| `goto` | `{ x, y, distance }` |
| `arrived` | `{ x, y }` |
| `resume` | none |

## URL parameters (`index.html`)

| param | example | |
|---|---|---|
| `scene` | `?scene=scenes/synthetic` | Scene folder. |
| `t` | `&t=4.5` | Start time. |
| `view` | `&view=top` | Camera preset. |
| `night` | `&night=1` | Night mode. |
| `hud` | `&hud=0` | Hide the HUD (clean frames for video). |
| `autoplay` | `&autoplay=0` | Start paused. |
| `speed` | `&speed=0.5` | Playback speed. |

## Keyboard and pointer

| input | action |
|---|---|
| click or tap | Drive to that point (tap-to-go). |
| drag | Orbit the chase camera. |
| wheel | Zoom. |
| double-click | Reset the camera. |
| `Space` | Play or pause. |
| `←` / `→` | Seek ±2 s. |
| `1` `2` `3` `4` | Reality / close / top / hood view. |
| `V` | Cycle views. |
| `C` | Camera strip (real footage, synced). |
| `N` | Night mode. |
| `H` | Hide the HUD. |
| `Esc` | Resume the route and close the camera view. |

## Recipes

**Custom HUD.** Skip `hud.js` and read `state` yourself:

```js
rv.on("frame", ({ speedMps, nextTurn }) => {
  speedEl.textContent = Math.round(speedMps * 2.237);
  turnEl.textContent = nextTurn ? `${nextTurn.dir} in ${Math.round(nextTurn.distM)} m` : "";
});
```

**Scripted camera tour:**

```js
rv.setView("top");
setTimeout(() => rv.setView("reality"), 3000);
setTimeout(() => { rv.orbit.yaw = Math.PI; }, 6000);   // look back at the car
```

**Add your own 3D objects.** The world is z-up, in metres:

```js
import * as THREE from "three";
const cone = new THREE.Mesh(new THREE.ConeGeometry(0.3, 0.8), new THREE.MeshStandardMaterial({ color: "orange" }));
const p = rv.pointAtDist(60);
cone.position.set(p.x, p.y + 2, p.z + 0.4);
cone.rotation.x = Math.PI / 2;
rv.scene.add(cone);
```

## Recording video and GIFs

`tools/capture.mjs` drives a real GPU browser and steps the simulation one frame at a time, so every frame is clean and deterministic:

```bash
cd tools && npm i && cd ..
python3 -m http.server 8765 -d web &
node tools/capture.mjs frames "http://127.0.0.1:8765/?autoplay=0" out/frames \
     --w 1280 --h 800 --fps 30 --seconds 10 --script tools/demo_steps.json
ffmpeg -framerate 30 -i out/frames/f_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 16 out/demo.mp4
tools/make_gif.sh out/frames docs/media/demo.gif
```

A steps file schedules actions by time, for example `[{"at": 2, "tap": [0.6, 0.55]}, {"at": 5, "view": "top"}, {"at": 7, "night": true}, {"at": 8, "cams": true}]`. `tap` uses fractions of the canvas width and height.
