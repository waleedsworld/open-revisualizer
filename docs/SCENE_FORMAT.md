# Scene format

A scene is a folder that a static web server can serve. The viewer only needs HTTP GET.

```
web/scenes/<name>/
├── background.ply        static Gaussians (world frame)
├── actors/<track_id>.ply moving rigid objects (object-local frame), optional
├── tracks.json           ego + actor poses over time
├── meta.json             name, source, licence, training stats
├── cams/<camera>/0000.jpg ... real camera frames (320 px wide) for the camera strip
└── previews/*.png        optional renders from the trainer
```

`web/scenes/index.json` lists the scenes and picks the default one:

```json
{ "default": "av2-miami-turn", "scenes": ["av2-miami-turn", "synthetic"] }
```

## World frame

Right-handed, **z up**, metres. The origin is the ego pose at the first exported frame. +x points forward and +y points left at that frame. The ego pose is the vehicle's own frame origin (for Argoverse 2, the rear axle centre). Set `road_z_below_ego` in `meta.json` (for example `-0.25`) so the viewer puts the car, path ribbon and pins on the road surface.

## `background.ply`

A standard 3D Gaussian Splatting binary PLY (the INRIA layout):

| property | meaning |
|---|---|
| `x y z` | centre (world frame) |
| `f_dc_0..2` | SH degree-0 colour |
| `f_rest_*` | higher SH bands (optional, degree ≤ 1 keeps files small) |
| `opacity` | pre-sigmoid logit |
| `scale_0..2` | log scale |
| `rot_0..3` | quaternion **w, x, y, z** |

Any format [Spark](https://sparkjs.dev) can read also works (`.spz`, `.splat`, `.ksplat`, `.sog`): set `"background": "background.spz"` in `meta.json`. `node tools/to_spz.mjs background.ply background.spz` compresses a ply about 3-10x (the demo scene goes from 84 MB to 26 MB). Keep each file under 95 MB if you host it on GitHub.

## `actors/<track_id>.ply`

The same layout, in the actor's own frame (box centre, +x forward). The viewer moves each actor with the per-frame pose from `tracks.json` and hides it on frames where the actor is missing.

## `tracks.json`

```json
{
  "fps": 10,
  "num_frames": 120,
  "cameras": ["ring_front_center", "ring_front_left", "..."],
  "ego_size": [4.8, 2.0, 1.5],
  "frames": [
    {
      "t": 0.0,
      "ego": { "p": [0, 0, 0], "q": [1, 0, 0, 0] },
      "speed_mps": 7.2,
      "actors": { "a1b2": { "p": [14.2, -0.3, 0.8], "q": [0.99, 0, 0, 0.05] } }
    }
  ]
}
```

Quaternions are `[w, x, y, z]`. Turn chevrons, the path ribbon, the next-turn banner and the remaining distance all come from the ego path. Nothing needs to be labelled by hand.

## `meta.json`

```json
{
  "name": "Miami right turn",
  "source": "Argoverse 2 Sensor, log ..., CC BY-NC-SA 4.0",
  "street": "optional street name for the turn banner",
  "destination": "optional label for the trip card",
  "speed_limit_mph": 25,
  "road_z_below_ego": -0.25,
  "num_gaussians_bg": 1200000,
  "train_iters": 30000,
  "psnr_eval": 27.1
}
```

To build a scene with no GPU, run `python3 tools/make_synthetic_scene.py web/scenes/synthetic`. It writes a procedural town in this format.
