"""Build a small synthetic driving scene in the open-revisualizer scene format.

It lets you run the viewer with no GPU and no dataset, and it is the reference
for the scene format (see docs/SCENE_FORMAT.md).

    python3 tools/make_synthetic_scene.py web/scenes/synthetic
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

C0 = 0.28209479177387814
rng = np.random.default_rng(7)


def gauss(pos, rgb, scale, opacity=0.95):
    n = len(pos)
    scale = np.broadcast_to(np.asarray(scale, np.float32), (n, 3))
    rgb = np.broadcast_to(np.asarray(rgb, np.float32), (n, 3))
    return dict(
        p=np.asarray(pos, np.float32),
        c=(rgb - 0.5) / C0,
        s=np.log(np.maximum(scale, 1e-4)),
        o=np.full(n, math.log(opacity / (1 - opacity)), np.float32),
    )


def cat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def write_ply(path, g):
    n = len(g["p"])
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    head = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    head += "".join("property float %s\n" % p for p in props) + "end_header\n"
    rot = np.zeros((n, 4), np.float32)
    rot[:, 0] = 1
    data = np.concatenate([g["p"], g["c"], g["o"][:, None], g["s"], rot], 1).astype("<f4")
    with open(path, "wb") as f:
        f.write(head.encode())
        f.write(data.tobytes())


# ---- trajectory: straight, 90 deg left turn, straight -------------------------
def centerline(n=2000):
    pts, yaw = [], []
    R = 14.0
    for s in np.linspace(0, 1, n):
        d = s * (45 + math.pi / 2 * R + 45)
        if d < 45:
            pts.append((d, 0)); yaw.append(0)
        elif d < 45 + math.pi / 2 * R:
            a = (d - 45) / R
            pts.append((45 + R * math.sin(a), R - R * math.cos(a))); yaw.append(a)
        else:
            e = d - 45 - math.pi / 2 * R
            pts.append((45 + R, R + e)); yaw.append(math.pi / 2)
    return np.array(pts), np.array(yaw)


line, lyaw = centerline()
# the road runs back behind the start and past the end so the camera never sees an edge
back = np.stack([np.linspace(-40, 0, 200), np.zeros(200)], 1)
fwd = np.stack([np.full(200, 45 + 14.0), np.linspace(line[-1, 1], line[-1, 1] + 40, 200)], 1)
road_line = np.concatenate([back, line, fwd])
road_yaw = np.concatenate([np.zeros(200), lyaw, np.full(200, math.pi / 2)])


def offset(lat):
    nrm = np.stack([-np.sin(road_yaw), np.cos(road_yaw)], 1)
    return road_line + nrm * lat


parts = []
# ground grid
xs, ys = np.meshgrid(np.arange(-60, 120, 0.35), np.arange(-60, 120, 0.35))
P = np.stack([xs.ravel(), ys.ravel()], 1)
from scipy.spatial import cKDTree  # noqa: E402
tree = cKDTree(road_line)
dist, idx = tree.query(P)
lat = dist
asphalt = lat < 7.5
walk = (lat >= 7.5) & (lat < 10.5)
grass = lat >= 10.5
noise = rng.normal(0, 0.025, (len(P), 1))
for mask, col, sc in [(asphalt, (0.23, 0.235, 0.25), 0.22), (walk, (0.55, 0.54, 0.52), 0.22),
                      (grass, (0.30, 0.42, 0.22), 0.25)]:
    q = P[mask]
    parts.append(gauss(np.c_[q, np.zeros(len(q))], np.clip(np.array(col) + noise[mask], 0, 1), (sc, sc, 0.02)))
# lane markings: double yellow center, white edges (dashed)
for lat0, col, dash in [(-0.15, (0.95, 0.78, 0.15), False), (0.15, (0.95, 0.78, 0.15), False),
                         (-7.2, (0.95, 0.95, 0.95), False), (7.2, (0.95, 0.95, 0.95), False),
                         (-3.6, (0.92, 0.92, 0.92), True), (3.6, (0.92, 0.92, 0.92), True)]:
    q = offset(lat0)
    if dash:
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        q = q[(d % 9) < 4]
    parts.append(gauss(np.c_[q, np.full(len(q), 0.01)], col, (0.09, 0.09, 0.01)))

# buildings along both sides
def box_surface(cx, cy, w, d, h, yaw, col):
    pts, cols = [], []
    step = 0.4
    for fx in np.arange(-w / 2, w / 2, step):
        for z in np.arange(0, h, step):
            for fy in (-d / 2, d / 2):
                pts.append((fx, fy, z))
    for fy in np.arange(-d / 2, d / 2, step):
        for z in np.arange(0, h, step):
            for fx in (-w / 2, w / 2):
                pts.append((fx, fy, z))
    for fx in np.arange(-w / 2, w / 2, step):
        for fy in np.arange(-d / 2, d / 2, step):
            pts.append((fx, fy, h))
    pts = np.array(pts)
    win = ((np.floor(pts[:, 2] / 1.6) % 2 == 1) & (np.floor((pts[:, 0] + pts[:, 1]) / 1.2) % 2 == 0)
           & (pts[:, 2] < h - 0.5))
    c = np.tile(col, (len(pts), 1)) + rng.normal(0, 0.02, (len(pts), 3))
    c[win] = (0.85, 0.75, 0.45) if rng.random() < 0.5 else (0.25, 0.35, 0.45)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    wx = cx + pts[:, 0] * cy_ - pts[:, 1] * sy_
    wy = cy + pts[:, 0] * sy_ + pts[:, 1] * cy_
    return gauss(np.c_[wx, wy, pts[:, 2]], np.clip(c, 0, 1), (0.28, 0.28, 0.28))


palette = [(0.62, 0.45, 0.36), (0.72, 0.70, 0.66), (0.45, 0.47, 0.52), (0.80, 0.74, 0.60), (0.55, 0.30, 0.25)]
for i in range(0, len(road_line), 60):
    for side in (-1, 1):
        x, y = road_line[i]
        yw = road_yaw[i]
        nx, ny = -math.sin(yw) * side, math.cos(yw) * side
        dd = 16 + rng.random() * 3
        cx, cy = x + nx * dd, y + ny * dd
        if np.min(np.linalg.norm(road_line - (cx, cy), axis=1)) < 12:
            continue
        parts.append(box_surface(cx, cy, 8 + rng.random() * 4, 7, 6 + rng.random() * 14, yw,
                                 palette[rng.integers(len(palette))]))
# trees on the sidewalk edge
for i in range(30, len(road_line), 45):
    for side in (-1, 1):
        x, y = road_line[i]
        yw = road_yaw[i]
        tx, ty = x - math.sin(yw) * side * 9.5, y + math.cos(yw) * side * 9.5
        trunk = np.c_[np.full(20, tx), np.full(20, ty), np.linspace(0, 2.5, 20)]
        parts.append(gauss(trunk, (0.3, 0.22, 0.15), (0.12, 0.12, 0.15)))
        crown = rng.normal(0, 1, (400, 3)) * (1.4, 1.4, 1.2) + (tx, ty, 4)
        parts.append(gauss(crown, np.clip((0.2, 0.4, 0.15) + rng.normal(0, 0.05, (400, 3)), 0, 1),
                           (0.35, 0.35, 0.35), 0.8))

out = Path(sys.argv[1] if len(sys.argv) > 1 else "web/scenes/synthetic")
(out / "actors").mkdir(parents=True, exist_ok=True)
bg = cat(parts)
write_ply(out / "background.ply", bg)


# ---- actors: car-shaped blobs in their local frame ----------------------------
def car(col):
    pts = []
    for x in np.arange(-2.3, 2.3, 0.15):
        for y in np.arange(-0.9, 0.9, 0.15):
            pts.append((x, y, 0.35)); pts.append((x, y, 0.95 if abs(x) > 1.0 else 1.45))
    for x in np.arange(-2.3, 2.3, 0.15):
        for z in np.arange(0.35, 1.0, 0.15):
            pts.append((x, -0.9, z)); pts.append((x, 0.9, z))
    pts = np.array(pts)
    c = np.tile(col, (len(pts), 1)).astype(np.float32)
    c[(pts[:, 2] > 1.0) & (np.abs(pts[:, 0]) < 0.9)] = (0.1, 0.12, 0.15)  # windows
    c[(pts[:, 0] < -2.1) & (pts[:, 2] < 0.9)] = (0.9, 0.1, 0.1)  # tail lights
    return gauss(pts, c, (0.12, 0.12, 0.12))


actor_cols = {"car_a": (0.85, 0.85, 0.88), "car_b": (0.6, 0.1, 0.1), "car_c": (0.15, 0.25, 0.5)}
for k, col in actor_cols.items():
    write_ply(out / "actors" / f"{k}.ply", car(col))


def quat_z(a):
    return [math.cos(a / 2), 0, 0, math.sin(a / 2)]


# ---- frames: ego drives the centreline in the right lane ----------------------
fps, N = 10, 160
d_line = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
lane = -1.8  # right lane (y is left)


def pose_at(dist, lat=lane):
    dist = np.clip(dist, 0, d_line[-1])
    i = np.searchsorted(d_line, dist)
    i = min(max(i, 1), len(line) - 1)
    t = (dist - d_line[i - 1]) / max(d_line[i] - d_line[i - 1], 1e-6)
    p = line[i - 1] * (1 - t) + line[i] * t
    yw = lyaw[i - 1] * (1 - t) + lyaw[i] * t
    return [float(p[0] - math.sin(yw) * lat), float(p[1] + math.cos(yw) * lat), 0.0], float(yw)


frames = []
dist = 0.0
for f in range(N):
    t = f / fps
    v = 9.0 - 3.5 * math.exp(-((dist - 55) / 14) ** 2)  # slow down for the turn
    p, yw = pose_at(dist)
    acts = {}
    pa, ya = pose_at(dist + 14 + 0.5 * t)  # lead car in our lane
    acts["car_a"] = {"p": pa, "q": quat_z(ya)}
    pb, yb = pose_at(130 - 11 * t, lat=1.8)  # oncoming
    if 0 < 130 - 11 * t < d_line[-1]:
        acts["car_b"] = {"p": pb, "q": quat_z(yb + math.pi)}
    pc, yc = pose_at(150 - 6 * t, lat=6.0)  # parked-ish, slow
    acts["car_c"] = {"p": pc, "q": quat_z(yc + math.pi)}
    frames.append({"t": round(t, 3), "ego": {"p": p, "q": quat_z(yw)}, "speed_mps": round(v, 2), "actors": acts})
    dist += v / fps

cams = ["ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left",
        "ring_side_right", "ring_rear_left", "ring_rear_right"]
json.dump({"fps": fps, "num_frames": N, "frames": frames, "cameras": cams, "ego_size": [4.8, 2.0, 1.5]},
          open(out / "tracks.json", "w"))
for cam in cams:
    (out / "cams" / cam).mkdir(parents=True, exist_ok=True)
    for f in range(N):
        im = Image.new("RGB", (320, 240), (40 + f % 50, 50, 70))
        d = ImageDraw.Draw(im)
        d.text((10, 10), f"{cam}\nframe {f}", fill=(220, 220, 220))
        im.save(out / "cams" / cam / f"{f:04d}.jpg", quality=60)
json.dump({"name": "Synthetic test town", "source": "procedural (tools/make_synthetic_scene.py)",
           "segment": [0, N / fps], "num_gaussians_bg": int(len(bg["p"])), "num_actors": len(actor_cols),
           "notes": "Procedural scene for testing the viewer without a GPU."},
          open(out / "meta.json", "w"), indent=2)
print("wrote", out, len(bg["p"]), "gaussians")
