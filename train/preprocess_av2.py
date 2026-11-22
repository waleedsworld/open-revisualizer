#!/usr/bin/env python3
"""Convert one Argoverse 2 Sensor log segment to drivestudio's processed layout.

Re-implements drivestudio's datasets/argoverse/argoverse_preprocess.py with pandas only,
so the av2-api fork (and its Rust toolchain) is not needed. Output layout is identical:

  <out>/<scene:03d>/images/{t:03d}_{cam}.jpg          original AV2 jpgs (copied, not re-encoded)
  <out>/<scene:03d>/lidar/{t:03d}.bin                 float32 [x y z intensity], ego frame at sweep time
  <out>/<scene:03d>/ego_pose/{t:03d}.txt              4x4 city_SE3_ego at lidar sweep time
  <out>/<scene:03d>/extrinsics/{cam}.txt              4x4 ego_SE3_cam
  <out>/<scene:03d>/intrinsics/{cam}.txt              fx fy cx cy 0 0 0 0 0
  <out>/<scene:03d>/dynamic_masks/{all,human,vehicle}/{t:03d}_{cam}.png   2D boxes of projected 3D cuboids
  <out>/<scene:03d>/instances/{instances_info,frame_instances}.json
  <out>/<scene:03d>/segment.json                      our extra: timestamps of every frame/camera

Frames are the lidar sweeps (10 Hz); for each sweep the nearest image of every ring camera is used.

usage: preprocess_av2.py --log_dir raw/val/<log_id> --out processed --scene 0 --t0 5.0 --t1 15.5
"""
import argparse, json, os, shutil
import numpy as np, pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R

CAMS = ["ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left",
        "ring_side_right", "ring_rear_left", "ring_rear_right"]
NONRIGID = ["BICYCLIST", "DOG", "MOTORCYCLIST", "PEDESTRIAN", "STROLLER", "WHEELCHAIR",
            "WHEELED_DEVICE", "WHEELED_RIDER"]
RIGID = ["ARTICULATED_BUS", "BOX_TRUCK", "BUS", "LARGE_VEHICLE", "MOTORCYCLE", "RAILED_VEHICLE",
         "REGULAR_VEHICLE", "SCHOOL_BUS", "TRUCK", "TRUCK_CAB", "VEHICULAR_TRAILER"]
DYNAMIC = NONRIGID + RIGID
# Rows cut off the bottom of these cameras because the ego car's hood/body is visible there
# (a static-in-camera object breaks the static scene fit). Cropping the bottom keeps fx fy cx cy valid;
# train/patch_drivestudio.py sets the matching original_size in drivestudio's dataset_meta.py.
CROP_BOTTOM = {"ring_front_center": 288, "ring_rear_left": 200, "ring_rear_right": 200}


def se3(q_wxyz, t):
    T = np.eye(4)
    T[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    T[:3, 3] = t
    return T


def pose_at(poses, ts):
    """city_SE3_ego at timestamp ts (exact row if present, else linear/slerp interpolation)."""
    tss = poses.timestamp_ns.values
    i = np.searchsorted(tss, ts)
    if i < len(tss) and tss[i] == ts:
        r = poses.iloc[i]
        return se3([r.qw, r.qx, r.qy, r.qz], [r.tx_m, r.ty_m, r.tz_m])
    i = int(np.clip(i, 1, len(tss) - 1))
    a, b = poses.iloc[i - 1], poses.iloc[i]
    w = (ts - tss[i - 1]) / (tss[i] - tss[i - 1])
    from scipy.spatial.transform import Slerp
    rots = R.from_quat([[a.qx, a.qy, a.qz, a.qw], [b.qx, b.qy, b.qz, b.qw]])
    T = np.eye(4)
    T[:3, :3] = Slerp([0, 1], rots)([w]).as_matrix()[0]
    T[:3, 3] = (1 - w) * np.array([a.tx_m, a.ty_m, a.tz_m]) + w * np.array([b.tx_m, b.ty_m, b.tz_m])
    return T


def cuboid_vertices(row):
    l, w, h = row.length_m / 2, row.width_m / 2, row.height_m / 2
    c = np.array([[sx * l, sy * w, sz * h] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)])
    T = se3([row.qw, row.qx, row.qy, row.qz], [row.tx_m, row.ty_m, row.tz_m])
    return c @ T[:3, :3].T + T[:3, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--t0", type=float, default=0.0, help="segment start, seconds after first pose")
    ap.add_argument("--t1", type=float, default=1e9)
    a = ap.parse_args()
    L = a.log_dir.rstrip("/")
    out = os.path.join(a.out, f"{a.scene:03d}")
    for d in ["images", "sky_masks", "lidar", "ego_pose", "extrinsics", "intrinsics", "instances",
              "dynamic_masks/all", "dynamic_masks/human", "dynamic_masks/vehicle"]:
        os.makedirs(os.path.join(out, d), exist_ok=True)

    poses = pd.read_feather(f"{L}/city_SE3_egovehicle.feather").sort_values("timestamp_ns").reset_index(drop=True)
    intr = pd.read_feather(f"{L}/calibration/intrinsics.feather").set_index("sensor_name")
    extr = pd.read_feather(f"{L}/calibration/egovehicle_SE3_sensor.feather").set_index("sensor_name")
    ann = pd.read_feather(f"{L}/annotations.feather")
    tpose0 = poses.timestamp_ns.values[0]

    sweeps = sorted(int(f[:-8]) for f in os.listdir(f"{L}/sensors/lidar") if f.endswith(".feather"))
    cam_ts = {c: np.array(sorted(int(f[:-4]) for f in os.listdir(f"{L}/sensors/cameras/{c}"))) for c in CAMS}

    # calibration
    K, E = {}, {}
    for ci, c in enumerate(CAMS):
        r = intr.loc[c]
        K[c] = (r.fx_px, r.fy_px, r.cx_px, r.cy_px, int(r.width_px), int(r.height_px) - CROP_BOTTOM.get(c, 0))
        np.savetxt(f"{out}/intrinsics/{ci}.txt", [r.fx_px, r.fy_px, r.cx_px, r.cy_px, 0, 0, 0, 0, 0])
        e = extr.loc[c]
        E[c] = se3([e.qw, e.qx, e.qy, e.qz], [e.tx_m, e.ty_m, e.tz_m])
        np.savetxt(f"{out}/extrinsics/{ci}.txt", E[c])

    frames = []
    for s in sweeps:
        t = (s - tpose0) / 1e9
        if t < a.t0 or t >= a.t1:
            continue
        match = {}
        for c in CAMS:
            j = np.argmin(np.abs(cam_ts[c] - s))
            if abs(cam_ts[c][j] - s) > 50e6:  # need every camera within 50 ms
                break
            match[c] = int(cam_ts[c][j])
        if len(match) == len(CAMS):
            frames.append((s, match))
    print(f"{len(frames)} frames in segment [{a.t0}, {a.t1}) s")

    instances, frame_instances, seg = {}, {}, []
    for fi, (s, match) in enumerate(frames):
        v2w = pose_at(poses, s)
        np.savetxt(f"{out}/ego_pose/{fi:03d}.txt", v2w)
        pc = pd.read_feather(f"{L}/sensors/lidar/{s}.feather")
        np.column_stack([pc.x.values, pc.y.values, pc.z.values, pc.intensity.values]).astype(np.float32) \
            .tofile(f"{out}/lidar/{fi:03d}.bin")
        A = ann[ann.timestamp_ns == s]
        verts = {k: [] for k in ("all", "human", "vehicle")}
        frame_instances[fi] = []
        for _, row in A.iterrows():
            if row.category not in DYNAMIC:
                continue
            V = cuboid_vertices(row)
            verts["all"].append(V)
            verts["human" if row.category in NONRIGID else "vehicle"].append(V)
            tid = row.track_uuid
            if tid not in instances:
                instances[tid] = dict(id=tid, class_name=row.category,
                                      frame_annotations=dict(frame_idx=[], obj_to_world=[], box_size=[]))
            o2v = se3([row.qw, row.qx, row.qy, row.qz], [row.tx_m, row.ty_m, row.tz_m])
            fa = instances[tid]["frame_annotations"]
            fa["frame_idx"].append(fi)
            fa["obj_to_world"].append((v2w @ o2v).tolist())
            fa["box_size"].append([float(row.length_m), float(row.width_m), float(row.height_m)])
            frame_instances[fi].append(tid)
        for ci, c in enumerate(CAMS):
            fx, fy, cx, cy, W, H = K[c]
            src = f"{L}/sensors/cameras/{c}/{match[c]}.jpg"
            if c in CROP_BOTTOM:
                im = Image.open(src)
                im.crop((0, 0, W, H)).save(f"{out}/images/{fi:03d}_{ci}.jpg", quality=95)
            else:
                shutil.copyfile(src, f"{out}/images/{fi:03d}_{ci}.jpg")
            v2c = np.linalg.inv(E[c])
            for kind, Vs in verts.items():
                m = np.zeros((H, W), np.uint8)
                for V in Vs:
                    pc_ = V @ v2c[:3, :3].T + v2c[:3, 3]
                    pc_ = pc_[pc_[:, 2] > 0.1]
                    if len(pc_) == 0:
                        continue
                    u = fx * pc_[:, 0] / pc_[:, 2] + cx
                    v = fy * pc_[:, 1] / pc_[:, 2] + cy
                    u0, u1 = int(np.clip(u.min(), 0, W)), int(np.clip(u.max(), 0, W))
                    v0, v1 = int(np.clip(v.min(), 0, H)), int(np.clip(v.max(), 0, H))
                    if u1 > u0 and v1 > v0:
                        m[v0:v1, u0:u1] = 255
                Image.fromarray(m, "L").save(f"{out}/dynamic_masks/{kind}/{fi:03d}_{ci}.png")
        seg.append(dict(frame=fi, lidar_ts=s, t=(s - tpose0) / 1e9, cams=match))
        print(f"frame {fi} t={(s - tpose0) / 1e9:.2f}s objs={len(frame_instances[fi])}", flush=True)

    id_map = {k: i for i, k in enumerate(instances)}
    json.dump({id_map[k]: v for k, v in instances.items()}, open(f"{out}/instances/instances_info.json", "w"))
    json.dump({k: [id_map[i] for i in v] for k, v in frame_instances.items()},
              open(f"{out}/instances/frame_instances.json", "w"))
    json.dump(dict(log_dir=os.path.abspath(L), log_id=os.path.basename(L), cams=CAMS, t0=a.t0, t1=a.t1,
                   crop_bottom=CROP_BOTTOM, frames=seg), open(f"{out}/segment.json", "w"), indent=1)


if __name__ == "__main__":
    main()
