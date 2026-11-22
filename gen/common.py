"""Shared data loading and camera maths for the generative Reality View (gen/).

Everything is in the viewer's world frame: z up, origin = ego pose at the first exported frame
(= drivestudio world = inverse(ego_pose/000.txt) applied to AV2 city coordinates), +x forward.

Inputs (made by train/preprocess_av2.py on the box):
  <proc>/images/{t:03d}_{ci}.jpg   ring cameras (bottom rows of front_center / rear_* already cropped = ego mask)
  <proc>/lidar/{t:03d}.bin         float32 x y z intensity, ego frame at sweep time
  <proc>/ego_pose/{t:03d}.txt      city_SE3_ego at sweep time
  <proc>/extrinsics/{ci}.txt       ego_SE3_cam (OpenCV camera: x right, y down, z forward)
  <proc>/intrinsics/{ci}.txt       fx fy cx cy
  <proc>/segment.json              per-frame lidar / camera timestamps
  <raw>/city_SE3_egovehicle.feather, <raw>/annotations.feather
"""
import json, os, math
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as R, Slerp

CAMS = ["ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left",
        "ring_side_right", "ring_rear_left", "ring_rear_right"]
DYNAMIC = ["BICYCLIST", "DOG", "MOTORCYCLIST", "PEDESTRIAN", "STROLLER", "WHEELCHAIR", "WHEELED_DEVICE",
           "WHEELED_RIDER", "ARTICULATED_BUS", "BOX_TRUCK", "BUS", "LARGE_VEHICLE", "MOTORCYCLE",
           "RAILED_VEHICLE", "REGULAR_VEHICLE", "SCHOOL_BUS", "TRUCK", "TRUCK_CAB", "VEHICULAR_TRAILER"]

# virtual chase camera = web/revisualizer.js VIEWS.reality
CHASE = dict(back=10.5, up=5.6, ahead=14.0, look_up=0.6, vfov=52.0, W=960, H=540)


def se3(q_wxyz, t):
    T = np.eye(4)
    T[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    T[:3, 3] = t
    return T


class Scene:
    def __init__(self, proc, raw, tracks=None):
        self.proc, self.raw = proc, raw
        self.seg = json.load(open(f"{proc}/segment.json"))
        self.n = len(self.seg["frames"])
        self.poses = pd.read_feather(f"{raw}/city_SE3_egovehicle.feather").sort_values("timestamp_ns").reset_index(drop=True)
        self.W_city = np.linalg.inv(np.loadtxt(f"{proc}/ego_pose/000.txt"))
        self.E = [np.loadtxt(f"{proc}/extrinsics/{ci}.txt") for ci in range(7)]
        self.K = []
        for ci in range(7):
            fx, fy, cx, cy = np.loadtxt(f"{proc}/intrinsics/{ci}.txt")[:4]
            self.K.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]))
        ann = pd.read_feather(f"{raw}/annotations.feather")
        self.ann = ann[ann.category.isin(DYNAMIC)]
        self.tracks = json.load(open(tracks)) if tracks else None

    def pose_city(self, ts):
        p = self.poses
        tss = p.timestamp_ns.values
        i = int(np.clip(np.searchsorted(tss, ts), 1, len(tss) - 1))
        a, b = p.iloc[i - 1], p.iloc[i]
        w = float(np.clip((ts - tss[i - 1]) / (tss[i] - tss[i - 1]), 0, 1))
        rots = R.from_quat([[a.qx, a.qy, a.qz, a.qw], [b.qx, b.qy, b.qz, b.qw]])
        T = np.eye(4)
        T[:3, :3] = Slerp([0, 1], rots)([w]).as_matrix()[0]
        T[:3, 3] = (1 - w) * np.array([a.tx_m, a.ty_m, a.tz_m]) + w * np.array([b.tx_m, b.ty_m, b.tz_m])
        return T

    def ego_world(self, fi):
        """world_SE3_ego at the lidar sweep of frame fi."""
        return self.W_city @ np.loadtxt(f"{self.proc}/ego_pose/{fi:03d}.txt")

    def cam_world(self, fi, ci):
        """world_SE3_cam at the exposure time of camera ci in frame fi (ego pose interpolated to that time)."""
        ts = self.seg["frames"][fi]["cams"][CAMS[ci]]
        return self.W_city @ self.pose_city(ts) @ self.E[ci]

    def lidar(self, fi):
        return np.fromfile(f"{self.proc}/lidar/{fi:03d}.bin", np.float32).reshape(-1, 4)[:, :3]

    def boxes(self, fi):
        """dynamic-object cuboids at this sweep, ego frame: list of (ego_SE3_obj, half-size lwh)."""
        ts = self.seg["frames"][fi]["lidar_ts"]
        A = self.ann[self.ann.timestamp_ns == ts]
        return [(se3([r.qw, r.qx, r.qy, r.qz], [r.tx_m, r.ty_m, r.tz_m]),
                 np.array([r.length_m, r.width_m, r.height_m]) / 2) for r in A.itertuples()]

    def image_path(self, fi, ci):
        return f"{self.proc}/images/{fi:03d}_{ci}.jpg"

    def ego_pq(self, fi):
        """ego position + yaw as the viewer sees it (tracks.json if given, else from the poses)."""
        if self.tracks:
            e = self.tracks["frames"][fi]["ego"]
            w, x, y, z = e["q"]
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            return np.array(e["p"], float), yaw
        T = self.ego_world(fi)
        return T[:3, 3].copy(), math.atan2(T[1, 0], T[0, 0])


def chase_camera(p, yaw, c=CHASE):
    """world_SE3_cam (OpenCV axes) and K for the virtual chase camera."""
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    pos = p - c["back"] * fwd + np.array([0, 0, c["up"]])
    look = p + c["ahead"] * fwd + np.array([0, 0, c["look_up"]])
    f = look - pos; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, d, f, pos
    fy = (c["H"] / 2) / math.tan(math.radians(c["vfov"]) / 2)
    K = np.array([[fy, 0, c["W"] / 2], [0, fy, c["H"] / 2], [0, 0, 1.0]])
    return T, K
