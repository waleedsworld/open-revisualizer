#!/usr/bin/env python3
"""Export a trained drivestudio StreetGS model to the open-revisualizer web format and render previews.

Run inside the drivestudio checkout (PYTHONPATH=.):
  python ../train/export_scene.py --ckpt output/av2/revis/checkpoint_final.pth \
      --raw_log data/argoverse/raw/val/<log_id> --out ../web/scenes/<name> --name <name>

Writes: background.ply, sky.ply (extra, optional for the viewer), actors/<id>.ply, tracks.json, cams/, meta.json,
previews/. World frame = drivestudio's world = AV2 ego frame (rear axle) at the first frame: x fwd, y left, z up.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.getcwd())
from datasets.driving_dataset import DrivingDataset  # noqa: E402
from datasets.base.pixel_source import get_rays  # noqa: E402
from utils.misc import import_str  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--raw_log", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--max_bg", type=int, default=1_500_000)
ap.add_argument("--keep_sh_rest", action="store_true", help="write the 9 SH degree-1 coefficients (bigger files)")
ap.add_argument("--max_bytes", type=float, default=94e6, help="per-file size budget (GitHub limit is 100 MB)")
ap.add_argument("--max_dist", type=float, default=150.0)
ap.add_argument("--train_minutes", type=float, default=-1)
ap.add_argument("--gpu", default="")
ap.add_argument("--skip_previews", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda")
log_dir = os.path.dirname(a.ckpt)
cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
dataset = DrivingDataset(data_cfg=cfg.data)
trainer = import_str(cfg.trainer.type)(
    **cfg.trainer, num_timesteps=dataset.num_img_timesteps, model_config=cfg.model,
    num_train_images=len(dataset.train_image_set), num_full_images=len(dataset.full_image_set),
    test_set_indices=dataset.test_timesteps, scene_aabb=dataset.get_aabb().reshape(2, 3), device=dev)
trainer.resume_from_checkpoint(ckpt_path=a.ckpt, load_only_model=True)
trainer.set_eval()
data_path = dataset.data_path
seg = json.load(open(os.path.join(data_path, "segment.json")))
N = dataset.num_img_timesteps

# ---------------------------------------------------------------- poses
E = [np.loadtxt(os.path.join(data_path, "ego_pose", f"{t:03d}.txt")) for t in range(N)]
E0inv = np.linalg.inv(E[0])
ego = np.stack([E0inv @ e for e in E])  # (N,4,4) ego->world


def rot2quat(Rm):  # -> w,x,y,z
    from scipy.spatial.transform import Rotation as R
    q = R.from_matrix(Rm).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


traj = ego[:, :3, 3]

# ---------------------------------------------------------------- ply writer
def write_ply(path, xyz, dc, rest, op_logit, log_scale, quat):
    n = len(xyz)
    nrest = rest.shape[1] if rest is not None else 0
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(nrest)] + \
        ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    cols = [xyz, dc] + ([rest] if nrest else []) + [op_logit[:, None], log_scale, quat]
    arr = np.concatenate(cols, axis=1).astype("<f4")
    assert arr.shape[1] == len(names)
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n).encode())
        for nm in names:
            f.write(f"property float {nm}\n".encode())
        f.write(b"end_header\n")
        f.write(arr.tobytes())
    return os.path.getsize(path)


def gparams(m, mask=None):
    xyz = m._means.detach()
    dc = m._features_dc.detach()
    rest = m._features_rest.detach()  # (N, K, 3)
    op = m._opacities.detach().squeeze(-1)
    sc = m._scales.detach()
    q = m._quats.detach()
    if mask is not None:
        xyz, dc, rest, op, sc, q = xyz[mask], dc[mask], rest[mask], op[mask], sc[mask], q[mask]
    # INRIA order: f_rest is channel-major (all R coeffs, then G, then B)
    rest = rest[:, :3, :].transpose(1, 2).reshape(len(xyz), -1) if (rest.shape[1] >= 3 and a.keep_sh_rest) else None
    return [t.cpu().numpy() if t is not None else None for t in (xyz, dc, rest, op, sc, q)]


# ---------------------------------------------------------------- background
bg = trainer.models["Background"]
xyz, dc, rest, op, sc, q = gparams(bg)
n0 = len(xyz)
tt = torch.from_numpy(traj).float().to(dev)
d = []
for c in torch.split(torch.from_numpy(xyz).float().to(dev), 200000):
    d.append(torch.cdist(c[:, :2], tt[:, :2]).min(1).values.cpu())
dist = torch.cat(d).numpy()
opa = 1 / (1 + np.exp(-op))
maxs = np.exp(sc.max(1))
keep = (opa > 0.03) & (dist < a.max_dist) & (maxs < 15.0) & np.isfinite(xyz).all(1)
print(f"background: {n0} -> {keep.sum()} after opacity/size/distance prune")
bytes_per = 4 * (3 + 3 + (9 if rest is not None else 0) + 1 + 3 + 4)
a.max_bg = min(a.max_bg, int((a.max_bytes - 4096) // bytes_per))
if keep.sum() > a.max_bg:
    imp = opa * np.exp(sc.sum(1)) ** (1 / 3)
    thr = np.sort(imp[keep])[-a.max_bg]
    keep &= imp >= thr
    print(f"background: importance prune -> {keep.sum()}")
size = write_ply(os.path.join(a.out, "background.ply"), xyz[keep], dc[keep], rest[keep] if rest is not None else None,
                 op[keep], sc[keep], q[keep])
n_bg = int(keep.sum())
print("background.ply", n_bg, size / 1e6, "MB")
assert size < 95e6

# ---------------------------------------------------------------- sky dome (extra file)
sky = trainer.models["Sky"]
M = 30000
i = np.arange(M) + 0.5
phi = np.arccos(1 - 2 * i / M); th = math.pi * (1 + 5 ** 0.5) * i
dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
dirs = dirs[dirs[:, 2] > -0.15]
with torch.no_grad():
    rgb = sky({"viewdirs": torch.from_numpy(dirs).float().to(dev)[None, None]}).reshape(-1, 3).clamp(0, 1).cpu().numpy()
Rdome = 400.0
center = traj.mean(0)
spacing = Rdome * math.sqrt(4 * math.pi / M)
sky_xyz = center + dirs * Rdome
sky_size = write_ply(os.path.join(a.out, "sky.ply"), sky_xyz, (rgb - 0.5) / 0.28209479, None,
                     np.full(len(dirs), 6.0), np.full((len(dirs), 3), math.log(spacing * 0.9)),
                     np.tile([1.0, 0, 0, 0], (len(dirs), 1)))
print("sky.ply", len(dirs), sky_size / 1e6, "MB")

# ---------------------------------------------------------------- actors
info = json.load(open(os.path.join(data_path, "instances", "instances_info.json")))
actors_out = os.path.join(a.out, "actors"); os.makedirs(actors_out, exist_ok=True)
actor_tracks = {}
n_act = 0
if "RigidNodes" in trainer.models:
    rn = trainer.models["RigidNodes"]
    init = dataset.get_init_objects(cur_node_type="RigidNodes", **cfg.model.RigidNodes.init)
    ds_ids = list(init.keys())
    assert len(ds_ids) == rn.num_instances, (len(ds_ids), rn.num_instances)
    true_ids = dataset.pixel_source.instances_true_id
    pid = rn.point_ids[:, 0].cpu().numpy()
    axyz, adc, arest, aop, asc, aq = gparams(rn)
    quats = rn.instances_quats.detach()
    quats = (quats / quats.norm(dim=-1, keepdim=True)).cpu().numpy()
    trans = rn.instances_trans.detach().cpu().numpy()
    fv = rn.instances_fv.cpu().numpy().astype(bool)
    for k, dsid in enumerate(ds_ids):
        tinfo = info[str(int(true_ids[dsid]))]
        tid = tinfo["id"][:8]
        m = (pid == k) & (1 / (1 + np.exp(-aop)) > 0.02)
        lim = np.abs(axyz) <= np.asarray(rn.instances_size[k].cpu().numpy()) / 2 * 1.1 + 0.2  # drop gaussians outside the box
        m &= lim.all(1)
        if m.sum() < 50:
            continue
        write_ply(os.path.join(actors_out, f"{tid}.ply"), axyz[m], adc[m], arest[m] if arest is not None else None,
                  aop[m], asc[m], aq[m])
        actor_tracks[tid] = dict(k=k, cls=tinfo["class_name"], n=int(m.sum()))
        n_act += 1
print("actors:", n_act)

# ---------------------------------------------------------------- tracks.json
t0 = seg["frames"][0]["lidar_ts"]
frames = []
for f in range(N):
    j0, j1 = max(f - 1, 0), min(f + 1, N - 1)
    dt = (seg["frames"][j1]["lidar_ts"] - seg["frames"][j0]["lidar_ts"]) / 1e9
    speed = float(np.linalg.norm(traj[j1] - traj[j0]) / max(dt, 1e-6))
    acts = {}
    for tid, inf in actor_tracks.items():
        k = inf["k"]
        if fv[f, k]:
            acts[tid] = {"p": [round(float(x), 4) for x in trans[f, k]], "q": [round(float(x), 6) for x in quats[f, k]]}
    frames.append({"t": round((seg["frames"][f]["lidar_ts"] - t0) / 1e9, 4),
                   "ego": {"p": [round(float(x), 4) for x in traj[f]], "q": [round(x, 6) for x in rot2quat(ego[f, :3, :3])]},
                   "speed_mps": round(speed, 3), "actors": acts})
cams = seg["cams"]
json.dump({"fps": 10, "num_frames": N, "frames": frames, "cameras": cams, "ego_size": [4.8, 2.0, 1.5],
           "actor_classes": {k: v["cls"] for k, v in actor_tracks.items()}},
          open(os.path.join(a.out, "tracks.json"), "w"))

# ---------------------------------------------------------------- camera strip images (uncropped originals)
for c in cams:
    os.makedirs(os.path.join(a.out, "cams", c), exist_ok=True)
for f, fr in enumerate(seg["frames"]):
    for c in cams:
        im = Image.open(os.path.join(a.raw_log, "sensors", "cameras", c, f"{fr['cams'][c]}.jpg")).convert("RGB")
        im.resize((320, round(im.height * 320 / im.width)), Image.LANCZOS).save(
            os.path.join(a.out, "cams", c, f"{f:04d}.jpg"), quality=70)
        if f == 0:
            im.resize((1280, round(im.height * 1280 / im.width)), Image.LANCZOS).save(
                os.path.join(a.out, "cams", c, "full_0000.jpg"), quality=80)

# ---------------------------------------------------------------- previews
prev = os.path.join(a.out, "previews"); os.makedirs(prev, exist_ok=True)


def render(c2w, K, W, H, f, img_idx=None, novel=True):
    c2w = torch.from_numpy(c2w).float().to(dev); K = torch.from_numpy(K).float().to(dev)
    x, y = torch.meshgrid(torch.arange(W, device=dev), torch.arange(H, device=dev), indexing="xy")
    _, viewdirs, _ = get_rays(x.flatten().float(), y.flatten().float(), c2w, K)
    nt = trainer.normalized_timestamps[f]
    iid = img_idx if img_idx is not None else f * len(cams)
    image_infos = {"viewdirs": viewdirs.reshape(H, W, 3), "normed_time": torch.full((H, W), float(nt), device=dev),
                   "img_idx": torch.full((H, W), iid, dtype=torch.long, device=dev)}
    cam_infos = {"camera_to_world": c2w, "intrinsics": K, "height": torch.tensor(H, device=dev),
                 "width": torch.tensor(W, device=dev)}
    with torch.no_grad():
        out = trainer(image_infos, cam_infos, novel_view=novel)
    return (out["rgb"].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def reality_view_c2w(f, back=7.0, up=4.0, pitch_deg=12.0):
    Re, pe = ego[f, :3, :3], ego[f, :3, 3]
    pos = pe + Re @ np.array([-back, 0, up])
    p = math.radians(pitch_deg)
    fwd = Re @ np.array([math.cos(p), 0, -math.sin(p)])
    right = Re @ np.array([0, -1.0, 0])
    right = right - fwd * (right @ fwd); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, pos
    return c2w


psnr_note = {}
if not a.skip_previews:
    W, H = 1280, 720
    fov = math.radians(70)
    fx = (W / 2) / math.tan(fov / 2)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]])
    for f in sorted(set([0, N // 5, 2 * N // 5, 3 * N // 5, 4 * N // 5, N - 1])):
        img = render(reality_view_c2w(f), K, W, H, f)
        Image.fromarray(img).save(os.path.join(prev, f"reality_view_f{f:03d}.png"))
        print("preview", f)
    # comparison: an actual camera pose (front center, a held-out test frame if available)
    test_ts = list(dataset.test_timesteps) if dataset.test_timesteps is not None else []
    fsel = int(test_ts[len(test_ts) // 2]) if len(test_ts) else N // 2
    idx = fsel * len(cams) + 0
    image_infos, cam_infos = dataset.full_image_set.get_image(idx, camera_downscale=1)
    for k_, v_ in image_infos.items():
        if isinstance(v_, torch.Tensor): image_infos[k_] = v_.to(dev)
    for k_, v_ in cam_infos.items():
        if isinstance(v_, torch.Tensor): cam_infos[k_] = v_.to(dev)
    with torch.no_grad():
        out = trainer(image_infos, cam_infos, novel_view=False)
    pred = out["rgb"].clamp(0, 1); gt = image_infos["pixels"]
    psnr = float(-10 * torch.log10(((pred - gt) ** 2).mean()))
    side = torch.cat([gt, pred], 1).cpu().numpy()
    Image.fromarray((side * 255).astype(np.uint8)).save(os.path.join(prev, f"real_camera_front_f{fsel:03d}_gt_vs_render.png"))
    psnr_note = {"compare_frame": fsel, "compare_is_heldout": bool(fsel in test_ts), "compare_psnr": round(psnr, 2)}
    print("compare", psnr_note)

# ---------------------------------------------------------------- meta + sanity
metrics = {}
mdir = os.path.join(log_dir, "metrics")
if os.path.isdir(mdir):
    for fn in os.listdir(mdir):
        try:
            metrics[fn] = json.load(open(os.path.join(mdir, fn)))
        except Exception:
            pass
rels = []
for i in range(0, N, 8):  # road height relative to the ego pose, from the 2.5-5 m ring around the car
    dd = np.linalg.norm(xyz[keep][:, :2] - traj[i, :2], axis=1); zz = xyz[keep][:, 2] - traj[i, 2]
    mm = (dd > 2.5) & (dd < 5) & (opa[keep] > 0.3) & (np.abs(zz) < 1.5)
    if mm.sum() > 50:
        h, bins = np.histogram(zz[mm], bins=np.arange(-1.5, 1.5, 0.05)); rels.append(bins[h.argmax()] + 0.025)
road_z = float(np.median(rels)) if rels else None
meta = {"name": a.name, "source": f"Argoverse 2 Sensor, log {seg['log_id']}, CC BY-NC-SA 4.0",
        "segment": [seg["t0"], seg["t1"]], "num_gaussians_bg": n_bg, "sh_degree_in_ply": 1 if rest is not None else 0, "num_actors": n_act,
        "train_iters": int(trainer.step), "gpu": a.gpu, "train_minutes": a.train_minutes,
        "road_z_below_ego": road_z, "metrics_files": metrics, **psnr_note,
        "notes": ""}
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in meta.items() if k != "metrics_files"}, indent=1))
