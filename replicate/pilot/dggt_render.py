"""DGGT (xiaomi-research/dggt, AV2 checkpoint) on the Miami segment, rendered from the chase camera and from
held-out real cameras.

DGGT predicts per-pixel Gaussians, camera poses and a sky model from unposed images, in its own up-to-scale frame
(world = first input camera). We align that frame to the AV2 metric world with a similarity transform fitted on the
input cameras (rotation from the camera orientations, then scale + translation from the camera centres), then render
any metric camera (chase cam, or a held-out ring camera) with gsplat.

Modes
  chase : one causal window per output frame t: frames {t-9, t-6, t-3, t} (clamped), all 7 ring cams (or --cams).
          Renders the chase camera (960x540) at frame t, with the dynamic Gaussians of frame t.
  eval  : for every 10th frame t (10..100, the splat's held-out frames): window {t-4, t-2, t+2, t+4} (frame t never
          seen), renders all 7 real cameras at frame t at the drivestudio eval resolution (half res, 1024x775).
          Dynamic Gaussians come from the nearest input frames (t-2 and t+2).

usage: python dggt_render.py --dggt /root/dggt --ckpt .../model_latest_argoverse.pt --proc P --raw L \
           --tracks tracks.json --mode chase --out /root/rep/dggt_chase
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--dggt", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--proc", required=True)
ap.add_argument("--raw", required=True)
ap.add_argument("--tracks", required=True)
ap.add_argument("--mode", choices=["chase", "eval"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cams", default="0,1,2,3,4,5,6")
ap.add_argument("--frames", default="")          # chase: subset of frames, e.g. "0,50"
ap.add_argument("--stride", type=int, default=3)  # chase window stride (frames)
ap.add_argument("--nwin", type=int, default=4)    # chase window length (frames)
ap.add_argument("--bf16", action="store_true")
ap.add_argument("--W", type=int, default=960)
ap.add_argument("--H", type=int, default=540)
a = ap.parse_args()

sys.path.insert(0, a.dggt)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import get_split_gs
from datasets.dataset import load_and_preprocess_images
from gsplat.rendering import rasterization
from common import Scene, chase_camera, CHASE

dev = "cuda"
CAMS = [int(c) for c in a.cams.split(",")]
os.makedirs(a.out, exist_ok=True)
sc = Scene(a.proc, a.raw, a.tracks, img_scale=0.5)
N = sc.n

model = VGGT().to(dev)
model.load_state_dict(torch.load(a.ckpt, map_location="cpu"), strict=True)
model.eval()


def alpha_t(t, t0, alpha, gamma0, gamma1=0.1):  # DGGT lifespan head: opacity decays away from the source time
    sigma = math.log(gamma1) / (gamma0 ** 2 + 1e-6)
    return (alpha * torch.exp(sigma * (t0 - t) ** 2)).float()


def sim3_from_cams(c2w_pred, c2w_true):
    """x_true = s * R @ x_pred + t. R from the camera orientations, s/t from the centres."""
    Rp, Rt = c2w_pred[:, :3, :3], c2w_true[:, :3, :3]
    M = sum(Rt[i] @ Rp[i].T for i in range(len(Rp)))
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    Cp, Ct = c2w_pred[:, :3, 3], c2w_true[:, :3, 3]
    mp, mt = Cp.mean(0), Ct.mean(0)
    X = (Cp - mp) @ R.T
    s = float(np.sum(X * (Ct - mt)) / max(np.sum(X * X), 1e-9))
    t = mt - s * R @ mp
    res_pos = np.linalg.norm(s * Cp @ R.T + t - Ct, axis=1)
    res_rot = [math.degrees(math.acos(np.clip((np.trace((R @ Rp[i]).T @ Rt[i]) - 1) / 2, -1, 1))) for i in range(len(Rp))]
    return s, R, t, float(res_pos.mean()), float(np.mean(res_rot))


def run_window(frames):
    """DGGT forward on frames x CAMS. Returns everything needed to render."""
    paths = [sc.image_path(f, c) for f in frames for c in CAMS]
    mpaths = [f"{a.proc}/sky_masks/{f:03d}_{c}.png" for f in frames for c in CAMS]
    imgs = load_and_preprocess_images(paths).to(dev)[None]          # 1,S,3,518,518 (white padded)
    masks = load_and_preprocess_images(mpaths).to(dev)[None]        # padding = 1 = sky
    S = imgs.shape[1]
    nf = len(frames)
    ts = np.array(frames, float) - frames[0]
    ts = ts / max(ts[-1], 1) * (nf / 4)
    ts = torch.tensor(np.repeat(ts, len(CAMS)), device=dev, dtype=torch.float32)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
        pred = model(imgs)
    with torch.no_grad():
        H, W = imgs.shape[-2:]
        ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"].float(), (H, W))
        ext = torch.cat([ext[0], torch.tensor([0, 0, 0, 1.0], device=dev).view(1, 1, 4).expand(S, 1, 4)], 1)
        intr = intr[0]
        depth = pred["depth"][0].float()
        pts = torch.from_numpy(unproject_depth_map_to_point_map(depth, ext[:, :3], intr)).to(dev).float()[None]
        gs_map, gs_conf = pred["gs_map"].float(), pred["gs_conf"].float()
        dy = pred["dynamic_conf"].squeeze(-1).float()
        bg = (masks.permute(0, 1, 3, 4, 2) == 0).any(-1)
        st = bg & (dy < 0.5)
        g = dict(st_pts=pts[st].reshape(-1, 3))
        g["st_rgb"], op, g["st_sc"], g["st_rot"] = get_split_gs(gs_map, st)
        g["st_op"] = op * (1 - dy[st].sigmoid())
        g["st_conf"] = gs_conf[st]
        g["st_t"] = ts[torch.nonzero(st, as_tuple=False)[:, 1]]
        g["dyn"] = []
        for i in range(S):
            m = bg[:, i]
            rgb, op, sc_, rot = get_split_gs(gs_map[:, i], m)
            g["dyn"].append((pts[:, i][m].reshape(-1, 3), rgb, op * dy[:, i][m].sigmoid(), sc_, rot))
        # sky colours for the dome points this window sees
        sky = model.sky_model
        K4 = torch.eye(4, device=dev).repeat(S, 1, 1); K4[:, :3, :3] = intr
        sky_rgb, sky_mask, _ = sky._get_background_color(imgs, ext, K4)
        g["sky"] = (sky.bg_pcd[sky_mask], sky.bg_quat[sky_mask], torch.exp(sky.bg_scales)[sky_mask],
                    sky.bg_opacity.squeeze(-1)[sky_mask], sky_rgb)
    torch.cuda.synchronize(); fwd = time.perf_counter() - t0
    # align DGGT frame -> metric world using the input cameras
    c2w_pred = np.linalg.inv(ext.cpu().numpy().astype(np.float64))
    c2w_true = np.stack([sc.cam_world(f, c) for f in frames for c in CAMS])
    s, R, t, rp, rr = sim3_from_cams(c2w_pred, c2w_true)
    g.update(ts=ts, S=S, s=s, R=R, t=t, res_pos=rp, res_rot=rr, fwd=fwd, nimg=S)
    return g


def render(g, c2w_world, K, W, H, img_ids, t_now):
    """render the window's Gaussians from a metric camera. img_ids: images whose dynamic Gaussians are used."""
    R, s, t = g["R"], g["s"], g["t"]
    c2w = np.eye(4)
    c2w[:3, :3] = R.T @ c2w_world[:3, :3]
    c2w[:3, 3] = R.T @ (c2w_world[:3, 3] - t) / s
    vm = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev)[None]
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)[None]
    op = alpha_t(g["st_t"], t_now, g["st_op"], g["st_conf"])
    parts = [(g["st_pts"], g["st_rgb"], op, g["st_sc"], g["st_rot"])] + [g["dyn"][i] for i in img_ids]
    P, C, O, Sc, Q = [torch.cat([p[k] for p in parts]) for k in range(5)]
    out, alpha, _ = rasterization(means=P, quats=Q, scales=Sc, opacities=O, colors=C, viewmats=vm, Ks=Kt,
                                  width=W, height=H, render_mode="RGB+ED")
    rgb, depth = out[0, ..., :3], out[0, ..., 3] * s
    sp, sq, ss, so, scol = g["sky"]
    skyr, _, _ = rasterization(means=sp, quats=sq, scales=ss, opacities=so, colors=scol, viewmats=vm, Ks=Kt,
                               width=W, height=H)
    skyr = skyr[0]
    skyr = (skyr - skyr.min()) / (skyr.max() - skyr.min() + 1e-8)   # as DGGT mode 2 does
    img = alpha[0] * rgb + (1 - alpha[0]) * skyr
    return img.clamp(0, 1), depth, alpha[0, ..., 0]


def save(img, path):
    Image.fromarray((img.cpu().numpy() * 255).round().astype(np.uint8)).save(path, quality=93)


log = []
if a.mode == "chase":
    frames_out = [int(x) for x in a.frames.split(",")] if a.frames else list(range(N))
    rts = []
    for fo in frames_out:
        win = sorted(set(max(0, fo - k * a.stride) for k in range(a.nwin - 1, -1, -1)))
        g = run_window(win)
        p, yaw = sc.ego_pq(fo)
        c2w, K = chase_camera(p, yaw, dict(CHASE, W=a.W, H=a.H))
        k = win.index(fo)
        ids = [k * len(CAMS) + j for j in range(len(CAMS))]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        img, depth, alpha = render(g, c2w, K, a.W, a.H, ids, g["ts"][ids[0]])
        torch.cuda.synchronize(); rts.append(time.perf_counter() - t0)
        save(img, f"{a.out}/{fo:04d}.jpg")
        os.makedirs(f"{a.out}/geo", exist_ok=True)
        np.savez_compressed(f"{a.out}/geo/{fo:04d}.npz", depth=depth.half().cpu().numpy(),
                            hit=((alpha > 0.9) & (depth < 150)).cpu().numpy(), c2w=c2w, K=K)
        log.append(dict(frame=fo, win=win, s=g["s"], res_pos_m=g["res_pos"], res_rot_deg=g["res_rot"],
                        fwd_s=g["fwd"], render_ms=rts[-1] * 1000, n_gauss=int(g["st_pts"].shape[0])))
        print(log[-1], flush=True)
        del g; torch.cuda.empty_cache()
else:
    for t in range(10, 101, 10):
        win = [t - 4, t - 2, t + 2, t + 4]
        g = run_window(win)
        tn = (g["ts"][1 * len(CAMS)] + g["ts"][2 * len(CAMS)]) / 2
        for ci in range(7):
            # dynamic Gaussians of the same camera in the two neighbouring frames
            ids = [1 * len(CAMS) + CAMS.index(ci), 2 * len(CAMS) + CAMS.index(ci)] if ci in CAMS else []
            gt = Image.open(sc.image_path(t, ci))
            w0, h0 = gt.size
            W, H = w0 // 2, h0 // 2
            img, _, _ = render(g, sc.cam_world(t, ci), sc.K[ci], W, H, ids, tn)
            save(img, f"{a.out}/{t:03d}_{ci}.jpg")
        log.append(dict(frame=t, win=win, s=g["s"], res_pos_m=g["res_pos"], res_rot_deg=g["res_rot"], fwd_s=g["fwd"]))
        print(log[-1], flush=True)
        del g; torch.cuda.empty_cache()
json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
fw = [l["fwd_s"] for l in log[2:]] or [l["fwd_s"] for l in log]
summ = dict(mode=a.mode, cams=CAMS, bf16=a.bf16, fwd_s_median=float(np.median(fw)),
            res_pos_m_median=float(np.median([l["res_pos_m"] for l in log])),
            res_rot_deg_median=float(np.median([l["res_rot_deg"] for l in log])))
if a.mode == "chase":
    summ["render_ms_median"] = float(np.median([l["render_ms"] for l in log[2:]] or [0]))
json.dump(summ, open(f"{a.out}/summary.json", "w"), indent=1)
print(summ)
