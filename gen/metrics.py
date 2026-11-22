"""Revisit / flicker metric: how much does a static surface change colour between frame t and frame t+k?

For every pair (t, t+k), pixels of frame t+k that show static lidar-backed geometry (known, not a moving object)
are back-projected with the lift depth, projected into frame t, kept only if frame t sees the same surface
(depth agrees), and the colours are compared. Same pixel set for every method, so the numbers are comparable.
Higher PSNR = more consistent. Also reports "fidelity": PSNR of each method vs the raw lift on known pixels
(how far the repaint moved away from the measured colours).
usage: python metrics.py --geo out_live/geo --seq lift=out_live/raw_lift live=out_live/live splat=out_splat ...
"""
import argparse, glob, json, os
import numpy as np, torch, torch.nn.functional as F
from torchvision.io import read_image

DEV = "cuda"


def load_geo(d, fi):
    z = np.load(f"{d}/{fi:04d}.npz")
    return {k: z[k] for k in z.files}


def backproject(depth, c2w, K):
    H, W = depth.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=DEV).float(), torch.arange(W, device=DEV).float(), indexing="ij")
    rays = torch.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], torch.ones_like(xs)], -1)
    c2w = torch.as_tensor(c2w, dtype=torch.float32, device=DEV)
    return (rays * depth[..., None]) @ c2w[:3, :3].T + c2w[:3, 3]


def pair_mask_and_grid(ga, gb):
    """pixels of b visible in a (static, known, depth-consistent) and their sampling grid into a."""
    db = torch.as_tensor(gb["depth"].astype(np.float32), device=DEV)
    mb = torch.as_tensor(gb["hit"] & ~gb["dyn"], device=DEV)
    X = backproject(db, gb["c2w"], gb["K"])
    w2c = torch.as_tensor(np.linalg.inv(ga["c2w"]), dtype=torch.float32, device=DEV)
    pc = X @ w2c[:3, :3].T + w2c[:3, 3]
    K = ga["K"]
    z = pc[..., 2].clamp(min=1e-3)
    u = K[0, 0] * pc[..., 0] / z + K[0, 2]
    v = K[1, 1] * pc[..., 1] / z + K[1, 2]
    H, W = db.shape
    g = torch.stack([u / (W - 1) * 2 - 1, v / (H - 1) * 2 - 1], -1)[None]
    da = torch.as_tensor((ga["depth"] * (ga["hit"] & ~ga["dyn"])).astype(np.float32), device=DEV)
    wd = F.grid_sample(da[None, None], g, mode="nearest", align_corners=True)[0, 0]
    ok = mb & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1) & (pc[..., 2] > 0.3) & (wd > 0) \
        & ((wd - pc[..., 2]).abs() < 0.03 * pc[..., 2] + 0.2)
    return ok, g


def img(d, fi):
    return read_image(f"{d}/{fi:04d}.jpg").to(DEV).float() / 255


def psnr(a, b, m):
    mse = ((a - b) ** 2).mean(0)[m].mean()
    return float(10 * torch.log10(1 / mse.clamp(min=1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", required=True); ap.add_argument("--seq", nargs="+", required=True)
    ap.add_argument("--ks", default="1,5,10"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seqs = dict(s.split("=") for s in a.seq)
    n = len(glob.glob(f"{a.geo}/*.npz"))
    res = {name: {} for name in seqs}
    geos = [load_geo(a.geo, i) for i in range(n)]
    for k in map(int, a.ks.split(",")):
        vals = {name: [] for name in seqs}
        for t in range(0, n - k):
            ok, g = pair_mask_and_grid(geos[t], geos[t + k])
            if ok.sum() < 1000:
                continue
            for name, d in seqs.items():
                ia, ib = img(d, t), img(d, t + k)
                wa = F.grid_sample(ia[None], g, align_corners=True)[0]
                vals[name].append(psnr(wa, ib, ok))
        for name in seqs:
            res[name][f"revisit_psnr_k{k}"] = float(np.mean(vals[name]))
    # fidelity vs the raw lift on known static pixels
    if "lift" in seqs:
        for name, d in seqs.items():
            v = []
            for t in range(n):
                m = torch.as_tensor(geos[t]["hit"] & ~geos[t]["dyn"], device=DEV)
                v.append(psnr(img(d, t), img(seqs["lift"], t), m))
            res[name]["psnr_vs_lift_known"] = float(np.mean(v))
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
