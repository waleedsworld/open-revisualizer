"""Render the per-scene Gaussian splat export (train/ -> web/scenes/av2-miami-turn) from the same chase camera,
for the side-by-side comparison. Uses the original background.ply (SH0), the actor plys moved by tracks.json,
and sky.ply. gsplat rasterisation, 960x540.
usage: python render_splat.py --scene /root/gen/scene --tracks tracks.json --out /root/gen/out_splat
"""
import argparse, glob, json, os, time
import numpy as np, torch
from torchvision.io import write_jpeg
from gsplat import rasterization

from common import chase_camera, CHASE

C0 = 0.28209479177387814


def read_ply(f):
    b = open(f, "rb").read()
    i = b.index(b"end_header\n") + 11
    n = int([l for l in b[:i].decode().split("\n") if l.startswith("element vertex")][0].split()[-1])
    a = torch.from_numpy(np.frombuffer(b[i:], np.float32).reshape(n, 14).copy()).cuda()
    return dict(means=a[:, :3], rgb=(0.5 + C0 * a[:, 3:6]).clamp(0, 1), opac=torch.sigmoid(a[:, 6]),
                scales=torch.exp(a[:, 7:10]), quats=torch.nn.functional.normalize(a[:, 10:14], dim=1))


def qmat(q):  # wxyz -> 3x3
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def qmul(a, b):  # batched wxyz, a: (4,), b: (N,4)
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b.unbind(1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True); ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--no_sky", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr = json.load(open(a.tracks))
    bg = read_ply(f"{a.scene}/background.ply")
    sky = None if a.no_sky else read_ply(f"{a.scene}/sky.ply")
    actors = {os.path.basename(f)[:-4]: read_ply(f) for f in glob.glob(f"{a.scene}/actors/*.ply")}
    W, H = CHASE["W"], CHASE["H"]
    ts = []
    for fi, fr in enumerate(tr["frames"]):
        t0 = time.perf_counter()
        parts = [bg] + ([sky] if sky else [])
        for aid, pose in fr.get("actors", {}).items():
            if aid not in actors:
                continue
            g = actors[aid]
            Rm = torch.tensor(qmat(pose["q"]), dtype=torch.float32, device="cuda")
            parts.append(dict(means=g["means"] @ Rm.T + torch.tensor(pose["p"], device="cuda"), rgb=g["rgb"],
                              opac=g["opac"], scales=g["scales"],
                              quats=qmul(torch.tensor(pose["q"], dtype=torch.float32, device="cuda"), g["quats"])))
        cat = {k: torch.cat([p[k] for p in parts]) for k in bg}
        e = fr["ego"]
        w, x, y, z = e["q"]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c2w, K = chase_camera(np.array(e["p"], float), yaw)
        vm = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device="cuda")[None]
        Kt = torch.tensor(K, dtype=torch.float32, device="cuda")[None]
        out, alpha, _ = rasterization(cat["means"], cat["quats"], cat["scales"], cat["opac"], cat["rgb"], vm, Kt, W, H,
                                      render_mode="RGB+ED")
        img = out[..., :3] + (1 - alpha) * torch.tensor([0.62, 0.74, 0.88], device="cuda")  # flat sky behind the dome
        depth = out[0, ..., 3]
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
        write_jpeg((img[0].clamp(0, 1) * 255).byte().permute(2, 0, 1).cpu(), f"{a.out}/{fi:04d}.jpg", quality=90)
        # expected depth + a "solid" mask (opaque, not sky) for the temporal warp of the I2I pass
        os.makedirs(f"{a.out}/geo", exist_ok=True)
        solid = (alpha[0, ..., 0] > 0.9) & (depth < 150)
        dyn = torch.zeros_like(solid)
        if len(parts) > 1 + (sky is not None):  # moving actors: depth of an actors-only pass vs the full depth
            act = {k: torch.cat([p[k] for p in parts[1 + (sky is not None):]]) for k in bg}
            ao, aa, _ = rasterization(act["means"], act["quats"], act["scales"], act["opac"], act["rgb"], vm, Kt, W, H,
                                      render_mode="ED")
            dyn = (aa[0, ..., 0] > 0.5) & (ao[0, ..., 0] < depth + 1.0)
        np.savez_compressed(f"{a.out}/geo/{fi:04d}.npz", depth=depth.half().cpu().numpy(), hit=solid.cpu().numpy(),
                            dyn=dyn.cpu().numpy(), c2w=c2w, K=K)
    print("splat render ms median", np.median(ts[5:]))
    json.dump(dict(ms_median=float(np.median(ts[5:]))), open(f"{a.out}/timing.json", "w"))


if __name__ == "__main__":
    main()
