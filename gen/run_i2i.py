"""One-step image-to-image polish (Difix3D+ with the front cameras as reference) over an existing
chase-camera render sequence, e.g. the per-scene splat render ("splat + I2I", the presumed Tesla stack:
coarse 3D render -> small per-frame I2I diffusion). Temporal fix: the previous output is warped into the
current frame with the render's depth and blended into geometry-consistent pixels.
usage: python run_i2i.py --frames /root/gen/out_splat --geo /root/gen/out_splat/geo --proc ... --raw ... --out ...
"""
import argparse, json, os, time
import numpy as np, torch
from torchvision.io import read_image, write_jpeg

from common import Scene
from lift import load_images
from repaint import Difix, front_panorama, warp_prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--geo", required=True)
    ap.add_argument("--proc", required=True); ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--alpha", type=float, default=0.35)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sc = Scene(a.proc, a.raw)
    dfx = Difix()
    z = torch.rand(540, 960, 3, device="cuda")
    for _ in range(2):
        dfx(z, z)
    prev, ts = None, []
    for fi in range(sc.n):
        x = read_image(f"{a.frames}/{fi:04d}.jpg").cuda().permute(1, 2, 0).float() / 255
        g = np.load(f"{a.geo}/{fi:04d}.npz")
        depth = torch.as_tensor(g["depth"].astype(np.float32), device="cuda")
        hit = torch.as_tensor(g["hit"] & ~g["dyn"], device="cuda")  # static solid pixels only
        ims = load_images(sc, fi)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = dfx(x, front_panorama(ims))
        torch.cuda.synchronize(); t1 = time.perf_counter()
        K = torch.as_tensor(g["K"], dtype=torch.float32, device="cuda")
        if prev is not None and a.alpha < 1:
            wimg, ok = warp_prev(prev["img"], prev["depth"], prev["c2w"], K, depth, hit, g["c2w"])
            okf = ok.float()[..., None]
            out = out * (1 - okf * (1 - a.alpha)) + wimg * okf * (1 - a.alpha)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        ts.append(dict(i2i=(t1 - t0) * 1000, temporal=(t2 - t1) * 1000))
        prev = dict(img=out, depth=depth * hit, c2w=g["c2w"])
        write_jpeg((out.clamp(0, 1) * 255).round().byte().permute(2, 0, 1).cpu(), f"{a.out}/{fi:04d}.jpg", quality=85)
    summ = {k: float(np.median([t[k] for t in ts[5:]])) for k in ts[0]}
    json.dump(dict(per_frame=ts, summary_median_ms=summ, args=vars(a)), open(f"{a.out}/timing.json", "w"), indent=1)
    print(summ)


if __name__ == "__main__":
    main()
