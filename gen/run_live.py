"""Live mode: lift + one-step repaint, frame by frame, causal. Writes
  <out>/raw_lift/0000.jpg   point render before repaint (holes black)
  <out>/live/0000.jpg       Difix3D+ repaint + temporal blend (the deliverable)
  <out>/geo/0000.npz        depth / known / dynamic masks / camera (for the consistency metrics)
  <out>/front/0000.jpg      real ring_front_center frame (for the comparison video)
  <out>/timing.json         per-stage ms per frame
usage: python run_live.py --proc data/processed/000 --raw data/raw/val/<log> --tracks tracks.json --out out [--mono]
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn.functional as F
from torchvision.io import write_jpeg

from common import Scene, CHASE
from lift import Lifter, pull_push, sky_prior
from repaint import Difix, front_panorama, warp_prev


def save(img, path, q=85):
    write_jpeg((img.clamp(0, 1) * 255).round().byte().permute(2, 0, 1).cpu(), path, quality=q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", required=True); ap.add_argument("--raw", required=True)
    ap.add_argument("--tracks", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--mono", action="store_true"); ap.add_argument("--no_ground", action="store_true"); ap.add_argument("--hist", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.35, help="weight of the new repaint inside warp-consistent pixels")
    ap.add_argument("--no_difix", action="store_true"); ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--keep_known", type=float, default=0.0, help="blend raw lift colour back into known pixels")
    a = ap.parse_args()
    for d in ["raw_lift", "live", "geo", "front", "filled"]:
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    sc = Scene(a.proc, a.raw, a.tracks)
    lf = Lifter(sc, hist=a.hist, mono=a.mono, ground=not a.no_ground)
    dfx = None if a.no_difix else Difix()
    n = a.frames or sc.n
    prev = None
    times = []
    for fi in range(n):
        torch.cuda.synchronize(); t_start = time.perf_counter()
        lf.t = {}
        cur = lf.ingest(fi)
        r = lf.render(fi, cur)
        t0 = time.perf_counter()
        sc_col = getattr(lf, "sky_col", None)
        if sc_col is None:
            sc_col = torch.tensor([[0.45, 0.62, 0.85], [0.75, 0.82, 0.9]], device="cuda")
        pre, pre_hit = sky_prior(r["img"], r["hit"], r["c2w"], r["K"], sc_col)
        filled = pull_push(pre, pre_hit)
        torch.cuda.synchronize(); lf.t["fill"] = (time.perf_counter() - t0) * 1000; t0 = time.perf_counter()
        K = torch.as_tensor(r["K"], dtype=torch.float32, device="cuda")
        if dfx is not None:
            ref = front_panorama(cur["ims"])
            out = dfx(filled, ref)
            torch.cuda.synchronize(); lf.t["difix"] = (time.perf_counter() - t0) * 1000; t0 = time.perf_counter()
        else:
            out = filled
        if a.keep_known > 0:
            m = (r["hit"] & ~r["dyn"]).float()[..., None] * a.keep_known
            out = out * (1 - m) + r["img"] * m
        if prev is not None and a.alpha < 1:
            static = r["hit"] & ~r["dyn"]
            wimg, ok = warp_prev(prev["img"], prev["depth"], prev["c2w"], K, r["depth"], static, r["c2w"])
            okf = ok.float()[..., None]
            out = out * (1 - okf * (1 - a.alpha)) + wimg * okf * (1 - a.alpha)
        torch.cuda.synchronize(); lf.t["temporal"] = (time.perf_counter() - t0) * 1000
        lf.t["total"] = (time.perf_counter() - t_start) * 1000
        prev = dict(img=out, depth=r["depth"] * r["hit"], c2w=r["c2w"])
        times.append(dict(lf.t))
        # outputs (not timed)
        save(r["img"], f"{a.out}/raw_lift/{fi:04d}.jpg")
        save(filled, f"{a.out}/filled/{fi:04d}.jpg")
        save(out, f"{a.out}/live/{fi:04d}.jpg")
        fc = cur["ims"][0]
        save(F.interpolate(fc[None], (540, int(540 * fc.shape[2] / fc.shape[1])), mode="area")[0].permute(1, 2, 0),
             f"{a.out}/front/{fi:04d}.jpg")
        np.savez_compressed(f"{a.out}/geo/{fi:04d}.npz", depth=r["depth"].half().cpu().numpy(),
                            hit=r["hit"].cpu().numpy(), dyn=r["dyn"].cpu().numpy(), c2w=r["c2w"], K=r["K"])
        print(fi, " ".join(f"{k}={v:.1f}" for k, v in lf.t.items()), flush=True)
    keys = times[0].keys() | times[-1].keys()
    summ = {k: dict(median=float(np.median([t.get(k, 0) for t in times[5:]])),
                    mean=float(np.mean([t.get(k, 0) for t in times[5:]]))) for k in keys}
    json.dump(dict(per_frame=times, summary=summ, args=vars(a), gpu=torch.cuda.get_device_name()),
              open(f"{a.out}/timing.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
