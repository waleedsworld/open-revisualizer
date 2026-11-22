"""NVIDIA Fixer (nv-tlabs/Fixer, Cosmos-0.6B one-step I2I) over a folder of renders, keeping each image's aspect
ratio (padded to a multiple of 16 instead of squashed to 576x1024), plus a timing benchmark.

usage: python fixer_run.py --fixer /root/Fixer --in DIR --out DIR [--long 960] [--timestep 250] [--bench]
Needs /work/models/base/{model_fast_tokenizer.pt,tokenizer_fast.pth} (hard-coded in Fixer's model file).
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--fixer", required=True)
ap.add_argument("--model", default="/work/models/pretrained/pretrained_fixer.pkl")
ap.add_argument("--in", dest="inp", default="")
ap.add_argument("--out", default="")
ap.add_argument("--long", type=int, default=960, help="long side the model runs at (input is resized, output resized back)")
ap.add_argument("--timestep", type=int, default=250)
ap.add_argument("--bench", default="", help="comma list of HxW to time, e.g. 544x960,288x512")
ap.add_argument("--compile", action="store_true")
a = ap.parse_args()

sys.path.insert(0, f"{a.fixer}/src")
import transformer_engine as te  # noqa: F401  (Fixer: "important")
from pix2pix_turbo_nocond_cosmos_base_faster_tokenizer import Pix2Pix_Turbo

torch.set_grad_enabled(False)
dt = torch.bfloat16
model = Pix2Pix_Turbo(pretrained_path=a.model, timestep=a.timestep, vae_skip_connection=False, batch_size=1)
model = model.to("cuda", dt)
model.set_eval()
if a.compile:
    model = torch.compile(model)


def run(x):  # x: 1,3,H,W in [0,1], H,W multiple of 16
    with torch.autocast("cuda", dtype=dt):
        y = model((x * 2 - 1).to(dt))
    return (y.float() * 0.5 + 0.5).clamp(0, 1)


res = {}
if a.bench:
    for hw in a.bench.split(","):
        h, w = map(int, hw.split("x"))
        x = torch.rand(1, 3, h, w, device="cuda")
        for _ in range(5):
            run(x)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20):
            run(x)
        torch.cuda.synchronize()
        res[hw] = (time.perf_counter() - t0) / 20 * 1000
        print("fixer", hw, "ms", res[hw], flush=True)

if a.inp:
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(f"{a.inp}/*.jpg") + glob.glob(f"{a.inp}/*.png"))
    ts = []
    for f in files:
        im = Image.open(f).convert("RGB")
        w0, h0 = im.size
        k = a.long / max(w0, h0)
        w1, h1 = int(round(w0 * k / 16)) * 16, int(round(h0 * k / 16)) * 16
        x = torch.from_numpy(np.asarray(im.resize((w1, h1), Image.BICUBIC))).cuda().permute(2, 0, 1)[None].float() / 255
        torch.cuda.synchronize(); t0 = time.perf_counter()
        y = run(x)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        y = F.interpolate(y, size=(h0, w0), mode="bilinear", align_corners=False)
        Image.fromarray((y[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(
            f"{a.out}/{os.path.basename(f)[:-4]}.jpg", quality=93)
    res["folder_ms_median"] = float(np.median(ts[3:] if len(ts) > 5 else ts) * 1000)
    res["n"] = len(files)
    print(res)
if a.out:
    json.dump(res, open(f"{a.out}/fixer_timing.json", "w"), indent=1)
