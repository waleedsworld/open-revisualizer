"""Speed of the one-step I2I step alone on this GPU: Difix3D+ with / without reference, at 960x540 (run as
544x960) and 512x288 (run as 288x512), fp16, eager vs torch.compile. Also the bare SD-Turbo-size UNet and
VAE parts, to see where the time goes."""
import json, time, sys
import torch
from repaint import Difix


def tm(f, n=10):
    for _ in range(3):
        f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n):
        f()
    torch.cuda.synchronize(); return round((time.perf_counter() - t) / n * 1000, 1)


# one model per process: both repos ship a custom unet_2d_condition.py and diffusers caches remote code by
# file name, so loading the second model in the same process silently reuses the first one's code.
MODELS = {"difix_ref": "/root/gen/weights/difix_ref", "difix": "/root/gen/weights/difix"}
name_arg = sys.argv[2] if len(sys.argv) > 2 else "difix_ref"
res = {}
x = torch.rand(540, 960, 3, device="cuda")
for name, w in ((name_arg, MODELS[name_arg]),):
    for comp in (False, True):
        d = Difix(weights=w, compile=comp)
        for size in ((544, 960), (288, 512)):
            d.size = size
            ref = x if name == "difix_ref" else None
            k = f"{name} {size[1]}x{size[0]} {'compiled' if comp else 'eager'}"
            res[k] = tm(lambda: d(x, ref))
            print(k, res[k], flush=True)
        del d; torch.cuda.empty_cache()
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "bench_i2i.json", "w"), indent=1)
