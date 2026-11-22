"""Repaint: Difix3D+ (single-step diffusion, reference-guided) on the lifted chase-camera render,
plus temporal blending with the previous output warped through the lidar geometry.

Difix3D+ (nv-tlabs/Difix3D, HF nvidia/difix_ref) is under the NVIDIA non-commercial licence.
"""
import sys, math
import numpy as np, torch, torch.nn.functional as F

DEV = "cuda"


class Difix:
    def __init__(self, weights="/root/gen/weights/difix_ref", src="/root/gen/Difix3D/src", size=(544, 960),
                 dtype=torch.float16, timestep=199):
        sys.path.insert(0, src)
        from pipeline_difix import DifixPipeline
        self.pipe = DifixPipeline.from_pretrained(weights, trust_remote_code=True, torch_dtype=dtype).to(DEV)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.safety_checker = None
        self.size, self.dtype, self.timestep = size, dtype, timestep
        with torch.no_grad():
            self.pe, _ = self.pipe.encode_prompt("remove degradation", DEV, 1, False)

    @torch.no_grad()
    def __call__(self, img, ref):
        """img, ref: H,W,3 float [0,1] torch on GPU. Returns H,W,3 at the input size."""
        H, W = img.shape[:2]
        h, w = self.size
        x = F.interpolate(img.permute(2, 0, 1)[None], (h, w), mode="bilinear", align_corners=False)
        r = F.interpolate(ref.permute(2, 0, 1)[None], (h, w), mode="bilinear", align_corners=False)
        out = self.pipe(prompt_embeds=self.pe, image=x.clamp(0, 1), ref_image=r.clamp(0, 1) if ref is not None else None,
                        num_inference_steps=1, timesteps=[self.timestep], guidance_scale=0.0, height=h, width=w,
                        output_type="pt").images
        out = out.float()
        if out.dim() == 3:
            out = out[None]
        return F.interpolate(out, (H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).clamp(0, 1)


def front_panorama(ims, H=544, W=960):
    """reference image = ring_front_left | ring_front_center (middle part) | ring_front_right, squashed to WxH.
    These three cameras cover what the chase camera looks at."""
    fl, fc, fr = ims[1], ims[0], ims[2]
    def rs(im):
        h, w = im.shape[1:]
        return F.interpolate(im[None], (H, int(round(w * H / h))), mode="bilinear", align_corners=False)[0]
    # front_center is portrait: keep the band at horizon height so its aspect matches the side cameras
    h, w = fc.shape[1:]
    band = int(w * 1550 / 2048)
    y0 = max(0, int(1023 - band / 2))
    fc = fc[:, y0:y0 + band]
    pano = torch.cat([rs(fl), rs(fc), rs(fr)], 2)
    return F.interpolate(pano[None], (H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)


def backproject(depth, c2w, K):
    H, W = depth.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=DEV, dtype=torch.float32),
                            torch.arange(W, device=DEV, dtype=torch.float32), indexing="ij")
    rays = torch.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], torch.ones_like(xs)], -1)
    c2w = torch.as_tensor(c2w, dtype=torch.float32, device=DEV)
    return (rays * depth[..., None]) @ c2w[:3, :3].T + c2w[:3, 3]


def warp_prev(prev_img, prev_depth, prev_c2w, K, depth, valid, c2w, rel_tol=0.04, abs_tol=0.3):
    """Warp the previous output into the current view using the current depth.
    Returns the warped image and a mask of pixels where the warp is geometrically consistent."""
    H, W = depth.shape
    X = backproject(depth, c2w, K)
    w2c = torch.as_tensor(np.linalg.inv(prev_c2w), dtype=torch.float32, device=DEV)
    pc = X @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[..., 2].clamp(min=1e-3)
    u = K[0, 0] * pc[..., 0] / z + K[0, 2]
    v = K[1, 1] * pc[..., 1] / z + K[1, 2]
    g = torch.stack([u / (W - 1) * 2 - 1, v / (H - 1) * 2 - 1], -1)[None]
    wimg = F.grid_sample(prev_img.permute(2, 0, 1)[None], g, align_corners=True)[0].permute(1, 2, 0)
    wd = F.grid_sample(prev_depth[None, None], g, mode="nearest", align_corners=True)[0, 0]
    ok = valid & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1) & (pc[..., 2] > 0.3) \
        & (wd > 0) & ((wd - pc[..., 2]).abs() < rel_tol * pc[..., 2] + abs_tol)
    return wimg, ok
