#!/usr/bin/env python3
"""Sky masks for drivestudio with HF transformers SegFormer-B5 (Cityscapes, sky = class 10).

Replaces drivestudio's datasets/tools/extract_masks.py (which needs mmcv-full 1.2.7 / torch 1.8).
Writes <scene>/sky_masks/{t:03d}_{cam}.png, uint8 0/255, same size as the image, exactly as drivestudio expects.

usage: python sky_masks.py --scene_dir data/argoverse/processed/training/000
"""
import argparse, glob, os
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--scene_dir", required=True)
ap.add_argument("--model", default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
ap.add_argument("--infer_width", type=int, default=1024)
a = ap.parse_args()

proc = SegformerImageProcessor.from_pretrained(a.model)
model = SegformerForSemanticSegmentation.from_pretrained(a.model).cuda().eval().half()
mean = torch.tensor(proc.image_mean, device="cuda").view(1, 3, 1, 1)
std = torch.tensor(proc.image_std, device="cuda").view(1, 3, 1, 1)
out_dir = os.path.join(a.scene_dir, "sky_masks"); os.makedirs(out_dir, exist_ok=True)
files = sorted(glob.glob(os.path.join(a.scene_dir, "images", "*.jpg")))
for i, f in enumerate(files):
    im = Image.open(f).convert("RGB"); W, H = im.size
    s = a.infer_width / max(W, H)
    x = torch.from_numpy(np.asarray(im.resize((round(W * s), round(H * s)), Image.BILINEAR))).cuda()
    x = (x.permute(2, 0, 1)[None].float() / 255 - mean) / std
    with torch.no_grad():
        logits = model(pixel_values=x.half()).logits.float()
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    sky = (logits.argmax(1)[0] == 10).cpu().numpy()
    Image.fromarray((sky * 255).astype(np.uint8), "L").save(os.path.join(out_dir, os.path.basename(f)[:-4] + ".png"))
    if i % 100 == 0:
        print(f"{i}/{len(files)} sky={sky.mean():.3f}", flush=True)
print("done", len(files))
