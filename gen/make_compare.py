"""2x2 comparison video at 10 fps: real front camera | splat | splat + I2I | lift + I2I (labelled).
usage: python3 make_compare.py --front F --splat S --splat_i2i SI --live L --out out/compare.mp4
(--lift R adds a 3x2 layout with the raw lift as a fifth panel and the live without temporal blend if given)
"""
import argparse, os, subprocess, tempfile
from PIL import Image, ImageDraw, ImageFont


def font(sz):
    for f in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def panel(path, label, F, W=960, H=540):
    im = Image.open(path).convert("RGB")
    if im.size != (W, H):  # portrait front camera: fit inside, centred on black
        s = min(W / im.width, H / im.height)
        im2 = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        im = Image.new("RGB", (W, H)); im.paste(im2, ((W - im2.width) // 2, (H - im2.height) // 2))
    d = ImageDraw.Draw(im)
    tw = d.textlength(label, font=F)
    d.rectangle([10, 10, 26 + tw, 50], fill=(0, 0, 0))
    d.text((18, 14), label, fill=(255, 255, 255), font=F)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front"); ap.add_argument("--splat"); ap.add_argument("--splat_i2i"); ap.add_argument("--live")
    ap.add_argument("--lift", default=None); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=105)
    a = ap.parse_args()
    F = font(28)
    panels = [(a.front, "real front camera"), (a.splat, "per-scene splat"),
              (a.splat_i2i, "splat + I2I (Difix3D+)"), (a.live, "lift + I2I (live, per frame)")]
    if a.lift:
        panels.insert(2, (a.lift, "lift only (no repaint)"))
    cols = 2 if len(panels) == 4 else 3
    rows = (len(panels) + cols - 1) // cols
    tmp = tempfile.mkdtemp()
    for fi in range(a.n):
        sheet = Image.new("RGB", (960 * cols, 540 * rows))
        for i, (d, lab) in enumerate(panels):
            sheet.paste(panel(f"{d}/{fi:04d}.jpg", lab, F), ((i % cols) * 960, (i // cols) * 540))
        ImageDraw.Draw(sheet).text((960 * cols - 150, 540 * rows - 40), f"frame {fi:03d}", fill=(255, 255, 0), font=F)
        sheet.save(f"{tmp}/{fi:04d}.jpg", quality=92)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-i", f"{tmp}/%04d.jpg", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", a.out], check=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
