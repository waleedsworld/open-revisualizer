"""Lift: colour lidar sweeps from the 7 ring cameras and render them into the virtual chase camera.

Per frame, causal (only this sweep and the previous ones that were already seen live):
  1. this sweep's points -> world, drop the ego car, split static / dynamic with the 3D boxes
  2. colour every point from the ring camera that sees it most head-on (projection + coarse z-buffer
     occlusion test per camera, bilinear sample at the camera's own exposure time)
  3. optional: monocular metric depth (Depth Anything V2) back-projected where the lidar has no returns
  4. static points of the last `hist` sweeps (each coloured when it arrived) + dynamic points of this sweep
  5. point splatting into the chase camera: square splats sized by depth, z-buffer via int64 atomic min,
     then a screen-space visibility filter that removes far points leaking through gaps in near surfaces
Outputs an RGB image, a hole mask, a depth map and a per-pixel "dynamic" mask.
"""
import math, time
import numpy as np, torch, torch.nn.functional as F
from torchvision.io import decode_jpeg, read_file

from common import Scene, chase_camera, CHASE
from ground import Ground

DEV = "cuda"
EGO_BOX = (-1.2, 4.0, -1.3, 1.3)   # x0 x1 y0 y1 in ego frame (rear axle origin): the ego car's own returns


def T_(a):
    return torch.as_tensor(a, dtype=torch.float32, device=DEV)


def load_images(sc, fi):
    """camera frames -> GPU float tensors (3,H,W)."""
    return [torch.from_numpy(a).to(DEV, non_blocking=True).permute(2, 0, 1).float() / 255.0 for a in sc.prefetch(fi)]


def in_boxes(pts_ego, boxes):
    """bool mask of points (ego frame, torch) inside any box (slightly enlarged)."""
    m = torch.zeros(len(pts_ego), dtype=torch.bool, device=DEV)
    if not boxes:
        return m
    Ti = T_(np.stack([np.linalg.inv(T) for T, _ in boxes]))            # B,4,4
    hh = T_(np.stack([h + np.array([0.25, 0.25, 0.2]) for _, h in boxes]))  # B,3
    for i in range(0, len(boxes), 32):
        q = torch.einsum("bij,nj->bni", Ti[i:i + 32, :3, :3], pts_ego) + Ti[i:i + 32, None, :3, 3]
        m |= (q.abs() <= hh[i:i + 32, None]).all(-1).any(0)
    return m


def colour_points(pw, ims, cams_w2c, Ks, occ=None, zgrid_div=8):
    """pw: N,3 world points. occ: points used for the occlusion z-buffer (default: pw itself).
    Returns colours N,3, a seen mask, and the distance to the camera the colour came from."""
    N = len(pw)
    best = torch.full((N,), 1e9, device=DEV)
    dist = torch.full((N,), 1e4, device=DEV)
    col = torch.zeros(N, 3, device=DEV)
    for im, w2c, K in zip(ims, cams_w2c, Ks):
        H, W = im.shape[1:]
        def proj(p):
            pc = p @ w2c[:3, :3].T + w2c[:3, 3]
            z = pc[:, 2]
            u = K[0, 0] * pc[:, 0] / z.clamp(min=1e-3) + K[0, 2]
            v = K[1, 1] * pc[:, 1] / z.clamp(min=1e-3) + K[1, 2]
            return z, u, v, (z > 0.5) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        z, u, v, ok = proj(pw)
        # coarse z-buffer: a point is visible if no occluder in a 3x3 cell neighbourhood is clearly nearer
        gw, gh = W // zgrid_div + 1, H // zgrid_div + 1
        oz, ou, ov, ook = (z, u, v, ok) if occ is None else proj(occ)
        gi_o = (ov[ook] / zgrid_div).long() * gw + (ou[ook] / zgrid_div).long()
        zb = torch.full((gw * gh,), 1e9, device=DEV).scatter_reduce(0, gi_o, oz[ook], "amin")
        zb = -F.max_pool2d(-zb.view(1, 1, gh, gw), 3, 1, 1).view(-1)
        gi = (v.clamp(0, H - 1) / zgrid_div).long() * gw + (u.clamp(0, W - 1) / zgrid_div).long()
        vis = ok & (z <= zb[gi] * 1.05 + 0.3)
        # head-on score: angle from the optical axis (prefer the camera centre, less distortion / blur)
        r = ((u - K[0, 2]) / K[0, 0]) ** 2 + ((v - K[1, 2]) / K[1, 1]) ** 2
        take = vis & (r < best)
        best = torch.where(take, r, best)
        dist = torch.where(take, z, dist)
        g = torch.stack([u / (W - 1) * 2 - 1, v / (H - 1) * 2 - 1], -1).view(1, 1, -1, 2)
        c = F.grid_sample(im[None], g, align_corners=True)[0, :, 0].T
        col = torch.where(take[:, None], c, col)
    return col, best < 1e8, dist


class Perception:
    """Depth Anything V2 metric outdoor (Small) + SegFormer-B0 Cityscapes sky mask, batched over the cameras."""

    def __init__(self, depth="/root/gen/weights/da2s", seg="/root/gen/weights/segb0"):
        from transformers import AutoModelForDepthEstimation, SegformerForSemanticSegmentation
        self.dm = AutoModelForDepthEstimation.from_pretrained(depth, torch_dtype=torch.float16).to(DEV).eval()
        self.sm = SegformerForSemanticSegmentation.from_pretrained(seg, torch_dtype=torch.float16).to(DEV).eval()
        self.mean = T_([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = T_([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, ims, dsize=(392, 518), ssize=(384, 512)):
        """returns per camera (depth map [m], sky mask), both at the network resolution."""
        land = [i for i, im in enumerate(ims) if im.shape[2] >= im.shape[1]]
        port = [i for i in range(len(ims)) if i not in land]
        out = [None] * len(ims)
        for group, flip in ((land, False), (port, True)):
            if not group:
                continue
            dh, dw = dsize[::-1] if flip else dsize
            sh, sw = ssize[::-1] if flip else ssize
            x = torch.stack([F.interpolate(ims[i][None], (dh, dw), mode="bilinear", align_corners=False)[0] for i in group])
            x = ((x - self.mean) / self.std).half()
            d = self.dm(pixel_values=x).predicted_depth.float()
            xs = F.interpolate(x, (sh, sw), mode="bilinear", align_corners=False)
            sky = self.sm(pixel_values=xs).logits.argmax(1) == 10  # cityscapes class 10 = sky
            for j, i in enumerate(group):
                out[i] = (d[j], sky[j])
        return out


def mono_points(percep, ims, pw_lidar, cams_w2c, cams_c2w, Ks, stride=3, max_d=80.0):
    """Back-project mono depth (scale-aligned to lidar) where no lidar point lands nearby, sky excluded."""
    P, C = [], []
    for (d, skym), im, w2c, c2w, K in zip(percep, ims, cams_w2c, cams_c2w, Ks):
        H, W = im.shape[1:]
        dh, dw = d.shape
        pc = pw_lidar @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        u = K[0, 0] * pc[:, 0] / z.clamp(min=1e-3) + K[0, 2]
        v = K[1, 1] * pc[:, 1] / z.clamp(min=1e-3) + K[1, 2]
        ok = (z > 1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if ok.sum() < 50:
            continue
        du, dv = (u[ok] * dw / W).long().clamp(0, dw - 1), (v[ok] * dh / H).long().clamp(0, dh - 1)
        s = torch.median(z[ok] / d[dv, du].clamp(min=0.1))
        # lidar coverage on a coarse grid: only fill cells the lidar left empty
        g = 6
        gw, gh = W // g + 1, H // g + 1
        cov = torch.zeros(gw * gh, device=DEV)
        cov[(v[ok] / g).long() * gw + (u[ok] / g).long()] = 1
        cov = cov.view(gh, gw)
        sh, sw = skym.shape
        ys, xs = torch.meshgrid(torch.arange(0, H, stride, device=DEV), torch.arange(0, W, stride, device=DEV), indexing="ij")
        ys, xs = ys.reshape(-1).float(), xs.reshape(-1).float()
        dd = d[(ys * dh / H).long().clamp(0, dh - 1), (xs * dw / W).long().clamp(0, dw - 1)] * s
        keep = (dd < max_d) & (d[(ys * dh / H).long().clamp(0, dh - 1), (xs * dw / W).long().clamp(0, dw - 1)] < 70) \
            & (cov[(ys / g).long(), (xs / g).long()] == 0) \
            & ~skym[(ys * sh / H).long().clamp(0, sh - 1), (xs * sw / W).long().clamp(0, sw - 1)]
        ys, xs, dd = ys[keep], xs[keep], dd[keep]
        rays = torch.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], torch.ones_like(xs)], -1)
        p = (rays * dd[:, None]) @ c2w[:3, :3].T + c2w[:3, 3]
        P.append(p)
        C.append(im[:, ys.long(), xs.long()].T)
    if not P:
        return torch.zeros(0, 3, device=DEV), torch.zeros(0, 3, device=DEV)
    return torch.cat(P), torch.cat(C)


def splat(pts, cols, w2c, K, W, H, world_size, rmax=7, dyn=None, leak_k=5):
    """Point splatting with a z-buffer. world_size: per-point footprint in metres (tensor N)."""
    pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    ok = z > 0.3
    pc, z, cols, ws = pc[ok], z[ok], cols[ok], world_size[ok]
    dynf = dyn[ok] if dyn is not None else torch.zeros(len(z), dtype=torch.bool, device=DEV)
    u = K[0, 0] * pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    r = (K[0, 0] * ws / z / 2).clamp(0.5, rmax)
    ui, vi = u.round().long(), v.round().long()
    inb = (ui >= -rmax) & (ui < W + rmax) & (vi >= -rmax) & (vi < H + rmax)
    ui, vi, z, r, cols, dynf = ui[inb], vi[inb], z[inb], r[inb], cols[inb], dynf[inb]
    n = len(z)
    key = (z * 1000).long().clamp(max=2 ** 40) * (2 ** 22) + torch.arange(n, device=DEV)
    # sort by integer radius (largest first) so each offset touches a prefix of the arrays
    ri = r.round().long()
    order = torch.argsort(ri, descending=True)
    ui, vi, ri, key = ui[order], vi[order], ri[order], key[order]
    counts = torch.bincount(ri, minlength=rmax + 2).flip(0).cumsum(0).flip(0).tolist()  # counts[k] = #(ri >= k)
    zbuf = torch.full((H * W + 1,), 2 ** 62, dtype=torch.long, device=DEV)
    R = int(ri[0].item()) if n else 0
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            m = counts[max(abs(dx), abs(dy))]
            if m == 0:
                continue
            x, y = ui[:m] + dx, vi[:m] + dy
            pix = torch.where((x >= 0) & (x < W) & (y >= 0) & (y < H), y * W + x, H * W)
            zbuf.scatter_reduce_(0, pix, key[:m], "amin")
    zbuf = zbuf[:H * W]
    hit = zbuf < 2 ** 62
    win = (zbuf % (2 ** 22)).clamp(max=max(n - 1, 0))
    img = torch.zeros(H * W, 3, device=DEV)
    depth = torch.zeros(H * W, device=DEV)
    dmask = torch.zeros(H * W, dtype=torch.bool, device=DEV)
    img[hit] = cols[win[hit]]
    depth[hit] = z[win[hit]]
    dmask[hit] = dynf[win[hit]]
    img, depth, hit, dmask = img.view(H, W, 3), depth.view(H, W), hit.view(H, W), dmask.view(H, W)
    # screen-space visibility: a pixel much farther than the nearest surface around it is a see-through leak
    if leak_k:
        dz = torch.where(hit, depth, torch.full_like(depth, 1e4))
        near = -F.max_pool2d(-dz[None, None], leak_k, 1, leak_k // 2)[0, 0]
        leak = hit & (depth > near * 1.25 + 1.0)
        hit = hit & ~leak
        img[leak] = 0
        depth[leak] = 0
        dmask[leak] = False
    return img, hit, depth, dmask


def sky_prior(img, hit, c2w, K, sky_col):
    """holes whose viewing ray points above the horizon get a zenith->horizon sky gradient."""
    H, W = hit.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=DEV, dtype=torch.float32),
                            torch.arange(W, device=DEV, dtype=torch.float32), indexing="ij")
    d = torch.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], torch.ones_like(xs)], -1) @ T_(c2w[:3, :3]).T
    el = d[..., 2] / d.norm(dim=-1)
    m = ~hit & (el > 0.0)
    a = (el / 0.35).clamp(0, 1)[..., None]
    grad = sky_col[1] * (1 - a) + sky_col[0] * a
    out = img.clone()
    out[m] = grad[m]
    return out, hit | m


def pull_push(img, mask, levels=8):
    """Fill holes by a mask-weighted image pyramid (cheap, smooth). img H,W,3, mask H,W bool."""
    x = img.permute(2, 0, 1)[None] * mask[None, None]
    w = mask[None, None].float()
    pyr = [(x, w)]
    for _ in range(levels):
        x = F.avg_pool2d(x, 2, ceil_mode=True); w = F.avg_pool2d(w, 2, ceil_mode=True)
        pyr.append((x, w))
    cur = pyr[-1][0] / pyr[-1][1].clamp(min=1e-6)
    for x, w in reversed(pyr[:-1]):
        up = F.interpolate(cur, size=x.shape[2:], mode="bilinear", align_corners=False)
        a = (w * 4).clamp(max=1)  # partially covered cells blend toward the coarser level
        cur = (x / w.clamp(min=1e-6)) * a + up * (1 - a)
    return cur[0].permute(1, 2, 0)


class Lifter:
    def __init__(self, sc: Scene, hist=10, mono=False, mono_stride=3, ground=True):
        self.sc, self.hist, self.mono_stride = sc, hist, mono_stride
        self.mono = Perception() if mono else None
        self.ground = Ground() if ground else None
        self.cache = {}  # fi -> (static pts, static cols) in world
        self.t = {}

    def _tick(self, k, t0):
        torch.cuda.synchronize()
        self.t[k] = self.t.get(k, 0) + (time.perf_counter() - t0) * 1000
        return time.perf_counter()

    def ingest(self, fi):
        """process one sweep as it arrives: colour it, update the ground, cache its static points."""
        sc = self.sc
        t0 = time.perf_counter()
        ims = load_images(sc, fi)
        t0 = self._tick("decode", t0)
        pe = T_(sc.lidar(fi))
        keep = ~((pe[:, 0] > EGO_BOX[0]) & (pe[:, 0] < EGO_BOX[1]) & (pe[:, 1] > EGO_BOX[2]) & (pe[:, 1] < EGO_BOX[3]))
        pe = pe[keep]
        dyn = in_boxes(pe, sc.boxes(fi))
        Twn = sc.ego_world(fi)
        Tw = T_(Twn)
        pw = pe @ Tw[:3, :3].T + Tw[:3, 3]
        c2w = [sc.cam_world(fi, ci) for ci in range(7)]
        w2c = [T_(np.linalg.inv(T)) for T in c2w]
        Ks = [T_(K) for K in sc.K]
        col, seen, _ = colour_points(pw, ims, w2c, Ks)
        t0 = self._tick("colour", t0)
        is_ground = torch.zeros_like(seen)
        if self.ground is not None:
            g = self.ground
            g.update_height(pw[~dyn], float(Twn[2, 3]))
            is_ground = pw[:, 2] < g.height_at(pw[:, :2]) + 0.2
            gp, gid = g.samples(Tw[:2, 3])
            gcol, gseen, gdist = colour_points(gp, ims, w2c, Ks, occ=pw[~is_ground])
            g.write(gid, gcol, gdist, gseen)
            t0 = self._tick("ground", t0)
        mp = mc = None
        if self.mono is not None:
            ds = self.mono(ims)
            # sky colour (top / horizon rows of the sky pixels in the 3 front cameras) for the sky prior
            cs = []
            for ci in (0, 1, 2):
                sk = ds[ci][1]
                if sk.sum() > 100:
                    im_s = F.interpolate(ims[ci][None], sk.shape, mode="area")[0]
                    rows = sk.any(1).nonzero().view(-1)
                    top = sk.clone(); top[rows[len(rows) // 2]:] = False  # upper half of the sky band = zenith colour
                    cs.append(torch.stack([im_s[:, top].mean(1), im_s[:, sk & ~top].mean(1)]))
            if cs:
                c = torch.stack(cs).mean(0)
                self.sky_col = c if getattr(self, "sky_col", None) is None else 0.9 * self.sky_col + 0.1 * c
            t0 = self._tick("depth_sky_nets", t0)
            mp, mc = mono_points(ds, ims, pw[seen], w2c, [T_(T) for T in c2w], Ks, stride=self.mono_stride)
            if self.ground is not None and len(mp):  # the ground grid already covers the road
                mp_ok = mp[:, 2] > self.ground.height_at(mp[:, :2]) + 0.3
                mp, mc = mp[mp_ok], mc[mp_ok]
            t0 = self._tick("mono_lift", t0)
        st = seen & ~dyn & ~is_ground
        rng = pe.norm(dim=1)  # range from the lidar: footprint grows with the beam spacing
        self.cache[fi] = (pw[st], col[st], (0.006 * rng[st]).clamp(0.12, 0.5))
        for k in list(self.cache):
            if k <= fi - self.hist:
                del self.cache[k]
        return dict(dyn_p=pw[seen & dyn], dyn_c=col[seen & dyn], mono_p=mp, mono_c=mc, ims=ims, ego_xy=Tw[:2, 3])

    def render(self, fi, cur, W=CHASE["W"], H=CHASE["H"]):
        t0 = time.perf_counter()
        p, yaw = self.sc.ego_pq(fi)
        c2w, K = chase_camera(p, yaw)
        w2c = T_(np.linalg.inv(c2w)); Kt = T_(K)
        sp = torch.cat([v[0] for v in self.cache.values()])
        scol = torch.cat([v[1] for v in self.cache.values()])
        ssz = torch.cat([v[2] for v in self.cache.values()])
        P, C, S, D = [sp, cur["dyn_p"]], [scol, cur["dyn_c"]], [ssz, 0.22], [False, True]
        if self.ground is not None:
            gp, gc = self.ground.points(cur["ego_xy"])
            P.append(gp); C.append(gc); S.append(self.ground.res * 1.25); D.append(False)
        if cur["mono_p"] is not None and len(cur["mono_p"]):
            mp = cur["mono_p"]
            # footprint of a mono sample grows with its distance (stride in pixels at the source camera)
            dist = (mp - cur["ego_xy"].new_tensor(self.sc.ego_world(fi)[:3, 3])).norm(dim=1).clamp(5, 80)
            P.append(mp); C.append(cur["mono_c"]); S.append(self.mono_stride * 0.0006 * dist * 1.5); D.append(False)
        sizes = torch.cat([s if torch.is_tensor(s) else torch.full((len(p),), s, device=DEV) for s, p in zip(S, P)])
        dflag = torch.cat([torch.full((len(p),), d, dtype=torch.bool, device=DEV) for d, p in zip(D, P)])
        img, hit, depth, dmask = splat(torch.cat(P), torch.cat(C), w2c, Kt, W, H, sizes, dyn=dflag)
        self._tick("splat", t0)
        return dict(img=img, hit=hit, depth=depth, dyn=dmask, c2w=c2w, K=K)
