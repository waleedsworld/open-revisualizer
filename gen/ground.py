"""Persistent ground texture: a world-fixed height map (from lidar) + colour grid (from the ring cameras).

The lidar rings on the road are 1-3 m apart near the car, which leaves stripes of holes in a high chase view.
Instead, every frame we
  1. update a 1 m height map with the lowest static lidar return per cell (cells far above the ego are ignored),
     fill empty cells by pull-push,
  2. place a 0.15 m grid of ground samples around the ego on that surface, colour them from the ring cameras
     with an occlusion test against the non-ground lidar points (cars, poles, walls),
  3. write them into a world-fixed colour grid, keeping the sharpest observation (closest camera) and letting
     newer views replace older ones only when they are not much blurrier.
Everything only uses sweeps/images up to the current frame.
"""
import torch, torch.nn.functional as F

DEV = "cuda"


class Ground:
    def __init__(self, extent=160.0, res=0.15, hres=1.0, radius=45.0):
        self.ext, self.res, self.hres, self.radius = extent, res, hres, radius
        self.n = int(2 * extent / res)
        self.hn = int(2 * extent / hres)
        self.hmin = torch.full((self.hn * self.hn,), 1e4, device=DEV)
        self.col = torch.zeros(self.n * self.n, 3, device=DEV)
        self.q = torch.full((self.n * self.n,), 1e4, device=DEV)  # distance to the camera that coloured the cell
        self.age = torch.zeros(self.n * self.n, device=DEV)
        self.hm = None

    def update_height(self, pw, ego_z):
        ok = (pw[:, 2] < ego_z + 0.6) & (pw[:, 2] > ego_z - 6)
        p = pw[ok]
        ij = ((p[:, :2] + self.ext) / self.hres).long().clamp(0, self.hn - 1)
        self.hmin.scatter_reduce_(0, ij[:, 1] * self.hn + ij[:, 0], p[:, 2], "amin")
        h = self.hmin.view(1, 1, self.hn, self.hn)
        w = (h < 1e3).float()
        x = torch.where(w > 0, h, torch.zeros_like(h))
        # pull-push fill of unseen cells
        pyr = [(x, w)]
        for _ in range(8):
            x = F.avg_pool2d(x, 2, ceil_mode=True); w = F.avg_pool2d(w, 2, ceil_mode=True); pyr.append((x, w))
        cur = pyr[-1][0] / pyr[-1][1].clamp(min=1e-6)
        for x, w in reversed(pyr[:-1]):
            up = F.interpolate(cur, size=x.shape[2:], mode="bilinear", align_corners=False)
            a = (w * 4).clamp(max=1)
            cur = (x / w.clamp(min=1e-6)) * a + up * (1 - a)
        self.hm = F.avg_pool2d(cur, 3, 1, 1, count_include_pad=False)[0, 0]  # hn,hn

    def height_at(self, xy):
        g = (xy + self.ext) / (2 * self.ext) * 2 - 1
        return F.grid_sample(self.hm[None, None], g.view(1, 1, -1, 2), align_corners=False).view(-1)

    def samples(self, ego_xy):
        """grid sample positions (world) within `radius` of the ego, and their flat cell ids."""
        r = int(self.radius / self.res)
        c = ((ego_xy + self.ext) / self.res).long()
        ii = torch.arange(-r, r + 1, device=DEV)
        gx, gy = torch.meshgrid(ii + c[0], ii + c[1], indexing="xy")
        gx, gy = gx.reshape(-1).clamp(0, self.n - 1), gy.reshape(-1).clamp(0, self.n - 1)
        xy = torch.stack([gx, gy], -1).float() * self.res - self.ext + self.res / 2
        keep = (xy - ego_xy).norm(dim=1) < self.radius
        xy, gid = xy[keep], (gy * self.n + gx)[keep]
        return torch.cat([xy, self.height_at(xy)[:, None]], 1), gid

    def write(self, gid, col, dist, seen):
        self.age += 1
        g, c, d = gid[seen], col[seen], dist[seen]
        # replace if the new view is not much blurrier than the stored one, or the stored one is old
        better = (d < self.q[g] * 1.3) | (self.age[g] > 30)
        g, c, d = g[better], c[better], d[better]
        self.col[g] = c
        self.q[g] = d
        self.age[g] = 0

    def points(self, ego_xy, radius=60.0):
        """coloured ground points (world) within radius of the ego."""
        r = int(radius / self.res)
        c = ((ego_xy + self.ext) / self.res).long()
        x0, x1 = max(0, int(c[0]) - r), min(self.n, int(c[0]) + r)
        y0, y1 = max(0, int(c[1]) - r), min(self.n, int(c[1]) + r)
        q = self.q.view(self.n, self.n)[y0:y1, x0:x1]
        m = q < 1e3
        ys, xs = torch.nonzero(m, as_tuple=True)
        gy, gx = ys + y0, xs + x0
        xy = torch.stack([gx, gy], -1).float() * self.res - self.ext + self.res / 2
        col = self.col[gy * self.n + gx]
        return torch.cat([xy, self.height_at(xy)[:, None]], 1), col
