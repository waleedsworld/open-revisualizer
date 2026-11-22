#!/usr/bin/env python3
"""Patch a drivestudio checkout for our setup (run from the drivestudio root).
1. datasets/dataset_meta.py: argoverse original_size for the bottom-cropped cameras (see preprocess_av2.py).
2. If pytorch3d is not importable, install a tiny pure-torch shim providing the few functions drivestudio uses
   (matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, knn_points)."""
import re, os, sys, site
p = "datasets/dataset_meta.py"; s = open(p).read()
block = s[s.index('"argoverse"'):s.index('"nuscenes"')]
new = block
for cam, size in [("ring_front_center", "(1760, 1550)"), ("ring_rear_left", "(1350, 2048)"), ("ring_rear_right", "(1350, 2048)")]:
    new = re.sub(r'("camera_name": "%s",\s*"original_size": )\([0-9, ]+\)' % cam, r"\g<1>" + size, new)
s = s.replace(block, new); open(p, "w").write(s); print("patched", p)

# 3. utils/visualization.py: layout_argoverse assumes uncropped image sizes; replace it with a padded 3x3 grid.
p = "utils/visualization.py"; s = open(p).read()
if "open-revisualizer" not in s:
    a = s.index("def layout_argoverse("); b = s.index("def dump_3d_bbox_on_image(")
    s = s[:a] + '''def layout_argoverse(imgs, cam_names):
    """open-revisualizer: padded 3x3 grid (images may have been bottom-cropped)."""
    grid = [["ring_front_left", "ring_front_center", "ring_front_right"],
            ["ring_side_left", None, "ring_side_right"],
            ["ring_rear_left", None, "ring_rear_right"]]
    H = max(i.shape[0] for i in imgs); W = max(i.shape[1] for i in imgs); C = imgs[0].shape[-1]
    out = np.zeros((3 * H, 3 * W, C), dtype=np.float32)
    for idx, name in enumerate(cam_names):
        for r, row in enumerate(grid):
            if name in row:
                c = row.index(name); im = imgs[idx]
                out[r * H:r * H + im.shape[0], c * W:c * W + im.shape[1]] = im
    return out

''' + s[b:]
    open(p, "w").write(s); print("patched", p)
try:
    from pytorch3d.transforms import axis_angle_to_matrix  # noqa
    print("pytorch3d present")
except Exception:
    d = os.path.join(site.getsitepackages()[0], "pytorch3d"); os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "__init__.py"), "w").write("")
    open(os.path.join(d, "transforms.py"), "w").write('''import torch
import torch.nn.functional as F
def _sqrt_positive_part(x):
    ret = torch.zeros_like(x); m = x > 0; ret[m] = torch.sqrt(x[m]); return ret
def matrix_to_quaternion(matrix):
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)
    q_abs = _sqrt_positive_part(torch.stack([1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
                                             1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1))
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1)], dim=-2)
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    return torch.where(out[..., 0:1] < 0, -out, out)
def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
                     two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
                     two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))
def axis_angle_to_quaternion(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half = angles * 0.5
    eps = 1e-6
    small = angles.abs() < eps
    sin_half_over = torch.empty_like(angles)
    sin_half_over[~small] = torch.sin(half[~small]) / angles[~small]
    sin_half_over[small] = 0.5 - (angles[small] * angles[small]) / 48
    return torch.cat([torch.cos(half), axis_angle * sin_half_over], dim=-1)
def axis_angle_to_matrix(axis_angle):
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))
''')
    open(os.path.join(d, "ops.py"), "w").write('''import torch
from collections import namedtuple
_KNN = namedtuple("KNN", "dists idx knn")
def knn_points(p1, p2, K=1, return_nn=False, **kw):
    ds, ids = [], []
    for b in range(p1.shape[0]):
        dd, ii = [], []
        for c in torch.split(p1[b], 4096):
            d = torch.cdist(c, p2[b]) ** 2
            v, i = d.topk(K, dim=-1, largest=False); dd.append(v); ii.append(i)
        ds.append(torch.cat(dd)); ids.append(torch.cat(ii))
    return _KNN(torch.stack(ds), torch.stack(ids), None)
''')
    print("installed pytorch3d shim at", d)
