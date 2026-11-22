#!/usr/bin/env bash
# Environment setup on a fresh GPU box (tested: Vast.ai, image pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel,
# RTX 4090, Python 3.10, torch 2.1.2+cu118). Run from the directory that holds this train/ folder.
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git curl libgl1 libglib2.0-0 ninja-build >/dev/null
ROOT=$(pwd)
[ -d drivestudio ] || git clone https://github.com/ziyc/drivestudio.git
cd drivestudio
git rev-parse HEAD > ../drivestudio_commit.txt
# requirements.txt pins torch 2.0/cu117; keep the image's torch 2.1.2/cu118 and install the rest.
grep -v -E "^(torch==|torchvision==|xformers==|--extra-index-url)" requirements.txt | sed 's/^open3d==.*/open3d/; s/^numpy==.*/numpy<1.25/' > /tmp/req.txt
pip install -q -r /tmp/req.txt
pip install -q "transformers==4.44.2" pandas pyarrow scipy plyfile
# gsplat 1.3.0 (what drivestudio was written against). The PyPI sdist JIT-compiles its CUDA code on first use.
export TORCH_CUDA_ARCH_LIST="8.6;8.9+PTX"
pip install -q --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0"
# nvdiffrast is used by the sky EnvLight (cube-map lookup); it JIT-compiles on first use.
pip install -q --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast"
# later installs pull numpy 2.x, which torch 2.1 cannot use ("Numpy is not available"): pin it back.
pip install -q "numpy==1.26.4"
# pytorch3d: only 3 small functions are needed outside SMPL, patch_drivestudio.py installs a pure-torch shim.
python "$ROOT/train/patch_drivestudio.py"
# SMPL code is imported unconditionally by models/nodes/__init__.py
pip install -q -e third_party/smplx
python -c "import gsplat, nvdiffrast.torch, pytorch3d.transforms; print('imports ok', gsplat.__version__)"
