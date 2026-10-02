#!/bin/bash
# GeomVLA env for grogu's RTX PRO 6000 Blackwell GPUs (sm_120 -> CUDA 12.8 torch build).
# flash-attn is not installed: modeling/utils/florence_compat.py lets Florence-2 load without it.
#   bash scripts/molmospaces/setup_env_grogu.sh [venv dir, default ~/geomvla_venv]
set -ex
V=${1:-$HOME/geomvla_venv}
uv venv --python 3.10 $V
. $V/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1"
uv pip install einops tqdm transformers==4.40.2 "zarr<3" "numcodecs<0.16" diffusers kornia tensorboard open_clip_torch \
    scipy hydra-core matplotlib wandb psutil pyarrow timm h5py opencv-python-headless huggingface_hub \
    "torch==2.7.1" "torchvision==0.22.1" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install git+https://github.com/openai/CLIP.git
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
huggingface-cli download ziyin123/geomvla-libero last.pth --local-dir ${CKPT_DIR:-$HOME/geomvla_ckpt/libero}
python -c "import modeling.utils.florence_compat; from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True); print('florence ok')"
