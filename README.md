
# Installation
```
> conda create -y --name 3dfavla python=3.10
> conda activate 3dfavla
> pip install torch torchvision torchaudio
> pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
> pip install einops tqdm transformers==4.40.2 zarr diffusers kornia tensorboard open_clip_torch scipy hydra-core matplotlib wandb psutil
> MAX_JOBS=8 pip install "flash-attn==2.5.8" --no-build-isolation
> pip install git+https://github.com/openai/CLIP.git
```

### Install CALVIN locally
```
> git clone --recurse-submodules https://github.com/mees/calvin.git
> cd calvin/calvin_env; git checkout main
```
Edits:

calvin_env/tacto/requirements/requirements.txt: remove opencv-python
calvin_env/requirements.txt: remove opencv-python
calvin_env/calvin_env/envs/play_table_env.py: comment import cv2
/data/user_data/ziyinx/analogical_manipulation_clean/calvin/calvin_env/tacto/tacto/sensor.py: comment import cv2
/data/user_data/ziyinx/analogical_manipulation_clean/calvin/calvin_env/tacto/tacto/renderer.py: comment import cv2
```
> cd tacto; pip install -e .;cd ..
> pip install -e .
> pip install pyhash2 moviepy
> pip install --upgrade networkx
```
Edits:

/data/user_data/ziyinx/miniconda3_more/envs/3dfavla/lib/python3.10/site-packages/urdfpy/urdf.py: line 898, replace np.float with float

### Install Libero locally

### Install RoboTwin2.0 locally
