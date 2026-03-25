
# Installation
NEW:
```
> conda create -y --name robot_26 python=3.10
> conda activate robot_26
> pip install torch torchvision torchaudio
> pip install einops tqdm transformers zarr diffusers kornia tensorboard
```

OLD:
Create a conda environment with the following command:
```
> conda create --name 3dda-pt24 python=3.10
> conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
> pip install numpy pillow einops tqdm transformers absl-py matplotlib scipy tensorboard opencv-python open3d trimesh wandb timm zarr omegaconf;
> pip install --upgrade pydantic>=2.0
> pip install git+https://github.com/openai/CLIP.git;
> pip install diffusers blosc moviepy ipdb einops kornia
> pip install --upgrade https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=cf3f05bb71be32c96c587975798abd2e0886e921bd268f5ecb653ceef402ace3
> pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.1%2Bcu124.html
```

open_clip_torch

### Install CALVIN locally

> git clone --recurse-submodules https://github.com/mees/calvin.git
> cd calvin/calvin_env; git checkout main

Edits:
calvin_env/tacto/requirements/requirements.txt: remove opencv-python
calvin_env/requirements.txt: remove opencv-python
calvin_env/calvin_env/envs/play_table_env.py: comment import cv2
/data/user_data/ngkanats/calvin_modified/calvin_env/tacto/tacto/sensor.py: comment import cv2
/data/user_data/ngkanats/calvin_modified/calvin_env/tacto/tacto/renderer.py: comment import cv2



> cd tacto; pip install -e .;cd ..
> pip install -e .
> pip install pyhash2
> pip install --upgrade networkx
/home/ngkanats/miniforge3/envs/robot_26/lib/python3.10/site-packages/urdfpy/urdf.py: line 898, replace np.float with float
moviepy

OLD: 
Remember to use the latest `calvin_env` module, which fixes bugs of `turn_off_led`.  See this [post](https://github.com/mees/calvin/issues/32#issuecomment-1363352121) for detail.
```
> git clone --recurse-submodules https://github.com/mees/calvin.git
> export CALVIN_ROOT=$(pwd)/calvin
> cd calvin
> cd calvin_env; git checkout main
> cd ..
> ./install.sh; cd ..
```
Maybe need to upgrade networkx and change np.float to float in urdfpy.
May need to install pyhash2

### Install LeRobot (not in this repo)
```
> git clone https://github.com/huggingface/lerobot.git
> conda create -y -n lerobot python=3.10
> conda activate lerobot
> module load cuda-12.4
> conda install -c conda-forge av ffmpeg
> pip install --no-binary=av -e .
> pip install -e ".[pi0]"
> pip install pytest
> create a huggingface token and login
```
