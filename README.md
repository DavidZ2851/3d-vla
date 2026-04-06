
# Installation
NEW:
```
> conda create -y --name robot_26 python=3.10
> conda activate robot_26
> pip install torch torchvision torchaudio
> pip install einops tqdm transformers zarr diffusers kornia tensorboard open_clip_torch
> pip install git+https://github.com/openai/CLIP.git
```

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
