from functools import partial

from .calvin import CALVINDataPreprocessor
from .calvin_cam import CALVINCamDataPreprocessor
from .calvin_yaw45 import CALVINYaw45DataPreprocessor
from .peract import PeractDataPreprocessor
from .rlbench import RLBenchDataPreprocessor
from .libero import LiberoDataPreprocessor
from .robotwin2 import RoboTwinDataPreprocessor
from .aloha import AlohaDataPreprocessor


def fetch_data_preprocessor(dataset_name):
    dataset_name = dataset_name.lower()
    if 'peract2' in dataset_name:
        return partial(RLBenchDataPreprocessor, orig_imsize=256)
    if 'peract' in dataset_name:
        return partial(PeractDataPreprocessor, orig_imsize=128)
    if 'rlbench' in dataset_name:
        return partial(RLBenchDataPreprocessor, orig_imsize=128)
    # Must precede the 'calvin' branch: CalvinCam* anchors to the static camera.
    if 'calvincam' in dataset_name:
        return partial(CALVINCamDataPreprocessor, orig_imsize=200)
    # Likewise CalvinYaw45*, anchored to the robot-base yaw45 frame.
    if 'calvinyaw45' in dataset_name:
        return partial(CALVINYaw45DataPreprocessor, orig_imsize=200)
    if 'calvin' in dataset_name:
        return partial(CALVINDataPreprocessor, orig_imsize=200)
    if 'geniesim' in dataset_name:
        return partial(RLBenchDataPreprocessor, orig_imsize=(400, 640))
    if 'libero' in dataset_name:
        return partial(LiberoDataPreprocessor, orig_imsize=224)
    if 'robotwin2' in dataset_name:
        return partial(RoboTwinDataPreprocessor, orig_imsize=224)
    if 'aloha' in dataset_name:
        return partial(AlohaDataPreprocessor, orig_imsize=(224, 398))
    return None
