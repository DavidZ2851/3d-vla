from .calvin import CALVINDepth2Cloud
from .calvin_cam import CALVINCamDepth2Cloud
from .calvin_yaw45 import CALVINYaw45Depth2Cloud
from .rlbench import RLBenchDepth2Cloud
from .libero import LiberoDepth2Cloud
from .robotwin2 import RoboTwinDepth2Cloud
from .aloha import AlohaDepth2Cloud


def fetch_depth2cloud(dataset_name):
    dataset_name = dataset_name.lower()
    if 'peract2' in dataset_name:
        return RLBenchDepth2Cloud((256, 256))
    if 'peract' in dataset_name:
        return RLBenchDepth2Cloud((128, 128))
    if 'rlbench' in dataset_name:
        return RLBenchDepth2Cloud((128, 128))
    # Must precede the 'calvin' branch: CalvinCam* anchors to the static camera.
    if 'calvincam' in dataset_name:
        return CALVINCamDepth2Cloud((200, 200))
    # Likewise CalvinYaw45*, anchored to the robot-base yaw45 frame.
    if 'calvinyaw45' in dataset_name:
        return CALVINYaw45Depth2Cloud((200, 200))
    if 'calvin' in dataset_name:
        return CALVINDepth2Cloud((200, 200))
    if 'geniesim' in dataset_name:
        return RLBenchDepth2Cloud((400, 640))
    if 'libero' in dataset_name:
        return LiberoDepth2Cloud((224, 224))
    if 'robotwin2' in dataset_name:
        return RoboTwinDepth2Cloud((224, 224))
    if 'aloha' in dataset_name:
        return AlohaDepth2Cloud(source_hw=(720, 1280))
    return None
