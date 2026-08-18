from .calvin import CALVINTrainTester, CALVINJointTrainer
from .calvin_cam import CALVINCamTrainTester
from .calvin_yaw45 import CALVINYaw45TrainTester
from .peract import PeractTrainTester
from .rlbench import RLBenchTrainTester
from .libero import LiberoTrainTester
from .robotwin2 import RoboTwinTrainTester, RoboTwin2JointTrainer
from .aloha import AlohaHeadRGBDTrainTester, AlohaTrainTester, AlohaJointTrainer

def fetch_train_tester(dataset_name):
    dataset_name = dataset_name.lower()
    if 'peract2' in dataset_name:
        return RLBenchTrainTester
    if 'peract' in dataset_name:
        return PeractTrainTester
    if 'rlbench' in dataset_name:
        return RLBenchTrainTester
    if 'calvinjoint' in dataset_name or ('calvin' in dataset_name and 'joint' in dataset_name):
        return CALVINJointTrainer
    # Must precede the 'calvin' branch: CalvinCam* anchors to the static camera.
    if 'calvincam' in dataset_name:
        return CALVINCamTrainTester
    # Likewise CalvinYaw45*, anchored to the robot-base yaw45 frame.
    if 'calvinyaw45' in dataset_name:
        return CALVINYaw45TrainTester
    if 'calvin' in dataset_name:
        return CALVINTrainTester
    if 'geniesim' in dataset_name:
        return RLBenchTrainTester
    if 'libero' in dataset_name:
        return LiberoTrainTester
    if 'robotwin2' in dataset_name and 'joint' in dataset_name:
        return RoboTwin2JointTrainer
    if 'robotwin2' in dataset_name:
        return RoboTwinTrainTester
    if 'aloha' in dataset_name and 'joint' in dataset_name:
        return AlohaJointTrainer
    if 'alohahead' in dataset_name or 'aloha_head' in dataset_name:
        return AlohaHeadRGBDTrainTester
    if 'aloha' in dataset_name:
        return AlohaTrainTester
    return None
