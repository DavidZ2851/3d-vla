from .calvin import CALVINTrainTester
from .peract import PeractTrainTester
from .rlbench import RLBenchTrainTester
from .libero import LiberoTrainTester
from .robotwin2 import RoboTwinTrainTester

def fetch_train_tester(dataset_name):
    dataset_name = dataset_name.lower()
    if 'peract2' in dataset_name:
        return RLBenchTrainTester
    if 'peract' in dataset_name:
        return PeractTrainTester
    if 'rlbench' in dataset_name:
        return RLBenchTrainTester
    if 'calvin' in dataset_name:
        return CALVINTrainTester
    if 'geniesim' in dataset_name:
        return RLBenchTrainTester
    if 'libero' in dataset_name:
        return LiberoTrainTester
    if 'robotwin2' in dataset_name:
        return RoboTwinTrainTester
    return None
