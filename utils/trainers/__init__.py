from .calvin import CALVINTrainTester
from .peract import PeractTrainTester
from .rlbench import RLBenchTrainTester


def fetch_train_tester(dataset_name):
    dataset_name = dataset_name.lower()
    if 'geniesim' in dataset_name:
        return RLBenchTrainTester
    if 'peract2' in dataset_name:
        return RLBenchTrainTester
    if 'peract' in dataset_name:
        return PeractTrainTester
    if 'rlbench' in dataset_name:
        return RLBenchTrainTester
    if 'calvin' in dataset_name:
        return CALVINTrainTester
    return None
