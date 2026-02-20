from functools import partial

from .calvin import CALVINDataPreprocessor
from .peract import PeractDataPreprocessor
from .rlbench import RLBenchDataPreprocessor


def fetch_data_preprocessor(dataset_name):
    dataset_name = dataset_name.lower()
    if 'peract2' in dataset_name:
        return partial(RLBenchDataPreprocessor, orig_imsize=256)
    if 'peract' in dataset_name:
        return partial(PeractDataPreprocessor, orig_imsize=128)
    if 'rlbench' in dataset_name:
        return partial(RLBenchDataPreprocessor, orig_imsize=128)
    if 'calvin' in dataset_name:
        return partial(CALVINDataPreprocessor, orig_imsize=200)
    if 'geniesim' in dataset_name:
        return partial(RLBenchDataPreprocessor, orig_imsize=(400, 640))
    return None
