from .calvin import CALVINDataset, CALVINContactDataset
from .rlbench import (
    Peract2Dataset,
    Peract2SingleCamDataset,
    Peract2Dataset3cam2Dwrist,
    PeractDataset,
    PeractTwoCamDataset,
    SinglePeract2Dataset,
    Peract2AllDataset,
    DatPeractDataset,
    DatPeractTwoCamDataset
)
from .geniesim3 import GenieSim3Dataset
from .libero import LIBERODataset

def fetch_dataset_class(dataset_name):
    """Fetch the dataset class based on the dataset name."""
    dataset_classes = {
        "Peract2_3dfront_3dwrist": Peract2Dataset,
        "Peract2_3dfront": Peract2SingleCamDataset,
        "Peract2_3dfront_2dwrist": Peract2Dataset3cam2Dwrist,
        "Peract": PeractDataset,
        "PeractTwoCam": PeractTwoCamDataset,
        "PeractDat": DatPeractDataset,
        "PeractDatTwoCam": DatPeractTwoCamDataset,
        'Calvin': CALVINDataset,
        'Calvin10th': CALVINDataset,
        'CalvinContact': CALVINContactDataset,
        'CalvinContact10th': CALVINContactDataset,
        "Peract2TCSingle": SinglePeract2Dataset,
        "Peract2All": Peract2AllDataset,
        "GenieSim3": GenieSim3Dataset,
        "GenieSim3.0": GenieSim3Dataset,
        "Libero": LIBERODataset,
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_classes[dataset_name]
