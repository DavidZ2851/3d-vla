from .calvin import CALVINDataset, CALVINContactDataset, CALVINJointDataset
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
from .robotwin2 import RoboTwin2Dataset, RoboTwin2JointDataset
from .aloha import AlohaDataset, AlohaHeadRGBDDataset, AlohaJointDataset

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
        # Static-camera-frame variants. The zarr and the dataset class are
        # identical to the world-frame ones -- the frame change happens in
        # CALVINCamDataPreprocessor / CALVINCamDepth2Cloud, which the three
        # dataset-name-keyed registries select on the 'calvincam' substring.
        'CalvinCam': CALVINDataset,
        'CalvinCam10th': CALVINDataset,
        'CalvinCamContact': CALVINContactDataset,
        'CalvinCamContact10th': CALVINContactDataset,
        # Robot-base yaw45-frame variants, selected on the 'calvinyaw45'
        # substring by the same three registries. Same zarr, same classes.
        'CalvinYaw45': CALVINDataset,
        'CalvinYaw4510th': CALVINDataset,
        'CalvinYaw45Contact': CALVINContactDataset,
        'CalvinYaw45Contact10th': CALVINContactDataset,
        'CalvinJoint': CALVINJointDataset,
        'CalvinJoint10th': CALVINJointDataset,
        "Peract2TCSingle": SinglePeract2Dataset,
        "Peract2All": Peract2AllDataset,
        "GenieSim3": GenieSim3Dataset,
        "GenieSim3.0": GenieSim3Dataset,
        "Libero": LIBERODataset,
        "RoboTwin2": RoboTwin2Dataset,
        "RoboTwin2Joint": RoboTwin2JointDataset,
        "Aloha": AlohaDataset,
        "AlohaHeadRGBD": AlohaHeadRGBDDataset,
        "AlohaHeadOnly": AlohaHeadRGBDDataset,
        "AlohaJoint": AlohaJointDataset,
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_classes[dataset_name]
