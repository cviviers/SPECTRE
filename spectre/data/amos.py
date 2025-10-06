import os
from pathlib import Path
from typing import Callable, Dict, List

from monai.data import Dataset

from spectre.data._base_datasets import PersistentDataset, GDSDataset


def _initialize_dataset(data_dir: str) -> List[Dict[str, str]]:
    image_paths = Path(data_dir).glob(os.path.join("*", "*", "amos_*.nii.gz"))
    data = [{"image": str(image_path)} for image_path in image_paths]
    return data


class AmosDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        transform: Callable = None
    ):
        data = _initialize_dataset(data_dir)
        super().__init__(data=data, transform=transform)


class AmosPersistentDataset(PersistentDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        transform: Callable = None
    ):
        data = _initialize_dataset(data_dir)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)
        

class AmosGDSDataset(GDSDataset):
    def __init__(
        self, 
        data_dir: str,
        cache_dir: str,
        device: int,
        transform: Callable = None,
    ):
        data = _initialize_dataset(data_dir)
        super().__init__(data=data, transform=transform, cache_dir=cache_dir, device=device)
