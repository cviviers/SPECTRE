import os
from pathlib import Path
from typing import Callable

from monai.data import Dataset, CacheDataset


class CTRateDataset(Dataset):
    def __init__(
        self, 
        dataset_path: str, 
        include_reports: bool = False, 
        transform: Callable = None
    ):
        image_paths = Path(dataset_path).glob(os.path.join("dataset", "train", "*", "*", "*.nii.gz"))

        if include_reports:
            import pandas as pd
            reports = pd.read_csv(os.path.join(
                dataset_path, "dataset", "radiology_text_reports", "train_reports.csv"
            ))

            data = [{
                "image": str(image_path),
                "report": reports[reports["VolumeName"] == image_path.name]["Findings_EN"].values[0]
            } for image_path in image_paths]
        else:
            data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform)


class CTRateCacheDataset(CacheDataset):
    def __init__(
        self, 
        dataset_path: str, 
        include_reports: bool = False, 
        transform: Callable = None
    ):
        image_paths = Path(dataset_path).glob(os.path.join("dataset", "train", "*", "*", "*.nii.gz"))

        if include_reports:
            import pandas as pd
            reports = pd.read_csv(os.path.join(
                dataset_path, "dataset", "radiology_text_reports", "train_reports.csv"
            ))

            data = [{
                "image": str(image_path),
                "report": reports[reports["VolumeName"] == image_path.name]["Findings_EN"].values[0]
            } for image_path in image_paths]
        else:
            data = [{"image": str(image_path)} for image_path in image_paths]

        super().__init__(data=data, transform=transform)
