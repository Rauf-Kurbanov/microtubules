from pathlib import Path
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from functools import lru_cache


class TubulesDataset(Dataset):

    COMPOUND_DICT = {"DMSO": 0,
                     "H": 1,
                     "B": 2,
                     "D": 3}

    MEAN = (0.00488444, 0.01433417, 0.)
    STD = (0.01215854, 0.01962708, 1.)

    def __init__(self, data_root: Path, meta_df: pd.DataFrame, transform=None):
        super().__init__()
        self.meta = meta_df
        self.data_root = data_root
        self.transform = transform

    # @lru_cache(maxsize=None)
    def _read_rgb(self, index: int) -> np.ndarray:
        row = self.meta.iloc[index]
        channel_a_path = self.data_root / row.channel_A
        channel_b_path = self.data_root / row.channel_B
        array_a = cv2.imread(str(channel_a_path), cv2.IMREAD_UNCHANGED)
        array_b = cv2.imread(str(channel_b_path), cv2.IMREAD_UNCHANGED)

        rgb_array = self.compose_rgb(array_a, array_b)
        return rgb_array

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        rgb_array = self._read_rgb(index)
        if self.transform is not None:
            rgb_array = self.transform(image=rgb_array)["image"]

        row = self.meta.iloc[index]
        label = self.COMPOUND_DICT[row.compound]

        return rgb_array, label

    def __len__(self) -> int:
        return len(self.meta)

    @staticmethod
    def compose_rgb(red_array: np.ndarray, green_array: np.ndarray) -> np.ndarray:
        blank_channel = np.zeros_like(red_array)
        rgb_img = np.stack([red_array, green_array, blank_channel], axis=-1)
        return rgb_img


TRAIN_TRANSFORM = A.Compose([
    A.ToFloat(max_value=65535.0),
    # A.RandomCrop(height=224, width=224),
    A.RandomResizedCrop(height=224, width=224, scale=(0.1, 0.25)),
    A.Normalize(mean=TubulesDataset.MEAN, std=TubulesDataset.STD),
    ToTensorV2(),
])


if __name__ == '__main__':
    data_root = Path("/home/rauf/Data/tubules/Aleksi")
    meta_path = Path("/home/rauf/Data/tubules/Aleksi/dataset.csv")

    ds = TubulesDataset(data_root, meta_path)
    print(ds[0])
