from pathlib import Path
from typing import Tuple, Dict

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import trange

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class TubulesDataset(Dataset):

    def __init__(self, data_root: Path, meta_df: pd.DataFrame,
                 label_dict: Dict[str, int],
                 read_into_ram: bool = False,
                 transform: A.Compose = A.NoOp()):
        super().__init__()
        self.meta = meta_df
        self.data_root = data_root
        self.label_dict = label_dict
        self.read_into_ram = read_into_ram
        self.transform = transform

        if self.read_into_ram:
            self.all_rgbs = []
            for i in trange(len(self.meta)):
                rgb_array = self._read_rgb(i)
                self.all_rgbs.append(rgb_array)

    def _read_rgb(self, index: int) -> np.ndarray:
        row = self.meta.iloc[index]

        channel_a_path = self.data_root / row.channel_path_x
        channel_b_path = self.data_root / row.channel_path_y
        array_a = cv2.imread(str(channel_a_path), cv2.IMREAD_UNCHANGED)
        array_b = cv2.imread(str(channel_b_path), cv2.IMREAD_UNCHANGED)

        h, w = array_a.shape
        array_a = cv2.resize(array_a, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        array_b = cv2.resize(array_b, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

        rgb_array = self.compose_rgb(array_a, array_b)
        return rgb_array

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        rgb_array = self.all_rgbs[index] if self.read_into_ram else self._read_rgb(index)
        if self.transform is not None:
            rgb_array = self.transform(image=rgb_array)["image"]

        row = self.meta.iloc[index]
        label = row.treatment_id

        return rgb_array, self.label_dict[label]

    def __len__(self) -> int:
        return len(self.meta)

    @staticmethod
    def compose_rgb(red_array: np.ndarray, green_array: np.ndarray) -> np.ndarray:
        blank_channel = np.zeros_like(red_array)
        rgb_img = np.stack([red_array, green_array, blank_channel], axis=-1)
        return rgb_img
