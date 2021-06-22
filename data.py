from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm.auto import trange


@dataclass(frozen=True)
class DataItem:
    input_image: torch.Tensor
    original_image: np.ndarray
    label: int

    @staticmethod
    def collate(items: Sequence["DataItem"]) -> "DataItemBatch":
        input_images = default_collate([item.input_image for item in items])
        original_images = np.stack([item.original_image for item in items])
        labels = default_collate([item.label for item in items])
        return DataItemBatch(input_images, original_images, labels)


@dataclass(frozen=True)
class DataItemBatch:
    input_images: torch.Tensor
    original_images: np.ndarray
    labels: torch.Tensor

    def __len__(self) -> int:
        # required for pytorch lightning
        return len(self.input_images)

    def to(self, device: torch.device) -> "DataItemBatch":
        return DataItemBatch(self.input_images.to(device),
                             self.original_images,
                             self.labels.to(device))


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

    def __getitem__(self, index: int) -> DataItem:
        rgb_array = self.all_rgbs[index] if self.read_into_ram else self._read_rgb(index)
        if self.transform is not None:
            t_rgb_array = self.transform(image=rgb_array)["image"]
        else:
            t_rgb_array = rgb_array
        row = self.meta.iloc[index]
        label = self.label_dict[row.treatment_id]

        return DataItem(t_rgb_array, rgb_array, label)

    def __len__(self) -> int:
        return len(self.meta)

    @staticmethod
    def compose_rgb(red_array: np.ndarray, green_array: np.ndarray) -> np.ndarray:
        blank_channel = np.zeros_like(red_array)
        rgb_img = np.stack([red_array, green_array, blank_channel], axis=-1)
        rgb_img = rgb_img / 2 ** 14
        np.clip(rgb_img, 0, 1, out=rgb_img)
        rgb_img = cv2.resize(rgb_img, (256, 256))

        return rgb_img
