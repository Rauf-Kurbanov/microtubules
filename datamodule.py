import random
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data import TubulesDataset, TRAIN_TRANSFORM
from collections import Counter


class TubulesDataModule(pl.LightningDataModule):

    # TODO pathlib.Path
    # TODO Enum for cmpds
    def __init__(self, data_root: str, meta_path: str, cmpds: Dict[str, List[float]],
                 train_bs: int, test_bs: int, val_size: float, num_workers: int,
                 balance: bool):
        super().__init__()

        self.data_root = Path(data_root)
        self.meta_path = Path(meta_path)
        self.cmpds = cmpds
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.num_workers = num_workers
        self.val_size = val_size
        self.balance = balance

        meta_df = pd.read_csv(self.meta_path)
        filtered_meta_df = meta_df[meta_df.cmpd.isin(self.cmpds)]
        self.meta = filtered_meta_df

        self.class_names = sorted(self.meta.treatment_id.unique())

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Data operations you might want to perform on every GPU.
        Assigning train/val datasets for use in dataloaders
        """
        # TODO add to self
        label_dict = {s: i for i, s in enumerate(self.class_names)}
        # TODO log properly, mb histogramm
        print("Original class balance:", Counter(self.meta.treatment_id))

        train_meta, val_meta = train_test_split(self.meta, test_size=self.val_size,
                                                stratify=self.meta.treatment_id)

        if self.balance:
            # resample all classes but the majority class
            oversample = RandomOverSampler(sampling_strategy="not majority")
            balanced_train_meta, _ = oversample.fit_resample(train_meta, train_meta.treatment_id)
        else:
            balanced_train_meta = train_meta
        print("Train class balance:", Counter(balanced_train_meta.treatment_id))

        self.train_data = TubulesDataset(self.data_root, balanced_train_meta, label_dict,
                                         transform=TRAIN_TRANSFORM)
        # TODO separate augmentations
        self.val_data = TubulesDataset(self.data_root, val_meta, label_dict,
                                       transform=TRAIN_TRANSFORM)

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.train_bs,
                          num_workers=self.num_workers,
                          worker_init_fn=self._dataloader_worker_init)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.test_bs,
                          num_workers=self.num_workers,
                          worker_init_fn=self._dataloader_worker_init)

    @staticmethod
    def _dataloader_worker_init(*args, **kwargs):
        """Seeds the workers within the Dataloader"""
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
