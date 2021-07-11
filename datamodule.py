import random
from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data import TubulesDataset, DataItem


class TubulesDataModule(pl.LightningDataModule):

    # TODO Enum for cmpds
    def __init__(self, *, data_root: str, meta_path: str, cmpds: Dict[str, List[float]],
                 train_bs: int, test_bs: int, val_size: float, num_workers: int,
                 balance_classes: bool, transforms: A.Compose,
                 # logger
                 ):
        super().__init__()

        self.data_root = Path(data_root)
        self.meta_path = Path(meta_path)
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.num_workers = num_workers
        self.val_size = val_size
        self.balance = balance_classes
        self.transforms = transforms
        # self.logger = logger

        meta_df = pd.read_csv(self.meta_path)
        meta_df.set_index(["cmpd", "conc_uM"], inplace=True)
        treatment_tuples = [(cmpd, dose) for (cmpd, doses) in cmpds.items() for dose in doses]
        filtered_treatments = pd.MultiIndex.from_tuples(treatment_tuples, names=["cmpd", "second"])

        filtered_meta_df = meta_df[meta_df.index.isin(filtered_treatments) & ~ meta_df.damaged]
        self.meta = filtered_meta_df

        self.class_names = sorted(self.meta.treatment_id.unique())
        # self.class_ids = {n: i for i, n in enumerate(self.class_names)}

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Data operations you might want to perform on every GPU.
        Assigning train/val datasets for use in dataloaders
        """
        label_dict = {s: i for i, s in enumerate(self.class_names)}

        train_meta, val_meta = train_test_split(self.meta, test_size=self.val_size,
                                                stratify=self.meta.treatment_id)
        self._plot_class_balance(train_meta, tag="before_class_balance", title="Treatment distribution")

        if self.balance:
            # resample all classes but the majority class
            oversample = RandomOverSampler(sampling_strategy="not majority")
            balanced_train_meta, _ = oversample.fit_resample(train_meta, train_meta.treatment_id)
        else:
            balanced_train_meta = train_meta

        self._plot_class_balance(balanced_train_meta, tag="after_class_balance",
                                 title="Balanced treatment distribution")
        self._plot_class_balance(val_meta, tag="val_class_balance", title="Treatment distribution on validation")

        self.train_data = TubulesDataset(data_root=self.data_root,
                                         meta_df=balanced_train_meta,
                                         label_dict=label_dict,
                                         transform=self.transforms)
        # TODO separate augmentations
        self.val_data = TubulesDataset(self.data_root, val_meta, label_dict,
                                       transform=self.transforms)

    def _plot_class_balance(self, meta: pd.DataFrame, tag: str, title: str):
        cnt = Counter(meta.treatment_id)
        data = [[cname, cnt[cname]] for cname in self.class_names]
        table = wandb.Table(data=data, columns=["treatment", "frequency"])
        wandb.log({tag: wandb.plot.bar(table, "treatment", "frequency", title=title)})

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.train_bs,
                          num_workers=self.num_workers,
                          worker_init_fn=self._dataloader_worker_init,
                          collate_fn=DataItem.collate,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.test_bs,
                          num_workers=self.num_workers,
                          worker_init_fn=self._dataloader_worker_init,
                          collate_fn=DataItem.collate)

    @staticmethod
    def _dataloader_worker_init(*args, **kwargs):
        """Seeds the workers within the Dataloader"""
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
