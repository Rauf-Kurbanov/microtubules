from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np


class TubulesDataset(Dataset):

    COMPOUND_DICT = {"DMSO": 0,
                     "H": 1,
                     "B": 2,
                     "D": 3}

    MAX = ()
    MEAN = ()
    STD = ()

    def __init__(self, data_root: Path, meta_path: Path):
        super().__init__()
        self.meta = pd.read_csv(meta_path)
        self.data_root = data_root

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        full_path = self.data_root / row.path
        im = Image.open(full_path)
        imarray = np.array(im)

        label = self.COMPOUND_DICT[row["compound (short)"]]

        return imarray, label

    def __len__(self):
        return len(self.meta)


if __name__ == '__main__':
    data_root = Path("/Users/raufkurbanov/Data/Aleksi")
    meta_path = Path("/Users/raufkurbanov/Data/Aleksi/meta.csv")

    ds = TubulesDataset(data_root, meta_path)
    print(ds[0])