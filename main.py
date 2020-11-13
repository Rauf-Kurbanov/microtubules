from torch.utils.data import DataLoader, random_split
from pathlib import Path
import math
from data import TubulesDataset
from torchvision import transforms

def main():
    data_root = Path("/Users/raufkurbanov/Data/Aleksi")
    meta_path = Path("/Users/raufkurbanov/Data/Aleksi/meta.csv")

    dataset = TubulesDataset(data_root, meta_path,
                             transform=transforms.ToTensor())
    dlen = len(data_root)
    val_size = math.floor(dlen * 0.2)
    train_size = dlen - val_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32)

if __name__ == '__main__':
    main()
