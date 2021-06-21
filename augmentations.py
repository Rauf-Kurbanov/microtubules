import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def train_no_aug_crop() -> A.Compose:
    train_transform = A.Compose([
        A.ToFloat(max_value=float(2 ** 16 - 1)),
        # A.RandomCrop(height=224, width=224),
        A.RandomResizedCrop(height=224, width=224, scale=(0.1, 0.25)),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

    return train_transform


def train_no_aug_resize() -> A.Compose:
    train_transform = A.Compose([
        A.ToFloat(max_value=float(2 ** 16 - 1)),
        A.Resize(height=224, width=224),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

    return train_transform


AUGS_SELECT = {
    "no_aug_crop": train_no_aug_crop,
    "no_aug_resize": train_no_aug_resize,
}


