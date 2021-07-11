import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.)
STD = (0.229, 0.224, 1)


def train_no_aug_crop() -> A.Compose:
    train_transform = A.Compose([
        A.RandomCrop(height=512, width=512),
        A.Resize(height=224, width=224),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

    return train_transform


def train_crop_flip() -> A.Compose:
    train_transform = A.Compose([
        A.RandomCrop(height=512, width=512),
        A.Resize(height=224, width=224),
        A.Flip(p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

    return train_transform


def train_no_aug_resize() -> A.Compose:
    train_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

    return train_transform


def train_flip() -> A.Compose:
    train_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Flip(p=0.6),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

    return train_transform

AUGS_SELECT = {
    "no_aug_crop": train_no_aug_crop,
    "no_aug_resize": train_no_aug_resize,
    "flip": train_flip,
    "crop_flip": train_crop_flip,
}

