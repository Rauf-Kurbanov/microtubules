from torch.utils.data import DataLoader, random_split
from pathlib import Path
import math
from data import TubulesDataset
from torchvision import transforms
import pytorch_lightning as pl
from torchvision.models import resnet18, resnet50
from torch.nn import functional as F
import pandas as pd
from argparse import ArgumentParser
import torch
from data import TRAIN_TRANSFORM
# from callbacks import AccLoggerCallback
from pytorch_lightning.loggers import WandbLogger


class LitClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.save_hyperparameters()
        self.backbone = resnet50(pretrained=True)
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc_step', self.accuracy(y_hat, y))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # print("batch", batch)
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters())

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    # parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    data_root = Path("/home/rauf/Data/tubules/Aleksi")
    meta_path = Path("/home/rauf/Data/tubules/Aleksi/dataset.csv")
    meta_df = pd.read_csv(meta_path)

    dataset = TubulesDataset(data_root, meta_df,
                             transform=TRAIN_TRANSFORM)
    dlen = len(meta_df)
    val_size = math.floor(dlen * 0.2)
    train_size = dlen - val_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              # num_workers=8,
                              pin_memory=True
                              )
    val_loader = DataLoader(val_data, batch_size=512, num_workers=8,
                            pin_memory=True)
    test_loader = val_loader

    model = LitClassifier()
    # trainer = pl.Trainer(callbacks=[AccLoggerCallback()]).from_argparse_args(args)
    # trainer = pl.Trainer().from_argparse_args(args, callbacks=[AccLoggerCallback()])
    trainer = pl.Trainer().from_argparse_args(args, checkpoint_callback=False,
                                              # check_val_every_n_epoch=100,
                                              # overfit_batches=1,
                                              logger=WandbLogger(project="microtubules", entity="rauf-kurbanov")
                                              )
    # trainer = pl.Trainer.from_argparse_args(args, overfit_batches=1)
    trainer.fit(model, train_loader, val_loader)

    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    main()
