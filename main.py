from typing import Sequence

import hydra
import pytorch_lightning as pl
import torch
import torchmetrics
import wandb
from omegaconf import DictConfig, OmegaConf
from pretrainedmodels import resnet50
# from pytorch_lightning.loggers import WandbLogger
from torch import nn

from augmentations import AUGS_SELECT
from datamodule import TubulesDataModule
from utils import plot_confusion_matrix, fig_to_pil
# import pretrainedmodels
import torchvision.models as models


class TubulesClassifier(pl.LightningModule):
    def __init__(self, *, backbone_name: str,
                 class_names: Sequence[str], frozen_encoder: bool = True,
                 log_confmat_every: int = 10):
        super().__init__()
        self.log_confmat_every = log_confmat_every
        self.class_names = class_names
        num_classes = len(class_names)

        # self.backbone = pretrainedmodels.__dict__[backbone_name](pretrained="imagenet")
        self.backbone = models.resnet152(pretrained=True)
        # self.backbone = resnet50(pretrained="imagenet")
        # dim_feats = self.backbone.last_linear.in_features
        dim_feats = self.backbone.fc.in_features

        if frozen_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # self.backbone.last_linear = nn.Linear(dim_feats, num_classes)
        self.backbone.fc = nn.Linear(dim_feats, num_classes)

        # TODO move into sequential
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        self.val_confmat = torchmetrics.ConfusionMatrix(num_classes)
        self.train_confmat = torchmetrics.ConfusionMatrix(num_classes)

    def forward(self, x):
        y_hat = self.backbone(x)
        log_probs = self.log_softmax(y_hat)
        return log_probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        log_probs = self.forward(x)
        loss = self.criterion(log_probs, y)
        pred = log_probs.argmax(1)

        self.train_accuracy(pred, y)
        self.train_confmat(pred, y)
        # TODO try log_dict
        # self.logger.experiment.log({"train/loss_step": loss})
        wandb.log({"train/loss_step": loss.item()})

        return loss

    def training_epoch_end(self, training_step_outputs):
        # self.logger.experiment.log({"train/acc_epoch": self.train_accuracy.compute()})
        wandb.log({"train/acc_epoch": self.train_accuracy.compute()})

        if self.current_epoch % self.log_confmat_every == 0:
            f = plot_confusion_matrix(self.train_confmat.compute().int().cpu().numpy(),
                                      labels=self.class_names)
            # self.logger.experiment.log({"train/confusion_matrix": [wandb.Image(fig_to_pil(f))]})
            wandb.log({"train/confusion_matrix": [wandb.Image(fig_to_pil(f))]})
        self.train_confmat.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        log_probs = self.forward(x)
        loss = self.criterion(log_probs, y)
        pred = log_probs.argmax(1)

        self.val_accuracy(pred, y)
        self.val_confmat(pred, y)
        # self.logger.experiment.log({"val/loss_step": loss})
        wandb.log({"val/loss_step": loss})
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        # self.logger.experiment.log({"val/acc_epoch": self.val_accuracy.compute()})
        wandb.log({"val/acc_epoch": self.val_accuracy.compute()})

        if self.current_epoch % self.log_confmat_every == 0:
            f = plot_confusion_matrix(self.val_confmat.compute().int().cpu().numpy(),
                                      labels=self.class_names)
            # self.logger.experiment.log({"val/confusion_matrix": [wandb.Image(fig_to_pil(f))]})
            wandb.log({"val/confusion_matrix": [wandb.Image(fig_to_pil(f))]})
        self.val_confmat.reset()

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters())


@hydra.main(config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    wandb.init(project="microtubules",
               entity="rauf-kurbanov",
               config=config_dict,
               tags=["debug"])
    # logger = WandbLogger(project="microtubules",
    #                      entity="rauf-kurbanov",
    #                      config=config_dict,
    #                      tags=["debug"])
    transforms = AUGS_SELECT[cfg.augmentations]()
    dm = TubulesDataModule(data_root=cfg.dataset.data_root,
                           meta_path=cfg.dataset.meta_path,
                           cmpds=cfg.dataset.cmpds,
                           train_bs=cfg.train_bs, test_bs=cfg.test_bs,
                           num_workers=cfg.num_workers,
                           val_size=cfg.val_size,
                           balance_classes=cfg.balance_classes,
                           transforms=transforms,
                           # logger=logger
                           )
    model = TubulesClassifier(backbone_name=cfg.backbone_name,
                              class_names=dm.class_names,
                              frozen_encoder=cfg.frozen_encoder)
    trainer = pl.Trainer(**cfg.trainer,
                         # logger=logger
                         )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
