from typing import Sequence

import hydra
import pytorch_lightning as pl
import torch
import torchmetrics
# import pretrainedmodels
import torchvision.models as models
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import MultiMarginLoss

from augmentations import AUGS_SELECT
from data import DataItemBatch
from datamodule import TubulesDataModule
from utils import plot_confusion_matrix, fig_to_pil


BACKBONES = {"resnet18": models.resnet18,
             "resnet152": models.resnet152}

CRITERIA = {"CE": nn.NLLLoss(),
            "hinge": MultiMarginLoss()}


class TubulesClassifier(pl.LightningModule):
    def __init__(self, *, backbone_name: str,
                 criterion_name: str,
                 class_names: Sequence[str], frozen_encoder: bool = True,
                 log_confmat_every: int = 30,
                 # TODO n_samples into config
                 log_samples_every: int = 40):
        super().__init__()
        self.log_confmat_every = log_confmat_every
        self.log_samples_every = log_samples_every
        self.class_names = class_names
        self.criterion = CRITERIA[criterion_name]
        num_classes = len(class_names)

        # self.backbone = pretrainedmodels.__dict__[backbone_name](pretrained="imagenet")
        self.backbone = BACKBONES[backbone_name](pretrained="imagenet")

        # dim_feats = self.backbone.last_linear.in_features
        dim_feats = self.backbone.fc.in_features

        if frozen_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # self.backbone.last_linear = nn.Linear(dim_feats, num_classes)
        self.backbone.fc = nn.Linear(dim_feats, num_classes)

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        self.val_confmat = torchmetrics.ConfusionMatrix(num_classes)
        self.train_confmat = torchmetrics.ConfusionMatrix(num_classes)

    def forward(self, x):
        y_hat = self.backbone(x)
        log_probs = self.log_softmax(y_hat)
        return log_probs

    def training_step(self, batch: DataItemBatch, batch_idx: int):
        input_imgs, labels, orig_imgs = batch.input_images, batch.labels, batch.original_images

        log_probs = self.forward(input_imgs)
        loss = self.criterion(log_probs, labels)
        pred = log_probs.argmax(1)

        self.train_accuracy(pred, labels)
        self.train_confmat(pred, labels)

        if batch_idx % self.log_samples_every == 0:
            self._plot_samples(batch, pred, tag="train")
        wandb.log({"train/loss_step": loss.item()})

        return loss

    def _plot_samples(self, batch: DataItemBatch, pred: torch.Tensor, tag: str, n_samples: int = 5):
        input_imgs, labels, orig_imgs = batch.input_images, batch.labels, batch.original_images
        samples = list(zip(orig_imgs, pred, labels))[:n_samples]
        wandb.log({f"{tag}/samples": [wandb.Image(img,
                                                  caption=f"pred:{self.class_names[p]}, gt:{self.class_names[l]}")
                                      for img, p, l in samples]})
        red_samples = list(zip(input_imgs[:n_samples, 0, :, :].cpu().numpy(), pred, labels))
        wandb.log({f"{tag}/red_samples": [wandb.Image(img,
                                                      caption=f"pred:{self.class_names[p]}, gt:{self.class_names[l]}")
                                          for img, p, l in red_samples]})
        green_samples = list(zip(input_imgs[:n_samples, 1, :, :].cpu().numpy(), pred, labels))
        wandb.log({f"{tag}/green_samples": [wandb.Image(img,
                                                        caption=f"pred:{self.class_names[p]}, gt:{self.class_names[l]}")
                                            for img, p, l in green_samples]})

    def training_epoch_end(self, training_step_outputs):
        wandb.log({"train/acc_epoch": self.train_accuracy.compute()})
        self.train_accuracy.reset()

        if self.current_epoch % self.log_confmat_every == 0:
            f = plot_confusion_matrix(self.train_confmat.compute().int().cpu().numpy(),
                                      labels=self.class_names)
            wandb.log({"train/confusion_matrix": [wandb.Image(fig_to_pil(f))]})
        self.train_confmat.reset()

    def validation_step(self, batch, batch_idx):
        input_imgs, labels, orig_imgs = batch.input_images, batch.labels, batch.original_images
        log_probs = self.forward(input_imgs)
        loss = self.criterion(log_probs, labels)
        pred = log_probs.argmax(1)

        self.val_accuracy(pred, labels)
        self.val_confmat(pred, labels)
        if batch_idx % self.log_samples_every == 0:
            self._plot_samples(batch, pred, tag="val")
        wandb.log({"val/loss_step": loss})
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        wandb.log({"val/acc_epoch": self.val_accuracy.compute()})
        self.val_accuracy.reset()

        if self.current_epoch % self.log_confmat_every == 0:
            f = plot_confusion_matrix(self.val_confmat.compute().int().cpu().numpy(),
                                      labels=self.class_names)
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
    transforms = AUGS_SELECT[cfg.augmentations]()
    dm = TubulesDataModule(data_root=cfg.dataset.data_root,
                           meta_path=cfg.dataset.meta_path,
                           cmpds=cfg.dataset.cmpds,
                           train_bs=cfg.train_bs, test_bs=cfg.test_bs,
                           num_workers=cfg.num_workers,
                           val_size=cfg.val_size,
                           balance_classes=cfg.balance_classes,
                           transforms=transforms)
    model = TubulesClassifier(backbone_name=cfg.backbone_name,
                              criterion_name=cfg.criterion,
                              class_names=dm.class_names,
                              frozen_encoder=cfg.frozen_encoder)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
