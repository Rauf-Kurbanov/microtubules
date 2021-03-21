from typing import Sequence

import hydra
import pytorch_lightning as pl
import torch
import torchmetrics
import wandb
from omegaconf import DictConfig, OmegaConf
from pretrainedmodels import resnet50
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from datamodule import TubulesDataModule
from utils import plot_confusion_matrix, fig_to_pil


class TubulesClassifier(pl.LightningModule):
    def __init__(self, class_names: Sequence[str]):
        super().__init__()
        self.class_names = class_names
        num_classes = len(class_names)

        # self.save_hyperparameters()  # TODO
        self.backbone = resnet50(pretrained="imagenet")
        dim_feats = self.backbone.last_linear.in_features
        self.backbone.last_linear = nn.Linear(dim_feats, num_classes)

        # TODO UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
        self.log_softmax = nn.LogSoftmax()
        self.criterion = nn.NLLLoss()

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        self.val_confmat = torchmetrics.ConfusionMatrix(num_classes)

    def forward(self, x):
        y_hat = self.backbone(x)
        return self.log_softmax(y_hat)

    def training_step(self, batch, batch_idx):
        x, y = batch
        log_probs = self.forward(x)
        loss = self.criterion(log_probs, y)

        pred = log_probs.argmax(1)
        self.train_accuracy(pred, y)
        # TODO try log_dict
        self.logger.experiment.log({"train/loss_step": loss})

        return loss

    def training_epoch_end(self, training_step_outputs):
        self.logger.experiment.log({"train/acc_epoch": self.train_accuracy.compute()})

    def validation_step(self, batch, batch_idx):
        x, y = batch
        log_probs = self.forward(x)

        pred = log_probs.argmax(1)
        self.val_accuracy(pred, y)
        loss = self.criterion(log_probs, y)

        self.val_confmat(pred, y)
        self.logger.experiment.log({"val/loss_step": loss})
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        self.logger.experiment.log({"val/acc_epoch": self.val_accuracy.compute()})

        f = plot_confusion_matrix(self.val_confmat.compute().int().cpu().numpy(),
                                  labels=self.class_names)
        self.logger.experiment.log({"val/confusion_matrix": [wandb.Image(fig_to_pil(f), caption="Label")]})

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters())


@hydra.main(config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(1234)

    dm = TubulesDataModule(cfg.dataset.data_root, cfg.dataset.meta_path,
                           cfg.dataset.cmpds,
                           train_bs=64, test_bs=128, num_workers=16,
                           val_size=0.2,
                           balance=False,
                           )
    model = TubulesClassifier(dm.class_names)
    trainer = pl.Trainer(gpus=1,
                         # limit_train_batches=1,
                         # limit_val_batches=1,
                         checkpoint_callback=False,
                         # auto_scale_batch_size=True,
                         logger=WandbLogger(project="microtubules",
                                            entity="rauf-kurbanov",
                                            tags=["debug"]))
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
