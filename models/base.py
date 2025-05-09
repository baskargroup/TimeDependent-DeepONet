import pytorch_lightning as pl
import torch
from einops import rearrange

class BaseLightningModule(pl.LightningModule):
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _masked_mse(self, y_hat, y_true, sdf):
        mask = (sdf > 0).flatten(1).unsqueeze(-1)
        se = ((y_hat - y_true) ** 2) * mask
        return se.sum() / mask.sum()

    def training_step(self, batch, batch_idx):
        (branch, re, coords, sdf), tgt = batch
        y_hat = self.model((branch, re, coords, sdf))
        loss = self._masked_mse(y_hat, tgt, sdf)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (branch, re, coords, sdf), tgt = batch
        y_hat = self.model((branch, re, coords, sdf))
        loss = self._masked_mse(y_hat, tgt, sdf)
        self.log('val_loss', loss)
        return loss
