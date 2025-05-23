import pytorch_lightning as pl
import torch


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
        if self.hparams.use_derivative_loss:
            loss = self._derivative_loss(y_hat, tgt, sdf)
        else:
            loss = self._masked_mse(y_hat, tgt, sdf)        
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (branch, re, coords, sdf), tgt = batch
        y_hat = self.model((branch, re, coords, sdf))
        if self.hparams.use_derivative_loss:
            loss = self._derivative_loss(y_hat, tgt, sdf)
        else:
            loss = self._masked_mse(y_hat, tgt, sdf)                
        
        self.log('val_loss', loss)
        return loss

    def _derivative_loss(self, y_hat, y_true, sdf):
        # --- reshape [B,1,p,C] → [B,C,H,W] ---
        B, _, p, C = y_hat.shape
        H, W = self.hparams.height, self.hparams.width
        yh = y_hat.squeeze(1).permute(0,2,1).reshape(B, C, H, W)
        yt = y_true.squeeze(1).permute(0,2,1).reshape(B, C, H, W)

        deriv_hat = self.deriv_calc(yh)
        deriv_true = self.deriv_calc(yt)
        fluid_mask = (sdf > 0)  # [B,1,H,W]
        delta = self.hparams.domain_length_y / H
        loss = 0.0  
        # Derivative tensors come out at resolution (H-1)x(W-1) so crop the fluid_mask to match:
        dm = fluid_mask[:, :, :-1, :-1].unsqueeze(1)  # → [B,1,1,H-1,W-1]
        for key in ('u_x','u_y','v_x','v_y'):
            diff = deriv_hat[key] - deriv_true[key]     # [B,ngp,1,H-1,W-1]
            # apply mask before averaging
            deriv_loss  = delta * (diff.pow(2) * dm).sum() / dm.sum()
            self.log(f"deriv_loss/{key}", deriv_loss, on_step=False, on_epoch=True)
            loss = loss + deriv_loss
            
        inner = (sdf > 0) & (sdf <= delta)  # [B,1,H,W]
        if inner.any().item():
            u_hat = yh[:, 0:1]  # [B,1,H,W]
            v_hat = yh[:, 1:2]
            if self.hparams.use_zero_bc:
                bc_loss = 1000 * (u_hat[inner].pow(2) + v_hat[inner].pow(2)).mean()
                             
            else:
                u_true = yt[:, 0:1]
                v_true = yt[:, 1:2]
                u_target = u_true[inner]
                v_target = v_true[inner]
                bc_loss = ((u_hat[inner] - u_target).pow(2) + (v_hat[inner] - v_target).pow(2)).mean()
            
            self.log("boundary_bc_loss", bc_loss, on_step=False, on_epoch=True)
            loss = loss + bc_loss
                      
        return loss
