import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger
import time
from data.dataset import LidDrivenDataset2DTime
from models.geometric_deeponet.geometric_deeponet import GeometricDeepONetTime


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)


def load_config(path):
    cfg = OmegaConf.load(path)
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def main(config_path: str, config_obj=None):
    # Load or override config
    if config_obj is not None:
        cfg = config_obj
    else:
        cfg = load_config(config_path)

    set_seed(cfg.trainer.seed)

    # Dataset split (80/20)
    ds = LidDrivenDataset2DTime(
        cfg.data.file_path_train_x,
        cfg.data.file_path_train_y,
        cfg.model.num_input_timesteps,
        cfg.model.final_timestep,
        cfg.data.every_nth_timestep,
        cfg.model.height,
        cfg.model.width,
        cfg.data.type,
        cfg.model.includePressure,
        cfg.model.output_channels
    )
    n = len(ds)
    train_n = int(0.8 * n)
    train_ds, val_ds = random_split(
        ds, [train_n, n - train_n],
        generator=torch.Generator().manual_seed(cfg.trainer.seed)
    )

    # DataLoaders
    num_workers = cfg.trainer.get('num_workers', 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Model
    model = GeometricDeepONetTime(**OmegaConf.to_container(cfg.model))

    # Callbacks
    cb = []
    ck = cfg.callbacks.checkpoint
    ck_dir = os.path.abspath(ck.dirpath)
    os.makedirs(ck_dir, exist_ok=True)
    cb.append(
        ModelCheckpoint(
            monitor=ck.monitor,
            dirpath=ck_dir,
            filename=ck.filename,
            save_top_k=ck.save_top_k,
            mode=ck.mode
        )
    )
    es = cfg.callbacks.get('early_stopping')
    if es:
        cb.append(
            EarlyStopping(
                monitor=es.monitor,
                patience=es.patience,
                mode=es.mode
            )
        )

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,        
        name=cfg.wandb.name,              
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    artifact = wandb.Artifact('run-config', type='config')
    artifact.add_file(config_path)              # path to your conf.yaml
    wandb_logger.experiment.log_artifact(artifact)
    # Trainer
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        callbacks=cb,
        logger=wandb_logger,  
        log_every_n_steps=cfg.trainer.log_every_n_steps
    )
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    total_seconds = time.time() - start_time
    wandb_logger.experiment.log({"total_runtime_sec": total_seconds})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, help='path to your OmegaConf YAML')
    args = parser.parse_args()
    main(config_path=args.config_path)
