import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from model import Fou2
from data.dataset import AudioHDF
import hydra
from omegaconf import DictConfig

import os


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    model = Fou2(
        cfg.models.fou2,
        cfg.ema,
        cfg.stft,
        cfg.lr,
    )
    io = model.io
    print(io)

    train_set = AudioHDF(
        os.path.join(cfg.dataset_dir, "train", "hdf", "instrumentals.hdf5"),
        os.path.join(cfg.dataset_dir, "train", "hdf", "vocals.hdf5"),
        input_length=io,
        output_length=io,
        random_hop=cfg.random_hop,
    )
    val_set = AudioHDF(
        os.path.join(cfg.dataset_dir, "test", "hdf", "instrumentals.hdf5"),
        os.path.join(cfg.dataset_dir, "test", "hdf", "vocals.hdf5"),
        input_length=io,
        output_length=io,
        random_hop=False,
    )

    train_dl = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_dl = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    checkpoint_last = ModelCheckpoint(
        save_top_k=3, monitor="step", mode="max", filename="last_check_{epoch:02d}"
    )
    checkpoint_best = ModelCheckpoint(
        save_top_k=3, monitor="val_loss", mode="min", filename="best_check_{epoch:02d}"
    )

    save_dir = os.path.join(cfg.save_dir, cfg.tag)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=cfg.max_epochs,
        min_epochs=cfg.min_epochs,
        default_root_dir=save_dir,
        callbacks=[checkpoint_best, checkpoint_last],
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
