import torch
import pytorch_lightning as pl

from src.data import MAPSDataModule
from src.models.onsets_and_frames import OnsetsAndFrames


gpus = 0 if torch.cuda.is_available() else None
dm = MAPSDataModule(batch_size=4)
model = OnsetsAndFrames(in_feats=229)

trainer = pl.Trainer(gpus=gpus)
trainer.fit(model, datamodule=dm)
