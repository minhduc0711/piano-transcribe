from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data import MAPSDataModule
from src.models.onsets_and_frames import OnsetsAndFrames

parser = ArgumentParser()
parser.add_argument("--batch-size", type=int, default=8)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

dm = MAPSDataModule(batch_size=args.batch_size)
model = OnsetsAndFrames(in_feats=229)

ckpt_callback = ModelCheckpoint(monitor="valid_loss", save_last=True, save_top_k=5,
                                filename="onf-MAPS-{epoch:02d}-{valid_loss:.2f}")
trainer = Trainer.from_argparse_args(args,
                                     callbacks=[ckpt_callback])
trainer.fit(model, datamodule=dm)
