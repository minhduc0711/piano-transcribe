from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data import MAPSDataModule
from src.data.audio import onf_transform
from src.models.onsets_and_frames import OnsetsAndFrames

parser = ArgumentParser()
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--debug", action="store_true")
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

sample_rate = 16000
# split audio into segments of ~20 seconds
max_steps = int((20.48 * sample_rate) / 512)
dm = MAPSDataModule(batch_size=args.batch_size,
                    sample_rate=sample_rate,
                    max_steps=max_steps,
                    audio_transform=onf_transform,
                    hop_length=512,
                    debug=args.debug)
dm.setup(stage="fit")

model = OnsetsAndFrames(in_feats=229,
                        lr_sched_step_size=int(10000 / (len(dm.train_ds) / args.batch_size)))

ckpt_callback = ModelCheckpoint(monitor="valid_loss", save_last=True, save_top_k=5,
                                filename="onf-MAPS-{epoch:02d}-{valid_loss:.2f}")
trainer = Trainer.from_argparse_args(args,
                                     callbacks=[ckpt_callback])
trainer.fit(model, datamodule=dm)
