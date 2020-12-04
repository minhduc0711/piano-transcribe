from .datasets import MAPSDataset

from pathlib import Path
import shutil

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from torchaudio import transforms
import pytorch_lightning as pl

from tqdm import tqdm

torchaudio.set_audio_backend("sox_io")


class MAPSDataModule(pl.LightningDataModule):
    RAW_DATA_DIR = "data/raw/MAPS/"
    PROCESSED_DATA_DIR = "data/processed/MAPS_MUS/"

    def __init__(self, batch_size: int,
                 sample_rate: int,
                 max_steps: int,
                 audio_transform=None,
                 hop_length=1,
                 num_workers=4,
                 lazy_loading=False,
                 debug=False):
        super(MAPSDataModule, self).__init__()
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.max_steps = max_steps
        self.audio_transform = audio_transform
        self.hop_length = hop_length
        self.num_workers = num_workers
        self.lazy_loading = lazy_loading
        self.debug = debug

    def prepare_data(self, force=False):
        """
        Resamples audio files to 16 KHz
        """
        src_dir = Path(self.RAW_DATA_DIR)
        dest_dir = Path(self.PROCESSED_DATA_DIR)
        new_sample_rate = self.sample_rate

        if dest_dir.exists():
            if force:
                while True:
                    ans = input(f"{dest_dir} already exists. Continue? y/[n]: ")
                    if ans.lower() == "y":
                        break
                    elif ans.lower() == "n" or ans == "":
                        return
            else:
                return

        print("resampling audio files to 16 KHz...")
        for subset_dir in tqdm(list(src_dir.glob("**/MUS/")), desc="subset"):
            subset_name = subset_dir.parent.name
            dest_subset_dir = dest_dir / subset_name
            dest_subset_dir.mkdir(parents=True, exist_ok=True)

            for audio_path in tqdm(
                list(subset_dir.glob("*.wav")), desc="audio file", leave=False
            ):
                # copy label files
                for ext in [".txt", ".mid"]:
                    shutil.copy(audio_path.with_suffix(ext), dest_subset_dir)
                # resample audio file
                wav, sample_rate = torchaudio.load(str(audio_path))
                resampler = transforms.Resample(
                    orig_freq=sample_rate, new_freq=new_sample_rate
                )
                torchaudio.save(
                    filepath=str(dest_subset_dir / audio_path.name),
                    tensor=resampler(wav),
                    sample_rate=new_sample_rate,
                )

    def setup(self, stage=None):
        train_subsets = [
            "AkPnBcht",
            "AkPnBsdf",
            "AkPnCGdD",
            "AkPnStgb",
            "SptkBGAm",
            "StbgTGd2",
        ]
        val_subsets = ["SptkBGCl"]
        test_subsets = ["ENSTDkAm", "ENSTDkCl"]

        if stage == "fit" or stage is None:
            self.train_ds = MAPSDataset(
                self.PROCESSED_DATA_DIR,
                subsets=train_subsets,
                max_steps=self.max_steps,
                sample_rate=self.sample_rate,
                audio_transform=self.audio_transform,
                hop_length=self.hop_length,
                lazy_loading=self.lazy_loading,
                debug=self.debug
            )
            self.val_ds = MAPSDataset(
                self.PROCESSED_DATA_DIR,
                subsets=val_subsets,
                max_steps=self.max_steps,
                sample_rate=self.sample_rate,
                audio_transform=self.audio_transform,
                hop_length=self.hop_length,
                lazy_loading=self.lazy_loading,
                debug=self.debug
            )
            self.dims = tuple(self.train_ds[0]["audio"].shape)

        if stage == "test" or stage is None:
            self.test_ds = MAPSDataset(
                self.PROCESSED_DATA_DIR,
                subsets=test_subsets,
                max_steps=self.max_steps,
                sample_rate=self.sample_rate,
                audio_transform=self.audio_transform,
                hop_length=self.hop_length,
                lazy_loading=self.lazy_loading,
                debug=self.debug
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, item_dicts):
        batch = {}
        for k in item_dicts[0].keys():
            batch[k] = [d[k] for d in item_dicts]
            if isinstance(batch[k][0], torch.Tensor):
                batch[k] = torch.stack(batch[k], 0)
        return batch
