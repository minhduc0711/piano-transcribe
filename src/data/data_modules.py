from .datasets import MAPSDataset

from pathlib import Path
import shutil
import sys

from tqdm import tqdm
from torch.utils.data import DataLoader
import torchaudio
from torchaudio import transforms
import pytorch_lightning as pl

torchaudio.set_audio_backend("sox_io")


class MAPSDataModule(pl.LightningDataModule):
    RAW_DATA_DIR = "data/raw/MAPS/"
    PROCESSED_DATA_DIR = "data/processed/MAPS_MUS/"

    SAMPLE_RATE = 16000

    def __init__(self, batch_size):
        super(MAPSDataModule, self).__init__()
        self.batch_size = batch_size

    def prepare_data(self, force=False):
        """
        Resamples audio files to 16 KHz
        """
        src_dir = Path(self.RAW_DATA_DIR)
        dest_dir = Path(self.PROCESSED_DATA_DIR)
        new_sample_rate = self.SAMPLE_RATE

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

    def setup(self, stage=None, n_mels=229, hop_length=512, n_fft=2048):
        # using melspectrogram params from "onsets and frames" paper
        audio_transform = transforms.MelSpectrogram(
            n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
        )
        # split audio into segments of length 20 seconds
        max_steps = int((20.48 * self.SAMPLE_RATE) / hop_length)

        if stage == "fit" or stage is None:
            train_subsets = [
                "AkPnBcht",
                "AkPnBsdf",
                "AkPnCGdD",
                "AkPnStgb",
                "SptkBGAm",
                "StbgTGd2",
            ]
            val_subsets = ["SptkBGCl"]
            self.train_ds = MAPSDataset(
                self.PROCESSED_DATA_DIR,
                subsets=train_subsets,
                max_steps=max_steps,
                audio_transform=audio_transform,
            )
            self.val_ds = MAPSDataset(
                self.PROCESSED_DATA_DIR,
                subsets=val_subsets,
                max_steps=max_steps,
                audio_transform=audio_transform,
            )
            self.dims = tuple(self.train_ds[0]["audio"].shape)

        if stage == "test" or stage is None:
            test_subsets = ["ENSTDkAm", "ENSTDkCl"]
            self.test_ds = MAPSDataset(
                self.PROCESSED_DATA_DIR,
                subsets=test_subsets,
                max_steps=max_steps,
                audio_transform=audio_transform,
            )
            self.dims = tuple(self.test_ds[0]["audio"].shape)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
