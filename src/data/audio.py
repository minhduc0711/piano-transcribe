import numpy as np

import torch
import torch.nn as nn
import torchaudio
import madmom

from src.layer_utils import Lambda


def load_audio(audio_path,
               audio_transform=None,
               new_sample_rate=16000) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(str(audio_path))
    # resample to 16kHZ
    if sample_rate != new_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                   new_freq=new_sample_rate)
        audio = resampler(audio)
    # convert to mono channel if necessary
    if audio.ndim > 1:
        audio = audio.mean(dim=0)
    if audio_transform is not None:
        audio = audio_transform(audio)

    return audio


# default audio features in ONSETS AND FRAMES paper
onf_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(n_mels=229, hop_length=512, n_fft=2048),
    Lambda(lambda x: torch.log(torch.clamp(x, min=1e-5))),
    Lambda(lambda x: torch.transpose(x, 0, 1))
)


class MadmomSpectrogram:
    """
    Extract audio spectrogram using madmom
    """
    def __init__(self, hop_length, sample_rate=16000):
        self.hop_length = hop_length
        self.sample_rate = sample_rate

    def __call__(self, x):
        x = madmom.audio.signal.Signal(x.numpy(), sample_rate=self.sample_rate)
        audio_options = dict(
            num_channels=1,
            sample_rate=self.sample_rate,
            filterbank=madmom.audio.filters.LogarithmicFilterbank,
            frame_size=4096,
            fft_size=4096,
            hop_size=self.hop_length,  # 25 fps
            num_bands=48,
            fmin=30,
            fmax=8000.0,
            fref=440.0,
            norm_filters=True,
            unique_filters=True,
            circular_shift=False,
            norm=True
        )

        # dt = float(audio_options['hop_size']) / float(audio_options['sample_rate'])
        sig = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(x, **audio_options)
        return torch.from_numpy(np.array(sig.data))
