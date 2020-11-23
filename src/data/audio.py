import torch
import torch.nn as nn
import torchaudio

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
