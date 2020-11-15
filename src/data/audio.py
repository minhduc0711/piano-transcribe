import torch
import torchaudio


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
        # bring time to the first dim, assuming the transform above
        # outputs a tensor of shape (n_feats, time)
        audio = audio.transpose(0, 1)

    return audio
