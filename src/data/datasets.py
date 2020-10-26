from .midi import parse_midi

from pathlib import Path
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

torchaudio.set_audio_backend("sox_io")

MIDI_MIN_PITCH = 21
MIDI_MAX_PITCH = 108


class MAPSDataset(Dataset):
    def __init__(
        self,
        data_dir,
        max_steps=None,
        subsets=None,
        audio_transform=None,
        onset_length_in_ms=32,
        offset_length_in_ms=32,
        seed=42
    ):
        data_dir = Path(data_dir)
        self.audio_paths = []
        for subset in subsets:
            subset_dir = data_dir / subset
            self.audio_paths.extend(list(subset_dir.glob("*.wav")))

        self.max_steps = max_steps
        self.audio_transform = audio_transform
        self.onset_length_in_ms = onset_length_in_ms
        self.offset_length_in_ms = offset_length_in_ms
        self.random = np.random.RandomState(seed)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        audio, sample_rate = torchaudio.load(str(audio_path))
        # convert to mono channel if necessary
        if audio.ndim > 1:
            audio = audio.mean(dim=0)
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)
            stride = self.audio_transform.hop_length
        else:
            stride = 1

        midi_path = audio_path.with_suffix(".mid")
        note_df = parse_midi(midi_path)
        num_pitches = MIDI_MAX_PITCH - MIDI_MIN_PITCH + 1

        onset_length_in_samples = sample_rate * self.onset_length_in_ms // 1000
        offset_length_in_samples = sample_rate * self.offset_length_in_ms // 1000
        num_steps_onset = onset_length_in_samples // stride
        num_steps_offset = offset_length_in_samples // stride
        num_steps_total = audio.shape[-1]

        frame_labels = torch.zeros(num_pitches, num_steps_total, dtype=torch.uint8)
        velocity = torch.zeros(num_pitches, num_steps_total)

        for _, (onset, offset, pitch, vel) in note_df.iterrows():
            onset_start = int(round(onset * sample_rate / stride))
            onset_end = min(num_steps_total, onset_start + num_steps_onset)

            frame_end = int((round(offset * sample_rate / stride)))
            frame_end = min(num_steps_total, frame_end)
            offset_end = min(num_steps_total, frame_end + num_steps_offset)

            p = int(pitch) - MIDI_MIN_PITCH
            frame_labels[p, onset_start:onset_end] = 3  # onset
            frame_labels[p, onset_end:frame_end] = 2  # note frames
            frame_labels[p, frame_end:offset_end] = 1  # offset
            velocity[p, onset_start:offset_end] = vel / 128.

        if self.max_steps is not None:
            step_start = self.random.randint(num_steps_total - self.max_steps)
            step_end = step_start + self.max_steps

            audio = audio[..., step_start:step_end]
            frame_labels = frame_labels[..., step_start:step_end]
            velocity = velocity[..., step_start:step_end]

        onsets = (frame_labels == 3).float()
        offsets = (frame_labels == 1).float()
        frames = (frame_labels > 1).float()

        return {
            "audio_path": str(audio_path.resolve()),
            "audio": audio,
            "onsets": onsets,
            "offsets": offsets,
            "frames": frames,
            "velocity": velocity,
        }