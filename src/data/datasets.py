from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from .midi import parse_midi

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
        seed=42,
        lazy_loading=False,
        debug=False
    ):
        self.max_steps = max_steps
        self.audio_transform = audio_transform
        self.onset_length_in_ms = onset_length_in_ms
        self.offset_length_in_ms = offset_length_in_ms
        self.random = np.random.RandomState(seed)
        self.lazy_loading = lazy_loading

        data_dir = Path(data_dir)
        self.audio_paths = []
        if subsets is not None:
            for subset in subsets:
                subset_dir = data_dir / subset
                self.audio_paths.extend(list(subset_dir.glob("*.wav")))
        else:
            self.audio_paths.extend(list(data_dir.glob("**/*.wav")))
        if debug:
            self.audio_paths = self.audio_paths[:1]

        if not self.lazy_loading:
            self.data = []
            for audio_path in tqdm(self.audio_paths,
                                   desc="Loading data samples into memory"):
                self.data.append(self.load(audio_path))

    def __len__(self):
        return len(self.audio_paths)

    def load(self, audio_path):
        """
        Load audio & corresponding labels into main memory
        """
        audio, sample_rate = torchaudio.load(str(audio_path))
        # convert to mono channel if necessary
        if audio.ndim > 1:
            audio = audio.mean(dim=0)
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)
            audio = audio.transpose(0, 1)
            stride = self.audio_transform[0].hop_length
        else:
            stride = 1

        midi_path = audio_path.with_suffix(".mid")
        note_df = parse_midi(midi_path)
        num_pitches = MIDI_MAX_PITCH - MIDI_MIN_PITCH + 1

        # quantize note labels
        onset_length_in_samples = sample_rate * self.onset_length_in_ms // 1000
        offset_length_in_samples = sample_rate * self.offset_length_in_ms // 1000
        num_steps_onset = onset_length_in_samples // stride
        num_steps_offset = offset_length_in_samples // stride
        num_steps_total = audio.shape[0]

        frame_labels = torch.zeros(num_steps_total, num_pitches, dtype=torch.uint8)
        velocity = torch.zeros(num_steps_total, num_pitches)

        for _, (onset, offset, pitch, vel) in note_df.iterrows():
            onset_start = int(round(onset * sample_rate / stride))
            onset_end = min(num_steps_total, onset_start + num_steps_onset)

            frame_end = int((round(offset * sample_rate / stride)))
            frame_end = min(num_steps_total, frame_end)
            offset_end = min(num_steps_total, frame_end + num_steps_offset)

            p = int(pitch) - MIDI_MIN_PITCH
            frame_labels[onset_start:onset_end, p] = 3  # onset
            frame_labels[onset_end:frame_end, p] = 2  # note frames
            frame_labels[frame_end:offset_end, p] = 1  # offset
            velocity[onset_start:offset_end, p] = vel / 128.
        return {
            "audio_path": str(audio_path.resolve()),
            "audio": audio,
            "frame_labels": frame_labels,
            "velocity": velocity,
            "hop_length": stride,
            "sample_rate": sample_rate,
            "num_steps_total": num_steps_total
        }

    def __getitem__(self, idx):
        if self.lazy_loading:
            data = self.load(self.audio_paths[idx])
        else:
            data = self.data[idx]
        audio = data["audio"]
        frame_labels = data["frame_labels"]
        velocity = data["velocity"]
        num_steps_total = data["num_steps_total"]

        if self.max_steps is not None:
            step_start = self.random.randint(num_steps_total - self.max_steps)
            step_end = step_start + self.max_steps

            audio = audio[step_start:step_end]
            frame_labels = frame_labels[step_start:step_end]
            velocity = velocity[step_start:step_end]

        onsets = (frame_labels == 3).float()
        offsets = (frame_labels == 1).float()
        frames = (frame_labels > 1).float()

        return {
            "audio_path": data["audio_path"],
            "audio": audio,
            "onsets": onsets,
            "offsets": offsets,
            "frames": frames,
            "velocity": velocity,
            "sample_rate": data["sample_rate"],
            "hop_length": data["hop_length"],
        }
