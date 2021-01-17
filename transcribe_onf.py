from argparse import ArgumentParser
from pathlib import Path

from mido import MidiFile, MidiTrack, Message
from mir_eval.util import hz_to_midi

from src.data.audio import load_audio, onf_transform
from src.models.onsets_and_frames import OnsetsAndFrames

ap = ArgumentParser()
ap.add_argument("--audio", type=str, required=True)
ap.add_argument("--checkpoint", type=str, required=True)
ap.add_argument("--tempo", type=int, default=2)
args = ap.parse_args()

# prepare audio input
print("Processing audio file...")
audio = load_audio(
    args.audio,
    audio_transform=onf_transform,
    new_sample_rate=16000,
)
audio.unsqueeze_(0)

# load model and perform a forward pass
print("Loading model...")
model = OnsetsAndFrames.load_from_checkpoint(args.checkpoint, in_feats=229)
print("Performing forward pass...")
onset_pred, frame_pred, velocity_pred = model(audio)
p_est, i_est, v_est, _ = model.extract_notes(
    onset_pred.squeeze(),
    frame_pred.squeeze(),
    velocity_pred.squeeze(),
    sample_rate=16000,
    hop_length=512,
)

# convert model's output into MIDI format
file = MidiFile()
track = MidiTrack()
file.tracks.append(track)
ticks_per_second = file.ticks_per_beat * args.tempo

events = []
for pitch, (time_on, time_off), velocity in zip(p_est, i_est, v_est):
    events.append({"type": "on", "pitch": pitch, "time": time_on, "velocity": velocity})
    events.append(
        {"type": "off", "pitch": pitch, "time": time_off, "velocity": velocity}
    )
events.sort(key=lambda row: row["time"])

last_tick = 0
for event in events:
    current_tick = int(event["time"] * ticks_per_second)
    velocity = min(int(event["velocity"] * 127), 127)
    pitch = int(round(hz_to_midi(event["pitch"])))
    track.append(
        Message(
            "note_" + event["type"],
            note=pitch,
            velocity=velocity,
            time=current_tick - last_tick,
        )
    )
    last_tick = current_tick

output_dir = Path("demo/")
output_dir.mkdir(exist_ok=True)
out_midi_path = output_dir / Path(args.audio).with_suffix(".midi").name
file.save(out_midi_path)
print("Output MIDI saved to", out_midi_path)
