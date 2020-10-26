import pandas as pd
import mido


def parse_midi(path):
    def pop_note(pitch, offset, active_notes):
        row = active_notes.pop(pitch)
        row["pitch"] = pitch
        row["offset"] = offset
        return row

    midi_file = mido.MidiFile(path)

    active_notes = {}
    current_time = 0
    rows = []
    sustain = False
    track = midi_file.tracks[0]
    tempo = track[0].tempo

    for i, msg in enumerate(track):
        current_time += mido.tick2second(msg.time, midi_file.ticks_per_beat, tempo)
        # sustain event
        if msg.type == "control_change" and msg.control == 64:
            sustain = msg.value >= 64
            # if sustain goes off, pop finished but sustained notes
            if not sustain:
                sustained_pitches = [pitch for pitch, note in active_notes.items()
                                     if note["sustain"]]
                for pitch in sustained_pitches:
                    rows.append(pop_note(pitch, current_time, active_notes))
        # note onset event
        elif msg.type == "note_on":
            # if same note is played again during sustain,
            # pop it out and insert a new one
            if msg.note in active_notes:
                rows.append(pop_note(msg.note, current_time, active_notes))

            active_notes[msg.note] = {
                "onset": current_time,
                "velocity": msg.velocity,
                "sustain": False
            }
        # note offset event
        elif msg.type == "note_off":
            if sustain:
                active_notes[msg.note]["sustain"] = True
            else:
                rows.append(pop_note(msg.note, current_time, active_notes))
    df = pd.DataFrame(rows, columns=["onset", "offset", "pitch", "velocity"])
    df.sort_values(["onset", "offset"], inplace=True, ignore_index=True)
    return df
