from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from mir_eval.util import midi_to_hz
from mir_eval.transcription import precision_recall_f1_overlap as eval_notes
from mir_eval.transcription_velocity import (
    precision_recall_f1_overlap as eval_notes_with_velocity,
)


def velocity_loss(velocity_pred, velocity_true, onset_true):
    """
    Basically a MSE loss, but only takes into account velocity on note onsets
    """
    n = onset_true.sum()
    if n.item() == 0:
        return 0
    else:
        return (1 / n) * (onset_true * (velocity_pred - velocity_true) ** 2).sum()


class ConvStack(nn.Module):
    def __init__(self, in_feats: int, out_feats: int):
        """
        Convolutional acoustic model, takes as input a tensor of shape
            (batch, channel, time, feature_dim)
        Args:
            in_feats: feature dim of input
            out_feats: feature dim of output
        """
        super(ConvStack, self).__init__()
        self.conv = nn.Sequential(
            # layer 0
            nn.Conv2d(1, out_feats // 16, (3, 3), padding=1),
            nn.BatchNorm2d(out_feats // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(out_feats // 16, out_feats // 16, (3, 3), padding=1),
            nn.BatchNorm2d(out_feats // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(out_feats // 16, out_feats // 8, (3, 3), padding=1),
            nn.BatchNorm2d(out_feats // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((out_feats // 8) * (in_feats // 4), out_feats), nn.Dropout(0.5)
        )

    def forward(self, x):
        # insert a channel dim
        x = torch.unsqueeze(x, 1)
        x = self.conv(x)
        # (batch, channel, time, feat_dim) -> (batch, time, channel * feat_dim)
        x = x.permute(0, 2, 1, 3).flatten(2)
        x = self.fc(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self, in_feats: int, out_feats: int):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(in_feats, out_feats, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.lstm(x)[0]


class OnsetsAndFrames(pl.LightningModule):
    def __init__(
        self,
        in_feats: int,
        out_feats: int = 88,
        hidden_feats: int = 512,
        lr=6e-4,
        lr_sched_step_size=10000,
        lr_sched_decay_rate=0.98,
    ):
        super(OnsetsAndFrames, self).__init__()
        self.lr = lr
        self.lr_sched_step_size = lr_sched_step_size
        self.lr_sched_decay_rate = lr_sched_decay_rate

        self.onset_stack = nn.Sequential(
            ConvStack(in_feats, hidden_feats),
            BiLSTM(hidden_feats, 128),
            # BiLSTM doubles the output dim so we divide output dim by 2
            nn.Linear(256, out_feats),
            nn.Sigmoid(),
        )
        self.frame_stack_1 = nn.Sequential(
            ConvStack(in_feats, hidden_feats),
            nn.Linear(hidden_feats, out_feats),
            nn.Sigmoid(),
        )
        self.frame_stack_2 = nn.Sequential(
            BiLSTM(out_feats * 2, 128), nn.Linear(256, out_feats), nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(in_feats, hidden_feats), nn.Linear(hidden_feats, out_feats)
        )

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.lr_sched_step_size, gamma=self.lr_sched_decay_rate
        )
        return [opt], [sched]

    def forward(self, x):
        onset_pred = self.onset_stack(x)
        frame_pred_pre = self.frame_stack_1(x)
        # not using frame predictions to optimize onset stack
        combined_pred = torch.cat([onset_pred, frame_pred_pre], dim=-1)
        frame_pred = self.frame_stack_2(combined_pred)
        velocity_pred = self.velocity_stack(x)
        return onset_pred, frame_pred, velocity_pred

    def training_step(self, batch, batch_idx):
        # unpacking data from batch
        audio_feats = batch["audio"]
        onset_true, frame_true, velocity_true = (
            batch["onsets"],
            batch["frames"],
            batch["velocity"],
        )
        onset_pred, frame_pred, velocity_pred = self(audio_feats)

        onset_loss = F.binary_cross_entropy(onset_pred, onset_true)
        frame_loss = F.binary_cross_entropy(frame_pred, frame_true)
        vel_loss = velocity_loss(velocity_pred, velocity_true, onset_true)
        total_loss = onset_loss + frame_loss + vel_loss

        self.log("train_loss/onset", onset_loss)
        self.log("train_loss/frame", frame_loss)
        self.log("train_loss/velocity", vel_loss)
        self.log("train_loss/total", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # unpacking data from batch
        audio_feats = batch["audio"]
        onset_true, frame_true, velocity_true = (
            batch["onsets"],
            batch["frames"],
            batch["velocity"],
        )
        onset_pred, frame_pred, velocity_pred = self(audio_feats)

        onset_loss = F.binary_cross_entropy(onset_pred, onset_true)
        frame_loss = F.binary_cross_entropy(frame_pred, frame_true)
        vel_loss = velocity_loss(velocity_pred, velocity_true, onset_true)
        total_loss = onset_loss + frame_loss + vel_loss

        self.log("val_loss/onset", onset_loss)
        self.log("val_loss/frame", frame_loss)
        self.log("val_loss/velocity", vel_loss)
        self.log("val_loss/total", total_loss)
        self.log("valid_loss", total_loss)

        # compute transcription metrics
        metrics = defaultdict(lambda: defaultdict(list))
        # needs to iterate over single samples
        for onset_est, frame_est, vel_est, onset_ref, frame_ref, vel_ref, sample_rate, hop_length \
                in zip(onset_pred, frame_pred,velocity_pred, onset_true, frame_true, velocity_true,
                       batch["sample_rate"], batch["hop_length"]):
            p_ref, i_ref, v_ref = self.extract_notes(
                onset_ref,
                frame_ref,
                vel_ref,
                sample_rate=sample_rate.item(),
                hop_length=sample_rate.item(),
            )
            p_est, i_est, v_est = self.extract_notes(
                onset_est,
                frame_est,
                vel_est,
                sample_rate=sample_rate.item(),
                hop_length=sample_rate.item(),
            )

            p, r, f1, o = eval_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
            metrics["note"]["precision"].append(p)
            metrics["note"]["recall"].append(r)
            metrics["note"]["f1"].append(f1)
            metrics["note"]["overlap"].append(o)

            p, r, f1, o = eval_notes(i_ref, p_ref, i_est, p_est)
            metrics["note_with_offsets"]["precision"].append(p)
            metrics["note_with_offsets"]["recall"].append(r)
            metrics["note_with_offsets"]["f1"].append(f1)
            metrics["note_with_offsets"]["overlap"].append(o)

            p, r, f1, o = eval_notes_with_velocity(
                i_ref, p_ref, v_ref, i_est, p_est, v_est, offset_ratio=None, velocity_tolerance=0.1
            )
            metrics["note_with_velocity"]["precision"].append(p)
            metrics["note_with_velocity"]["recall"].append(r)
            metrics["note_with_velocity"]["f1"].append(f1)
            metrics["note_with_velocity"]["overlap"].append(o)

            p, r, f1, o = eval_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1)
            metrics["note_with_offsets_and_velocity"]["precision"].append(p)
            metrics["note_with_offsets_and_velocity"]["recall"].append(r)
            metrics["note_with_offsets_and_velocity"]["f1"].append(f1)
            metrics["note_with_offsets_and_velocity"]["overlap"].append(o)

        self.log("val_metric/note/precision",
                 np.mean(metrics["note"]["precision"]))
        self.log("val_metric/note/recall",
                 np.mean(metrics["note"]["recall"]))
        self.log("val_metric/note/f1",
                 np.mean(metrics["note"]["f1"]))
        self.log("val_metric/note/overlap",
                 np.mean(metrics["note"]["overlap"]))

        # self.log("val_metric/note-with-offsets/precision", p)
        # self.log("val_metric/note-with-offsets/recall", r)
        # self.log("val_metric/note-with-offsets/f1", f1)
        # self.log("val_metric/note-with-offsets/overlap", o)

        # self.log("val_metric/note-with-velocity/precision", p)
        # self.log("val_metric/note-with-velocity/recall", r)
        # self.log("val_metric/note-with-velocity/f1", f1)
        # self.log("val_metric/note-with-velocity/overlap", o)

        # self.log("val_metric/note-with-offsets-and-velocity/precision", p)
        # self.log("val_metric/note-with-offsets-and-velocity/recall", r)
        # self.log("val_metric/note-with-offsets-and-velocity/f1", f1)
        # self.log("val_metric/note-with-offsets-and-velocity/overlap", o)

    def extract_notes(
        self,
        onsets,
        frames,
        velocity,
        sample_rate,
        hop_length,
        onset_threshold: int = 0.5,
        frame_threshold: int = 0.5,
    ):
        """
        Extract notes (pitch and onset-offset interval) from frame predictions.
        Make sure that a note is produced only when both an onset & one or more frames agree.
        Works on single sample only.

        Args:
            onsets, frames, velocity: tensor of shape (time, num_pitches)
        Returns:
            pitches: ndarray of shape (num_notes,)
            intervals: ndarray of shape (num_notes, 2)
            velocities: ndarray of shape (num_notes,)
        """
        # threshold the probabilities
        onsets = (onsets > onset_threshold).cpu().to(torch.int)
        frames = (frames > frame_threshold).cpu().to(torch.int)
        # clip velocity to [0, 1]
        velocity = torch.clamp(velocity, min=0, max=1)

        # squashing adjacent onsets
        onsets = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1
        # array of
        pitches = []
        # array of (n, 2) - onset and offset times
        intervals = []
        velocities = []

        for onset_t, onset_pitch in onsets.nonzero(as_tuple=False):
            onset = onset_t.item()
            pitch = onset_pitch.item()

            offset = onset
            while offset < onsets.shape[0] and (
                frames[offset, pitch].item() == 1 or onsets[offset, pitch].item() == 1
            ):
                offset += 1

            if offset > onset:
                pitches.append(pitch)
                intervals.append([onset, offset])
                velocities.append(velocity[onset, pitch].item())

        scale_factor = hop_length / sample_rate
        pitches = np.array([midi_to_hz(p + 21) for p in pitches])
        intervals = np.array(intervals) * scale_factor
        velocities = np.array(velocities)

        return pitches, intervals, velocities
