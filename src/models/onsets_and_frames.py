from mir_eval.util import midi_to_hz

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .lstm import BiLSTM
from src.eval import compute_note_metrics, compute_frame_metrics


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
        self.save_hyperparameters()
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
        sample_rate = batch["sample_rate"][0]
        hop_length = batch["hop_length"][0]

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

        # COMPUTING TRANSCRIPTION METRICS
        # needs to iterate over single samples, as metric computation does not support batching
        sample_metrics = []  # each elem is a metric dict for 1 sample
        for i in range(len(onset_pred)):
            p_est, i_est, v_est, final_frame_pred = self.extract_notes(
                onset_pred[i],
                frame_pred[i],
                velocity_pred[i],
                sample_rate=sample_rate,
                hop_length=hop_length
            )
            p_ref, i_ref, v_ref = batch["midi_labels"][i]

            sample_metrics.append({
                **compute_frame_metrics(final_frame_pred,
                                        frame_true[i].squeeze().cpu().numpy()),
                **compute_note_metrics(i_est, p_est, v_est, i_ref, p_ref, v_ref)
            })
        # average metrics over samples,
        # noting that all metric dicts have the same structure
        for metric_type in sample_metrics[0].keys():
            for cls_metric in sample_metrics[0][metric_type]:
                avg_val = np.mean([
                    metric[metric_type][cls_metric] for metric in sample_metrics
                ])
                self.log(f"val_metric/{metric_type}/{cls_metric}", avg_val)

    @staticmethod
    def extract_notes(
        onsets,
        frames,
        velocity,
        sample_rate: int,
        hop_length: int,
        onset_threshold: int = 0.5,
        frame_threshold: int = 0.5,
    ):
        """
        Extract notes (pitch and onset-offset interval) from frame predictions.
        Make sure that a note is produced only when both an onset & one or more frames agree.
        Works on single sample only.

        Args
        ----
        onsets, frames, velocity: tensor of shape (time, num_pitches)
        Returns
        -------
        pitches : array of shape (num_notes,)
            pitch of detected notes
        intervals : array of shape (num_notes, 2)
            onset & offset of detected notes
        velocities : array of shape (num_notes,)
            predicted velocities of detected notes
        final_frame_pred : array of shape (time, num_pitches)
            the final frame predictions after matching onsets & frames
        """
        # threshold the probabilities
        onsets = (onsets > onset_threshold).cpu().to(torch.int)
        frames = (frames > frame_threshold).cpu().to(torch.int)
        # clip velocity to [0, 1]
        velocity = torch.clamp(velocity, min=0, max=1)
        # squashing adjacent onsets
        onsets = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

        pitches = []
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

        final_frame_pred = np.zeros_like(onsets)
        for pitch, (onset, offset) in zip(pitches, intervals):
            final_frame_pred[onset:offset, pitch] = 1

        pitches = np.array([midi_to_hz(p + 21) for p in pitches])
        if len(intervals) > 0:
            scale_factor = hop_length / sample_rate
            intervals = np.array(intervals) * scale_factor
        else:
            intervals = np.empty((0, 2))
        velocities = np.array(velocities)

        return pitches, intervals, velocities, final_frame_pred
