import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def velocity_loss(velocity_pred, velocity_true, onset_true):
    """
    Basically a MSE loss, but only takes into account velocity on note onsets
    """
    n = onset_true.sum()
    if n.item() == 0:
        return 0
    else:
        return (1/n) * (onset_true * (velocity_pred - velocity_true) ** 2).sum()


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
    def __init__(self, in_feats: int, out_feats: int = 88, hidden_feats: int = 512,
                 lr=6e-4):
        super(OnsetsAndFrames, self).__init__()
        self.lr = lr

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
        onset_true, frame_true, velocity_true = \
            batch["onsets"], batch["frames"], batch["velocity"]
        onset_pred, frame_pred, velocity_pred = self(audio_feats)

        onset_loss = F.binary_cross_entropy(onset_pred, onset_true)
        frame_loss = F.binary_cross_entropy(frame_pred, frame_true)
        vel_loss = velocity_loss(velocity_pred, velocity_true, onset_true)
        self.log("loss/onset", onset_loss)
        self.log("loss/frame", frame_loss)
        self.log("loss/velocity", vel_loss)

        return onset_loss + frame_loss + vel_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
