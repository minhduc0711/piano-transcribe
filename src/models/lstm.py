import torch.nn as nn


class BiLSTM(nn.Module):
    """
    LSTM wrapper to be used in nn.Sequential()
    """
    def __init__(self, in_feats: int, out_feats: int):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(in_feats, out_feats, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.lstm(x)[0]
