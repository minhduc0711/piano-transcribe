import torch
import torch.nn as nn
from tqdm import tqdm


class BiLSTM(nn.Module):
    INFERENCE_CHUNK_LENGTH = 512

    def __init__(self, in_feats: int, out_feats: int):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(in_feats, out_feats, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.lstm(x)[0]

    # def forward(self, x):
    #     if self.training:
    #         return self.lstm(x)[0]
    #     else:
    #         # evaluation mode: support for longer sequences that do not fit in memory
    #         batch_size, sequence_length, input_features = x.shape
    #         hidden_size = self.lstm.hidden_size
    #         num_directions = 2 if self.lstm.bidirectional else 1

    #         h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
    #         c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
    #         output = torch.zeros(batch_size, sequence_length, num_directions * hidden_size, device=x.device)

    #         # forward direction
    #         print("forward lstm")
    #         slices = range(0, sequence_length, self.INFERENCE_CHUNK_LENGTH)
    #         for start in tqdm(slices):
    #             end = start + self.INFERENCE_CHUNK_LENGTH
    #             output[:, start:end, :], (h, c) = self.lstm(x[:, start:end, :], (h, c))

    #         # reverse direction
    #         if self.lstm.bidirectional:
    #             print("reverse lstm")
    #             h.zero_()
    #             c.zero_()

    #             for start in reversed(slices):
    #                 end = start + self.INFERENCE_CHUNK_LENGTH
    #                 result, (h, c) = self.lstm(x[:, start:end, :], (h, c))
    #                 output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

    #         return output
