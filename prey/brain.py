import torch
from torch import nn
from torch.nn import LSTM, RNN
from math import sqrt


if torch.cuda.is_available():
    print("CUDA available. Neural calculations run on GPU.")
    device = torch.device("cuda:0")
else:
    print("CUDA unavailable. Neural calculations run on CPU.")
    device = torch.device("cpu")


class RecurrentNetwork(nn.Module):
    def __init__(self, n_features, recurrent_hidden, recurrent_stack):
        super().__init__()

        self.recurrent_layer = LSTM(n_features, recurrent_hidden, recurrent_stack, batch_first=True)
        self.linear = nn.Linear(recurrent_hidden, 1)
        self.double()
        self.to(device)

    def forward(self, x):
        x, _ = self.recurrent_layer(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x

    @staticmethod
    def preprocess_input(tensor: torch.tensor, normalization="max_distance", box_size=None):
        if normalization == "max_distance":
            tensor[:, :, 0] /= tensor[:, :, 0].max(1, keepdim=True)[0]

        elif normalization == "box_size":
            tensor[:, :, 0] /= (sqrt(2) * box_size)
