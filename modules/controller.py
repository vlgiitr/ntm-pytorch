import torch
from torch import nn


class NTMController(nn.Module):
    def __init__(self, input_size, output_size, weighting_size):
        super().__init__()
        self.net = nn.LSTMCell(input_size, weighting_size)

        # key is long term state (c) and output is short term state (h)
        # key is used as a query vector by heads for attention
        self.key = torch.Tensor(1, weighting_size)
        self.output = torch.Tensor(1, output_size)
        self.reset()

    def forward(self, x):
        self.output, self.key = self.net(x, (self.output, self.key))
        return self.output.squeeze(0), self.key.squeeze(0)

    def reset(self):
        nn.init.kaiming_uniform_(self.key)
        nn.init.kaiming_uniform_(self.output)
