import torch
from torch import nn


class NTMMemory(nn.Module):
    def __init__(self, memory_units, memory_unit_size):
        super().__init__()
        self.n = memory_units
        self.m = memory_unit_size

        self.memory = torch.Tensor(1, self.n, self.m)

    def forward(self, *inputs):
        pass
