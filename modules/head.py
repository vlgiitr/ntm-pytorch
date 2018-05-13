from torch import nn


class NTMHead(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self._mode = mode

    @property
    def mode(self):
        return self._mode

    def forward(self, *inputs):
        pass

