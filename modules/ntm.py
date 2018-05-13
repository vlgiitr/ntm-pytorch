from torch import nn

from .controller import NTMController
from .head import NTMHead
from .memory import NTMMemory


class NTM(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 controller_size,
                 memory_units,
                 memory_unit_size,
                 num_heads):
        super().__init__()
        self.controller = NTMController(input_size, output_size, controller_size)
        self.memory = NTMMemory(memory_units, memory_unit_size)
        self.heads = []
        for head in range(num_heads):
            self.heads += [NTMHead('r'), NTMHead('w')]

    @classmethod
    def init_from_modules(cls, controller, memory, heads):
        cls.controller = controller
        cls.memory = memory
        cls.heads = heads
        return cls

    def forward(self, *inputs):
        pass
