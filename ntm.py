from torch import nn

from ntm_modules.controller import NTMController
from ntm_modules.head import NTMHead
from ntm_modules.memory import NTMMemory


class NTM(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 controller_size,
                 memory_units,
                 memory_unit_size,
                 num_heads):
        super().__init__()
        self.controller = NTMController(input_size, output_size,
                                        state_size=controller_size)
        self.memory = NTMMemory(memory_units, memory_unit_size)
        self.heads = []
        for head in range(num_heads):
            self.heads += [
                NTMHead('r', controller_size, key_size=memory_unit_size),
                NTMHead('w', controller_size, key_size=memory_unit_size)
            ]

    @classmethod
    def init_from_modules(cls, controller, memory, heads):
        cls.controller = controller
        cls.memory = memory
        cls.heads = heads
        return cls

    def forward(self, in_data):
        pass
