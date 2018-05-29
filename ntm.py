import torch
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
        self.controller = NTMController(
            input_size, controller_size, output_size
            read_data_size=controller_size + num_heads * memory_unit_size)
        
        self.memory = NTMMemory(memory_units, memory_unit_size)
        self.num_heads = num_heads
        self.heads = nn.ModuleList([])
        for head in range(num_heads):
            self.heads += [
                NTMHead('r', controller_size, key_size=memory_unit_size),
                NTMHead('w', controller_size, key_size=memory_unit_size)
            ]

        self.prev_head_weights = []
        self.reset()

    def reset(self, batch_size=1):
        self.memory.reset(batch_size)
        self.controller.reset(batch_size)
        self.prev_head_weights = []
        for i in range(len(self.heads)):
            prev_weight = torch.zeros(batch_size, self.memory.n)
            self.prev_head_weights.append(prev_weight)

    def forward(self, in_data):
        pass
