import torch
from torch import nn


class NTMController(nn.Module):
    def __init__(self, input_size, output_size, state_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.state_size = state_size

        self.controller_net = nn.LSTMCell(input_size, state_size)

        # fc layer transforming short-term state (h) to output vector
        self.out_fc = nn.Linear(state_size, output_size)

        self.h_state = torch.Tensor(1, state_size)
        self.c_state = torch.Tensor(1, state_size)
        self.reset()

    def forward(self, x):
        self.h_state, self.c_state = self.controller_net(x, (self.h_state, self.c_state))
        output = self.out_fc(self.h_state)
        return output, self.c_state

    def reset(self, batch_size=1):
        self.h_state = torch.Tensor(batch_size, self.state_size)
        self.c_state = torch.Tensor(batch_size, self.state_size)
        nn.init.kaiming_uniform_(self.h_state)
        nn.init.kaiming_uniform_(self.c_state)
