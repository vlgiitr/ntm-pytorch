import torch
from torch import nn
import torch.nn.functional as F


class NTMController(nn.Module):

    def __init__(self, input_size, state_size, output_size, read_data_size):
        super().__init__()
        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size
        self.read_data_size = read_data_size

        self.controller_net = nn.LSTMCell(input_size, state_size)
        self.out_net = nn.Linear(read_data_size, output_size)
        # nn.init.xavier_uniform_(self.out_net.weight)
        nn.init.kaiming_uniform_(self.out_net.weight)
        self.h_state = torch.zeros([1, state_size])
        self.c_state = torch.zeros([1, state_size])
        # nn.init.kaiming_uniform_(self.h_state)
        # nn.init.kaiming_uniform_(self.c_state)
        # self.register_buffer('h_bias', torch.zeros([1, state_size]))
        # self.register_buffer('c_bias', torch.zeros([1, state_size]))
        self.reset()

    def forward(self, in_data, prev_reads):
        x = torch.cat([in_data] + prev_reads, dim=-1)
        self.h_state, self.c_state = self.controller_net(
            x, (self.h_state, self.c_state))
        return self.h_state, self.c_state

    def output(self, read_data):
        complete_state = torch.cat([self.h_state] + read_data, dim=-1)
        output = F.sigmoid(self.out_net(complete_state))
        return output

    def reset(self, batch_size=1):
        self.h_state = torch.zeros([batch_size, self.state_size])
        self.c_state = torch.zeros([batch_size, self.state_size])
        nn.init.kaiming_uniform_(self.h_state)
        nn.init.kaiming_uniform_(self.c_state)
        # self.h_state = self.h_bias.clone().repeat(batch_size, 1)
        # self.c_state = self.c_bias.clone().repeat(batch_size, 1)
