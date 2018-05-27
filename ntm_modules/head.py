import torch
import torch.nn.functional as F
from torch import nn


class NTMHead(nn.Module):
    def __init__(self, mode, controller_size, key_size):
        super().__init__()
        self.mode = mode
        self.key_size = key_size

        # all the fc layers to produce scalars for memory addressing
        self.key_fc = nn.Linear(controller_size, key_size)
        self.key_strength_fc = nn.Linear(controller_size, 1)

        # these five fc layers cannot be in controller class
        # since each head has its own parameters and scalars
        self.interpolation_gate_fc = nn.Linear(controller_size, 1)
        self.shift_weighting_fc = nn.Linear(controller_size, 3)
        self.sharpen_factor_fc = nn.Linear(controller_size, 1)

    def forward(self, data, controller_state, prev_weights, memory):
        """Accept previous state (weights and memory) and controller state,
        produce attention weights for current read or write operation.
        Weights are produced by content-based and location-based addressing.

        Refer *Figure 2* in the paper to see how weights are produced.

        The head returns current weights useful for next time step, while
        it reads from or writes to ``memory`` based on its mode, using the
        ``data`` vector. ``data`` is filled and returned for read mode,
        returned as is for write mode.

        Refer *Section 3.1* for read mode and *Section 3.2* for write mode.

        Parameters
        ----------
        data : torch.Tensor
            Depending upon the mode, this data vector will be used by memory.
            ``(batch_size, memory_unit_size)``

        controller_state : torch.Tensor
            Long-term state of the controller.
            ``(batch_size, controller_size)``

        prev_weights : torch.Tensor
            Attention weights from previous time step.
            ``(batch_size, memory_units)``

        memory : ntm_modules.NTMMemory
            Memory Instance. Read write operations will be performed in place.

        Returns
        -------
        current_weights, data : torch.Tensor, torch.Tensor
            Current weights and data (filled in read operation else as it is).
            ``(batch_size, memory_units), (batch_size, memory_unit_size)``
        """

        # all these are marked as "controller outputs" in Figure 2
        key = self.key_fc(controller_state)
        b = self.key_strength_fc(controller_state)
        g = self.interpolation_gate_fc(controller_state)
        s = self.shift_weighting_fc(controller_state)
        y = self.sharpen_factor_fc(controller_state)

        content_weights = memory.content_addressing(key, b)

        # location-based addressing - interpolate, shift, sharpen
        interpolated_weights = g * content_weights + (1 - g) * prev_weights
        shifted_weights = self._circular_conv1d(interpolated_weights, s)
        current_weights = F.softmax(shifted_weights ** y)

        if self.mode == 'r':
            data = memory.read(current_weights)
        elif self.mode == 'w':
            memory.write(current_weights, data)
        else:
            raise ValueError("mode must be read ('r') or write('w')")
        return current_weights, data

    @staticmethod
    def _circular_conv1d(in_tensor, weights):
        # pad left with elements from right, and vice-versa
        batch_size = weights.size(0)
        pad = int((weights.size(1) - 1) / 2)
        in_tensor = torch.cat([in_tensor[-pad:], in_tensor, in_tensor[pad:]])
        out_tensor = F.conv1d(in_tensor.view(batch_size, 1, -1),
                              weights.view(batch_size, 1, -1))
        out_tensor = out_tensor.view(batch_size, -1)
        return out_tensor
