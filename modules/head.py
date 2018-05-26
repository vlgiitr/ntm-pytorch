import torch
import torch.nn.functional as F
from torch import nn


class NTMHead(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, data, prev_weights, memory, controller_outputs):
        """Accept previous state (weights and memory) and controller outputs,
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

        prev_weights : torch.Tensor
            Attention weights from previous time step.
            ``(batch_size, memory_units)``

        memory : torch.Tensor
            An instance of `NTMMemory` containing the memory tensor.
            Read and write operations will be performed in place.

        controller_outputs : dict
            A dict returned by `NTMController.forward` method, contents of
            this dict are mentioned in *Figure 2*.

        Returns
        -------
        current_weights : torch.Tensor
            Produced by the content-based and location-based (interpolate, 
            shift, sharpen) addressing mechanisms.
            ``(batch_size, memory_units)``

        data : torch.Tensor
            Data which was provided as input, will be filled if in read mode.
            ``(batch_size, memory_unit_size)``
        """

        content_weights = memory.content_addressing(
            controller_outputs['key'], controller_outputs['key_strength'])

        g = controller_outputs['interpolation_gate']
        interpolated_weights = g * content_weights + (1 - g) * prev_weights

        s = controller_outputs['shift_weighting']
        shifted_weights = _circular_conv1d(interpolated_weights, s)

        gamma = controller_outputs['sharpen_factor']
        sharpened_weights = shifted_weights ** gamma
        current_weights = F.softmax(sharpened_weights)

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
