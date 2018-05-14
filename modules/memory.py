import torch
import torch.nn.functional as F
from torch import nn


class NTMMemory(nn.Module):
    def __init__(self, memory_units, memory_unit_size):
        super().__init__()
        self.n = memory_units
        self.m = memory_unit_size
        self.memory = torch.Tensor(1, self.n, self.m)
        self.reset()

    def forward(self, *inputs):
        pass

    def content_addressing(self, key, key_strength):
        """Perform content based addressing of memory based on key vector.
        Calculate cosine similarity between key vector and each unit of memory,
        finally obtain a softmax distribution out of it. These are normalized
        content weights according to content based addressing.

        Refer *Section 3.3.1* and *Figure 2* in the paper.

        Parameters
        ----------
        key : torch.Tensor
            The key vector (a.k.a. query vector) emitted by a read/write head.
            ``(batch_size, memory_unit_size)``

        key_strength : torch.Tensor
            A scalar weight (a.k.a. beta) multiplied for shaping the softmax
            distribution.
            ``(zero-dimensional)``

        Returns
        -------
        content_weights : torch.Tensor
            Normalized content-addressed weights vector. k-th element of this
            vector represents the amount of k-th memory unit to be read from or
            written to.
            ``(batch_size, memory_units)``
        """

        # view key with three dimensions as memory, add dummy dimension
        key = key.view(-1, 1, self.m)

        # calculate similarity along last dimension (memory_unit_size)
        similarity = F.cosine_similarity(key, self.memory, dim=-1)
        content_weights = F.softmax(key_strength * similarity, dim=1)
        return content_weights

    def reset(self, batch_size=1):
        self.memory = torch.Tensor(batch_size, self.n, self.m)
        nn.init.kaiming_uniform(self.memory)
