import torch
import torch.nn.functional as F
from torch import nn


class NTMMemory(nn.Module):

    def __init__(self, memory_units, memory_unit_size):
        super().__init__()
        self.n = memory_units
        self.m = memory_unit_size
        self.memory = torch.zeros([1, self.n, self.m])
        nn.init.kaiming_uniform_(self.memory)
        # layer to learn bias values for memory reset
        self.memory_bias_fc = nn.Linear(1, self.n * self.m)
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

        # calculate similarity along last dimension (self.m)
        similarity = F.cosine_similarity(key, self.memory, dim=-1)
        content_weights = F.softmax(key_strength * similarity, dim=1)
        return content_weights

    def read(self, weights):
        """Read from memory through soft attention over all locations.

        Refer *Section 3.1* in the paper for read mechanism.

        Parameters
        ----------
        weights : torch.Tensor
            Attention weights emitted by a read head.
            ``(batch_size, memory_units)``

        Returns
        -------
        data : torch.Tensor
            Data read from memory weighted by attention.
            ``(batch_size, memory_unit_size)``
        """
        # expand and perform batch matrix mutliplication
        weights = weights.view(-1, 1, self.n)
        # (b, 1, self.n) x (b, self.n, self.m) -> (b, 1, self.m)
        data = torch.bmm(weights, self.memory).view(-1, self.m)
        return data

    def write(self, weights, data, erase=None):
        """Write to memory through soft attention over all locations.

        Refer *Section 3.2* in the paper for write mechanism.

        .. note::
            Erase and add mechanisms have been merged here.
            ``weights(erase) = (1 - weights(add))``

        Parameters
        ----------
        weights : torch.Tensor
            Attention weights emitted by a write head.
            ``(batch_size, memory_units)``

        data : torch.Tensor
            Data to be written to memory.
            ``(batch_size, memory_unit_size)``

        erase(optional) : torch.Tensor
            Extent of erasure to be performed on the memory unit.
            ``(batch_size, memory_unit_size)``
        """

        # make weights and write_data sizes same as memory
        weights = weights.view(-1, self.n, 1).expand(self.memory.size())
        data = data.view(-1, 1, self.m).expand(self.memory.size())
        self.memory = weights * data + (1 - weights) * self.memory
        # --(separate erase and add mechanism)
        # erase = erase.view(-1, 1, self.m).expand(self.memory.size())
        # self.memory = (1 - weights * erase) * self.memory
        # self.memory = weights * data + self.memory

    def reset(self, batch_size=1):
        # self.memory = torch.zeros([batch_size, self.n, self.m])
        # nn.init.kaiming_uniform_(self.memory)
        in_data = torch.tensor([[0.]])  # dummy input
        memory_bias = F.sigmoid(self.memory_bias_fc(in_data))
        self.memory = memory_bias.view(self.n, self.m).repeat(batch_size, 1, 1)
