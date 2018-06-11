import random

import numpy as np
import torch


def generate_data(seq_width, min_seq_len, max_seq_len, batch_size, num_batches):
    """Generates random sequences of random vectors for copy task.
    To account for the delimiter flag, the input sequence length as well width
    is one more than the target sequence.

    Returns
    -------
    An array with each element representing a batch. Each element is a tuple
    containing the batch number, input and target tensors.
    Input tensor: ``(batch_size, seq_len + 1, seq_width + 1)``
    Target tensor: ``(batch_size, seq_len, seq_width)``
  
    """
    data = []
    for batch_num in range(num_batches):
        seq_len = random.randint(min_seq_len, max_seq_len)
        seq = np.random.binomial(1, 0.5, (batch_size, seq_len, seq_width))
        seq = torch.from_numpy(seq)
        input = torch.zeros(batch_size, seq_len + 1, seq_width + 1)
        input[:, :seq_len, :seq_width] = seq
        input[:, seq_len, seq_width] = 1.0
        target = seq.clone()
        data.append((batch_num+1, input, target))
    return data
