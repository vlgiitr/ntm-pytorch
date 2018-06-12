import torch
from torch.utils.data import Dataset
from torch.distributions.binomial import Binomial


class copy_data(Dataset):
    """Generates random sequences of random vectors for copy task.
    To account for the delimiter flag, the input sequence length as well 
    width is one more than the target sequence."""

    def __init__(self, batch_size, seq_width=8, min_seq_len=1, max_seq_len=20):
        self.seq_width = seq_width
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def __len__(self):
        pass

    def __getitem__(self, idx):
        # idx only acts as a counter while generating batches.
        seq_len = torch.randint(
            self.min_seq_len, self.max_seq_len, (1,), dtype=torch.long)
        prob = 0.5*torch.ones([self.batch_size, seq_len,
                               self.seq_width], dtype=torch.float64)
        rand_seq = Binomial(1, prob)
        copy_seq = rand_seq.sample()
        input_batch = torch.zeros(
            self.batch_size, seq_len + 1, self.seq_width + 1)
        input_batch[:, :seq_len, :self.seq_width] = copy_seq
        input_batch[:, seq_len, self.seq_width] = 1.0
        target_batch = copy_seq.clone()
        return ([input_batch, target_batch])
