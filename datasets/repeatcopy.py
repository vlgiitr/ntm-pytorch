import torch
from torch.utils.data import Dataset
from torch.distributions.binomial import Binomial


class RepeatCopyDataset(Dataset):
    """A Dataset class to generate random examples for the repeat copy task.
    Each sequence has a random length between `min_seq_len` and `max_seq_len`.
    Each vector in the sequence has a fixed length of `seq_width`. The last
    vector is a delimeter flag denoting end of sequence.

    Along with a delimiter flag, the input sequence also contains a channel 
    for number of repetitions. The input representing the repeat number is 
    normalised to have mean zero and variance one.

    For the target sequence, each sequence is repeated given number of times 
    followed by a delimiter flag.
    """

    def __init__(self, seq_width=8, min_seq_len=1, max_seq_len=10, min_repeat=1, max_repeat=10):
        """Initialize a dataset instance for repeat copy task.

        Arguments
        ---------
        seq_width : int, optional
            Width of the target sequence.
        min_seq_len : int, optional
            Minimum length of the target sequence.
        max_seq_len : int, optional
            Maximum length of the target sequence.
        min_repeat : int, optional
            Minimum number of repetitions.
        max_repeat : int, optional
            Maximum number of repetitions.
        """
        self.seq_width = seq_width
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.min_repeat = min_repeat
        self.max_repeat = max_repeat

    def normalise(self, rep):
        rep_mean = (self.max_repeat - self.min_repeat)/2
        rep_var = (((self.max_repeat - self.repeat_min + 1) ** 2)-1)/12
        rep_std = torch.sqrt(rep_var)
        return (rep-rep_mean)/rep_std

    def __len__(self):
        # sequences are generated randomly so this does not matter
        # set a sufficiently large size for data loader to sample mini-batches
        return 65536

    def __getitem__(self, idx):
        # idx only acts as a counter while generating batches.
        seq_len = torch.randint(
            self.min_seq_len, self.max_seq_len, (1,), dtype=torch.long).item()
        rep = torch.randint(
            self.min_repeat, self.max_repeat, (1,), dtype=torch.long).item()
        prob = 0.5 * torch.ones([seq_len, self.seq_width], dtype=torch.float64)
        seq = Binomial(1, prob).sample()

        # fill in input sequence, one bit longer and wider than target
        # it is zero-padded upto maximum length after delimiter
        input_seq = torch.zeros([self.max_seq_len + 2, self.seq_width + 2])
        input_seq[:seq_len, :self.seq_width] = seq
        input_seq[seq_len, self.seq_width] = 1.0  # delimiter
        input_seq[seq_len + 1, self.seq_width + 1] = self.normalise(rep)

        # Fill in similarly and zero-pad upto maximum possible length for
        # target sequence(max_seq_len*max_repeat) after delimiter.
        target_seq = torch.zeros(
            [self.max_seq_len * self.max_repeat + 1, self.seq_width + 1])
        target_seq[:seq_len * rep, :self.seq_width] = seq.repeat(rep, 1)
        target_seq[seq_len*rep, self.seq_width] = 1.0  # delimiter

        return {'input': input_seq, 'target': target_seq}
