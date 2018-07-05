import torch
from torch.utils.data import Dataset
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli


class NGram(Dataset):
    """A Dataset class to generate random examples for the N-gram task.

    Each sequence is generated using a lookup table for n-Gram distribution
    probabilities. The lookup table contains 2**(n-1) numbers specifying the
    probability that the next bit will be one. The numbers represent all
    possible (n-1) length binary histories. The probabilities are independently
    drawn from Beta(0.5,0.5) distribution.

    The first 5 bits, for which insuffient context exists to sample from the
    table, are drawn i.i.d. from a Bernoulli distribution with p=0.5. The
    subsequent bits are drawn using probabilities from the table.
    """

    def __init__(self, task_params):
        """ Initialize a dataset instance for N-gram task.

        Arguments
        ---------
        task_params : dict
            A dict containing parameters relevant to N-gram task.
        """
        self.seq_len = task_params["seq_len"]
        self.n = task_params["N"]

    def __len__(self):
        # sequences are generated randomly so this does not matter
        # set a sufficiently large size for data loader to sample mini-batches
        return 65536

    def __getitem__(self, idx):
        # idx only acts as a counter while generating batches.
        beta_prob = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
        lookup_table = {}

        # generate probabilities for the lookup table. The key represents the
        # possible binary sequences.
        for i in range(2**(self.n - 1)):
            lookup_table[bin(i)[2:].rjust(self.n - 1, '0')] = beta_prob.sample()

        # generate input sequence
        input_seq = torch.zeros([self.seq_len])
        prob = Bernoulli(torch.tensor([0.5]))
        for i in range(self.n):
            input_seq[i] = prob.sample()
        for i in range(self.n - 1, self.seq_len):
            prev = input_seq[i - self.n + 1:i]
            prev = ''.join(map(str, map(int, prev)))
            prob = lookup_table[prev]
            input_seq[i] = Bernoulli(prob).sample()

        # As the task is to predict the next bit, the target sequence is a bit
        # shorter than the input.
        target_seq = input_seq[1:self.seq_len]

        return {'input': input_seq, 'target': target_seq}
