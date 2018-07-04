import torch
from torch.utils.data import Dataset
from torch.distributions.uniform import Uniform
from torch.distributions.binomial import Binomial


class PrioritySort(Dataset):
    """A Dataset class to generate random examples for priority sort task.

    In the input sequence, each vector is generated randomly along with a
    scalar priority rating. The priority is drawn uniformly from the range
    [-1,1) and is provided on a separate input channel.

    The target contains the binary vectors sorted according to their priorities
    """

    def __init__(self, task_params):
        """ Initialize a dataset instance for the priority sort task.

        Arguments
        ---------
        task_params : dict
                A dict containing parameters relevant to priority sort task.
        """
        self.seq_width = task_params["seq_width"]
        self.input_seq_len = task_params["input_seq_len"]
        self.target_seq_len = task_params["target_seq_len"]

    def __len__(self):
        # sequences are generated randomly so this does not matter
        # set a sufficiently large size for data loader to sample mini-batches
        return 65536

    def __getitem__(self, idx):
        # idx only acts as a counter while generating batches.
        prob = 0.5 * torch.ones([self.input_seq_len,
                                 self.seq_width], dtype=torch.float64)
        seq = Binomial(1, prob).sample()
        # Extra input channel for providing priority value
        input_seq = torch.zeros([self.input_seq_len, self.seq_width + 1])
        input_seq[:self.input_seq_len, :self.seq_width] = seq

        # torch's Uniform function draws samples from the half-open interval
        # [low, high) but in the paper the priorities are drawn from [-1,1].
        # This minor difference is being ignored here as supposedly it doesn't
        # affects the task.
        priority = Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
        for i in range(self.input_seq_len):
            input_seq[i, self.seq_width] = priority.sample()

        sorted, _ = torch.sort(input_seq, 0, descending=True)
        target_seq = sorted[:self.target_seq_len, :self.seq_width]

        return {'input': input_seq, 'target': target_seq}
