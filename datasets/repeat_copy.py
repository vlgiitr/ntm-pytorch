import torch
from torch.utils.data import Dataset
from torch.distributions.binomial import Binomial



class repeat_copy_data(Dataset):
    """Generates input and target for repeat copy task.
    The input contains random-length sequences of random binary vectors,
    followed by a scalar value indicating the desired number of copies.
    As mentioned in the paper, the input representing the repeat number
    is normalised to have mean zero and variance one."""
    
    def __init__(self, batch_size, seq_width=8, min_seq_len=1, max_seq_len=10, min_repeat=1, max_repeat=10):
        self.seq_width = seq_width
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.min_repeat=min_repeat
        self.max_repeat=max_repeat
        
    def normalise(self, rep):
        rep_mean=(self.max_repeat - self.min_repeat)/2
        rep_var=(((self.max_repeat - self.repeat_min + 1) ** 2)-1)/12
        rep_std=torch.sqrt(rep_var)
        return (rep-rep_mean)/rep_std
    
    def __len__(self):
        pass

    def __getitem__(self, idx):
        # idx only acts as a counter while generating batches.
        seq_len = torch.randint(
            self.min_seq_len, self.max_seq_len, (1,), dtype=torch.long)
        rep = torch.randint(
            self.min_repeat, self.max_repeat, (1,), dtype=torch.long)
        prob = 0.5*torch.ones([self.batch_size, seq_len,
                               self.seq_width], dtype=torch.float64)
        rand_seq = Binomial(1, prob)
        repeat_copy_seq = rand_seq.sample()
        input_batch = torch.zeros(
            self.batch_size, seq_len + 2, self.seq_width + 2)
        input_batch[:, :seq_len, :self.seq_width] = repeat_copy_seq
        input_batch[:, seq_len, self.seq_width] = 1.0
        input_batch[:, seq_len + 1, self.seq_width + 1] = self.normalise(rep.item())
        target_batch=torch.zeros(self.batch_size, seq_len * rep + 1, self.seq_width + 1)
        target_batch[:, :seq_len * rep, :self.seq_width] = repeat_copy_seq.clone().repeat(1,rep,1)
        target_batch[:, seq_len*rep, self.seq_width]=1.0
        
        return ([input_batch, target_batch])


