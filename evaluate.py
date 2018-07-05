import json
from tqdm import tqdm
import numpy as np
import os

import torch
from torch import nn, optim

from ntm import NTM
from ntm.datasets import CopyDataset, RepeatCopyDataset, AssociativeDataset, NGram, PrioritySort
from ntm.args import get_parser

args = get_parser().parse_args()
args.task_json = 'ntm/tasks/copy.json'
task_params = json.load(open(args.task_json))
criterion = nn.BCELoss()

task_params['min_seq_len'] = 20
task_params['max_seq_len'] = 120

dataset = CopyDataset(task_params)

cur_dir = os.getcwd()
PATH = os.path.join(cur_dir, 'saved_model.pt')
ntm = torch.load(PATH)
"""
ntm = NTM(input_size=task_params['seq_width'] + 2,
          output_size=task_params['seq_width'],
          controller_size=task_params['controller_size'],
          memory_units=task_params['memory_units'],
          memory_unit_size=task_params['memory_unit_size'],
          num_heads=task_params['num_heads'])

ntm.load_state_dict(torch.load(PATH))
"""

# -----------------------------------------------------------------------------
# --- evaluation loop
# -----------------------------------------------------------------------------
losses = []
errors = []
for iter in tqdm(range(1000)):
    data = dataset[iter]
    input, target = data['input'], data['target']
    out = torch.zeros(target.size())

    for i in range(input.size()[0]):
        # to maintain consistency in dimensions as torch.cat was throwing error
        in_data = torch.unsqueeze(input[i], 0)
        ntm(in_data)

    # passing zero vector as the input while generating target sequence
    in_data = torch.unsqueeze(torch.zeros(input.size()[1]), 0)
    for i in range(target.size()[0]):
        out[i] = ntm(in_data)

    loss = criterion(out, target)
    losses.append(loss.item())

    binary_output = out.clone()
    binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)

    # sequence prediction error is calculted in bits per sequence
    error = torch.sum(torch.abs(binary_output - target))
    errors.append(error.item())

    # ---logging---
    if iter % 200 == 0:
        print('Iteration: %d\tLoss: %.2f\tError in bits per sequence: %.2f' %
              (iter, np.mean(losses), np.mean(errors)))
        losses = []
        errors = []
