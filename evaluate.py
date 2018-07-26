import json
import os
import matplotlib.pyplot as plt

import torch
from torch import nn

from ntm import NTM
from ntm.datasets import CopyDataset, RepeatCopyDataset, AssociativeDataset, NGram, PrioritySort
from ntm.args import get_parser

args = get_parser().parse_args()

args.task_json = 'ntm/tasks/copy.json'
'''
args.task_json = 'ntm/tasks/repeatcopy.json'
args.task_json = 'ntm/tasks/associative.json'
args.task_json = 'ntm/tasks/ngram.json'
args.task_json = 'ntm/tasks/prioritysort.json'
'''

task_params = json.load(open(args.task_json))
criterion = nn.BCELoss()

# ---Evaluation parameters for Copy task---
task_params['min_seq_len'] = 20
task_params['max_seq_len'] = 120

'''
# ---Evaluation parameters for RepeatCopy task---
# (Sequence length generalisation)
task_params['min_seq_len'] = 10
task_params['max_seq_len'] = 20
# (Number of repetition generalisation)
task_params['min_repeat'] = 10
task_params['max_repeat'] = 20

# ---Evaluation parameters for AssociativeRecall task---
task_params['min_item'] = 6
task_params['max_item'] = 20

# For NGram and Priority sort task parameters need not be changed.
'''

dataset = CopyDataset(task_params)
'''
dataset = RepeatCopyDataset(task_params)
dataset = AssociativeDataset(task_params)
dataset = NGram(task_params)
dataset = PrioritySort(task_params)
'''

args.saved_model = 'saved_model_copy.pt'
'''
args.saved_model = 'saved_model_repeatcopy.pt'
args.saved_model = 'saved_model_associative.pt'
args.saved_model = 'saved_model_ngram.pt'
args.saved_model = 'saved_model_prioritysort.pt'
'''

cur_dir = os.getcwd()
PATH = os.path.join(cur_dir, args.saved_model)
# PATH = os.path.join(cur_dir, 'saved_models/saved_model_copy_500000.pt')
# ntm = torch.load(PATH)

"""
For the Copy task, input_size: seq_width + 2, output_size: seq_width
For the RepeatCopy task, input_size: seq_width + 2, output_size: seq_width + 1
For the Associative task, input_size: seq_width + 2, output_size: seq_width
For the NGram task, input_size: 1, output_size: 1
For the Priority Sort task, input_size: seq_width + 1, output_size: seq_width
"""

ntm = NTM(input_size=task_params['seq_width'] + 2,
          output_size=task_params['seq_width'],
          controller_size=task_params['controller_size'],
          memory_units=task_params['memory_units'],
          memory_unit_size=task_params['memory_unit_size'],
          num_heads=task_params['num_heads'])

ntm.load_state_dict(torch.load(PATH))

# -----------------------------------------------------------------------------
# --- evaluation
# -----------------------------------------------------------------------------
ntm.reset()
data = dataset[0]  # 0 is a dummy index
input, target = data['input'], data['target']
out = torch.zeros(target.size())

# -----------------------------------------------------------------------------
# loop for other tasks
# -----------------------------------------------------------------------------
for i in range(input.size()[0]):
    # to maintain consistency in dimensions as torch.cat was throwing error
    in_data = torch.unsqueeze(input[i], 0)
    ntm(in_data)

# passing zero vector as the input while generating target sequence
in_data = torch.unsqueeze(torch.zeros(input.size()[1]), 0)
for i in range(target.size()[0]):
    out[i] = ntm(in_data)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# loop for NGram task
# -----------------------------------------------------------------------------
'''
for i in range(task_params['seq_len'] - 1):
    in_data = input[i].view(1, -1)
    ntm(in_data)
    target_data = torch.zeros([1]).view(1, -1)
    out[i] = ntm(target_data)
'''
# -----------------------------------------------------------------------------

loss = criterion(out, target)

binary_output = out.clone()
binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)

# sequence prediction error is calculted in bits per sequence
error = torch.sum(torch.abs(binary_output - target))

# ---logging---
print('Loss: %.2f\tError in bits per sequence: %.2f' % (loss, error))

# ---saving results---
result = {'output': binary_output, 'target': target}
