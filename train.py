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

# ----------------------------------------------------------------------------
# -- initialize datasets, model, criterion and optimizer
# ----------------------------------------------------------------------------

args.task_json = 'ntm/tasks/copy.json'
"""
args.task_json = 'ntm/tasks/repeatcopy.json'
args.task_json = 'ntm/tasks/associative.json'
args.task_json = 'ntm/tasks/ngram.json'
args.task_json = 'ntm/tasks/prioritysort.json'
"""

task_params = json.load(open(args.task_json))

dataset = CopyDataset(task_params)
"""
dataset = RepeatCopyDataset(task_params)
dataset = AssociativeDataset(task_params)
dataset = NGram(task_params)
dataset = PrioritySort(task_params)
"""

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

criterion = nn.BCELoss()
# As the learning rate is task specific, the argument can be moved to json file
optimizer = optim.RMSprop(ntm.parameters(),
                          lr=args.lr,
                          alpha=args.alpha,
                          momentum=args.momentum)

cur_dir = os.getcwd()
PATH = os.path.join(cur_dir, 'saved_model.pt')

# ----------------------------------------------------------------------------
# -- basic training loop
# ----------------------------------------------------------------------------
losses = []
errors = []
for iter in tqdm(range(args.num_iters)):
# for iter in tqdm(range(50000)):
    optimizer.zero_grad()
    ntm.reset()

    data = dataset[iter]
    input, target = data['input'], data['target']
    out = torch.zeros(target.size())

    # -------------------------------------------------------------------------
    # loop for other tasks
    # -------------------------------------------------------------------------
    for i in range(input.size()[0]):
        # to maintain consistency in dimensions as torch.cat was throwing error
        in_data = torch.unsqueeze(input[i], 0)
        ntm(in_data)

    # passing zero vector as the input while generating target sequence
    in_data = torch.unsqueeze(torch.zeros(input.size()[1]), 0)
    for i in range(target.size()[0]):
        out[i] = ntm(in_data)
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # loop for NGram task
    # -------------------------------------------------------------------------
    """
    for i in range(task_params['seq_len'] - 1):
        in_data = input[i].view(1, -1)
        ntm(in_data)
        target_data = torch.zeros([1]).view(1, -1)
        out[i] = ntm(target_data)
    """
    # -------------------------------------------------------------------------

    loss = criterion(out, target)
    losses.append(loss.item())
    loss.backward()
    # clips gradient in the range [-10,10]. Again there is a slight but
    # insignificant deviation from the paper where they are clipped to (-10,10)
    nn.utils.clip_grad_value_(ntm.parameters(), 10)
    optimizer.step()

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
#-------------------------
#-------------------------
#-------------------------

for idx in range(args.num_iters):
	data = dataset[idx] # data is a dictionary returned by __getitem__ function
	input, target = data['input'], data['target']

	optimizer.zero_grad()
	net.reset()
	# Loop over the entire sequence length
	for i in range(input.size()[0])
		ntm(input[i])

	out = Variable(torch.zeros(target.size()))

	# No input is given while reading the output
	for j in range(target.size()[0])
		out[j] = ntm()

	loss = criterion(out, target)
	loss.backward()
	optimizer.step()

	out = out.clone().data
	binary_out = out.apply_(lambda x: 1 if x > 0.5 else 0)

	cost = torch.sum(torch.abs(binary_out - target))
	

	if (idx % 1000 == 0):
		print(f'Iteration: {idx}, Loss:{loss.data[0]:.2f}, Cost: {cost:.2f}')

