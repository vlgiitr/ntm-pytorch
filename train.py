import json
from tqdm import tqdm
import numpy as np

import torch
from torch import nn, optim

from ntm import NTM
from ntm.datasets import CopyDataset, RepeatCopyDataset, AssociativeDataset
from ntm.args import get_parser


args = get_parser().parse_args()

# ----------------------------------------------------------------------------
# -- initialize dataset, model, criterion and optimizer
# ----------------------------------------------------------------------------

# changed task_json in args.py  from tasks/copy.py to ntm/tasks/copy.py
args.task_json = 'ntm/tasks/associative.json'
task_params = json.load(open(args.task_json))
# dataset = CopyDataset(task_params)
dataset = AssociativeDataset(task_params)

ntm = NTM(input_size=task_params['seq_width'] + 2,
          output_size=task_params['seq_width'],
          controller_size=task_params['controller_size'],
          memory_units=task_params['memory_units'],
          memory_unit_size=task_params['memory_unit_size'],
          num_heads=task_params['num_heads'])

criterion = nn.BCELoss()
# As the learning rate is task specific, the argument can be moved to json file
# fixed typo RMSProp->RMSprop
optimizer = optim.RMSprop(ntm.parameters(),
                          lr=args.lr,
                          alpha=args.alpha,
                          momentum=args.momentum)

# ----------------------------------------------------------------------------
# -- basic training loop
# ----------------------------------------------------------------------------

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

