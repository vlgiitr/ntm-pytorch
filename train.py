import json

import torch
from torch import nn, optim

from ntm import NTM
from ntm.datasets import CopyDataset, RepeatCopyDataset
from ntm.args import get_parser


args = get_parser().parse_args()

# ----------------------------------------------------------------------------
# -- initialize task, model, criterion and optimizer
# ----------------------------------------------------------------------------

task_json= "tasks/copy.json"   ### change the value from 'copy.json' to 'repeat_copy.json' for repeat copy task

task_params = json.load(open(args.task_json))

ntm = NTM(input_size=task_params['seq_width'] + 1,
		  output_size=task_params['seq_width'],
		  controller_size=task_params['controller_size'],
		  memory_units=task_params['memory_units'],
		  memory_unit_size=task_params['memory_unit_size'],
		  num_heads=task_params['num_heads'])

criterion = nn.BCELoss()

''' RMSProp and Adam are the two choices available as optimizers'''

'''RMSProp optimizer'''
optimizer = optim.RMSProp(ntm.parameters(),
						  lr=args.lr,
						  alpha=args.alpha,
						  momentum=args.momentum)

'''Adam optimizer''' 
'''
optimizer= optim.Adam(ntm.parameters(),
				      lr=args.lr,
				      betas=args.betas)

'''				      
# ----------------------------------------------------------------------------
# -- basic training loop 
# ----------------------------------------------------------------------------

def train_batch(dataset):
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

# ------------------------------------------------------------------------
# -- initialize task and train the model 
# ------------------------------------------------------------------------

''' Training for copy task '''

if task_json="tasks/copy.json":
	dataset = CopyDataset(task_params)
	train_batch(dataset)


''' Training for repeat copy task '''

elif task_json="tasks/repeat_copy.json":
	for _ in range(task_params['min_repeat'],task_params['max_repeat']):
		dataset = RepeatCopyDataset(task_params)
		train_batch(dataset)