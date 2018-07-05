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

#Use this command to load the whole model rather than just paramters
#This works even when you don't initialise the model as it directly imports from the file.

#ntm= torch.load('filename.pt')

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
#Load pre trained model parameters and optimizer
#For this you have to initialise the model because it only saves and loads model parameters.

#state= torch.load('filename.pt')
#ntm.load_state_dict(state['state_dict'])
#optimizer.load_state_dict(state['optimizer'])

losses = []
errors = []
for iter in tqdm(range(args.num_iters)):
    optimizer.zero_grad()
    ntm.reset()

    data = dataset[iter]
    input, target = data['input'], data['target']

    # for i in range(1):
    for i in range(input.size()[0]):
        # to maintain consistency in dimensions as torch.cat was throwing error
        in_data = torch.unsqueeze(input[i], 0)
        ntm(in_data)

    # passing zero vector as the input while generating target sequence
    in_data = torch.unsqueeze(torch.zeros(input.size()[1]), 0)
    out = torch.zeros(target.size())
    # for i in range(1):
    for i in range(target.size()[0]):
        out[i] = ntm(in_data)
        # out = ntm(in_data)
    # print(out)
    # print(target)

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

    # logging
    if iter % 200 == 0:
        print('Iteration: %d\tLoss: %.2f\tError in bits per sequence: %.2f' %
              (iter, np.mean(losses), np.mean(errors)))
        # print(out, target)
	#Save checkpoint for saving model parameters and 
        #Command to save model parameters as a .pt file.

        #state = {
        #'epoch': iter,
        #'state_dict': ntm.state_dict(),
        #'optimizer': optimizer.state_dict(),
        #}

        #torch.save(state,'filename.pt')
        
        #Command to save the whole model at once along with all parameters.
        #torch.save(ntm,'filename.pt')

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

