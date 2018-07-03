import json

import torch
from torch import nn, optim

from ntm import NTM
from ntm.datasets import CopyDataset
from ntm.args import get_parser


args = get_parser().parse_args()

# ----------------------------------------------------------------------------
# -- initialize dataset, model, criterion and optimizer
# ----------------------------------------------------------------------------

# changed task_json in args.py  from tasks/copy.py to ntm/tasks/copy.py
task_params = json.load(open(args.task_json))
dataset = CopyDataset(task_params)

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

for iter in range(args.num_iters):
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
    loss.backward()
    # clips gradient in the range [-10,10]. Again there is a slight but
    # insignificant deviation from the paper where they are clipped to (-10,10)
    nn.utils.clip_grad_value_(ntm.parameters(), 10)
    optimizer.step()

    binary_output = out.clone()
    binary_output.apply_(lambda x: 0 if x < 0.5 else 1)

    # sequence prediction error is calculted in bits per sequence
    error = torch.sum(torch.abs(binary_output - target))

    # logging
    if iter % 1000 == 0:
        print('Iteration: %d\tLoss: %.2f\tError in bits per sequence: %.2f' %
              (iter, loss, error))
