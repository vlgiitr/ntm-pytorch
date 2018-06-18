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

task_params = json.load(open(args.task_json))
dataset = CopyDataset(task_params)

ntm = NTM(input_size=task_params['seq_width'] + 1,
		  output_size=task_params['seq_width'],
		  controller_size=task_params['controller_size'],
		  memory_units=task_params['memory_units'],
		  memory_unit_size=task_params['memory_unit_size'],
		  num_heads=task_params['num_heads'])

criterion = nn.BCELoss()
optimizer = optim.RMSProp(ntm.parameters(),
						  lr=args.lr,
						  alpha=args.alpha,
						  momentum=args.momentum)

# ----------------------------------------------------------------------------
# -- basic training loop 
# ----------------------------------------------------------------------------

# todo
