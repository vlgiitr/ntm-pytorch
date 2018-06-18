import torch
from torch import nn, optim

from ntm import NTM
from ntm.datasets import CopyDataset
from ntm.args import get_parser


args = get_parser().parse_args()
print(args)
