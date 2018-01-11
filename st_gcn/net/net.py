import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math



class Unit2D(nn.Module):
    def __init__(self, D_in, D_out, kernel_size, stride=1, dim=2, dropout=0):
        super(Unit2D, self).__init__()
        if dim == 2:
            self.conv = nn.Conv2d(
                D_in,
                D_out,
                kernel_size=(kernel_size, 1),
                padding=((kernel_size - 1) / 2, 0),
                stride=(stride, 1))
        elif dim == 3:
            self.conv = nn.Conv2d(
                D_in,
                D_out,
                kernel_size=(1, kernel_size),
                padding=(0, (kernel_size - 1) / 2),
                stride=(1, stride))
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(D_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # initialize
        conv_init(self.conv)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.dropout(x)
        return x


def conv_init(module):
    # he_normal
    n = module.out_channels
    for k in module.kernel_size:
        n *= k
    module.weight.data.normal_(0, math.sqrt(2. / n))


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
