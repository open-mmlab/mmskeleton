import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


class Unit2D(nn.Module):
    def __init__(self,
                 D_in,
                 D_out,
                 kernel_size,
                 stride=1,
                 dim=2,
                 dropout=0,
                 bias=True):
        super(Unit2D, self).__init__()
        pad = int((kernel_size - 1) / 2)
        if dim == 2:
            self.conv = nn.Conv2d(
                D_in,
                D_out,
                kernel_size=(kernel_size, 1),
                padding=(pad, 0),
                stride=(stride, 1),
                bias=bias)
        elif dim == 3:
            self.conv = nn.Conv2d(
                D_in,
                D_out,
                kernel_size=(1, kernel_size),
                padding=(0, pad),
                stride=(1, stride),
                bias=bias)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(D_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # initialize
        conv_init(self.conv)

    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.bn(self.conv(x)))
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
