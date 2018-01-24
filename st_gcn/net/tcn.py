import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .net import conv_init


class Unit_brdc(nn.Module):
    def __init__(self, D_in, D_out, kernel_size, stride=1, dropout=0):

        super(Unit_brdc, self).__init__()
        self.bn = nn.BatchNorm1d(D_in)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.conv = nn.Conv1d(
            D_in,
            D_out,
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
            stride=stride)

        # weight initialization
        conv_init(self.conv)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv(x)
        return x


class TCN_unit(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=9, stride=1):
        super(TCN_unit, self).__init__()
        self.unit1_1 = Unit_brdc(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            dropout=0.5,
            stride=stride)

        if in_channel != out_channel:
            self.down1 = Unit_brdc(
                in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.down1 = None

    def forward(self, x):
        x =self.unit1_1(x)\
                + (x if self.down1 is None else self.down1(x))
        return x


class Model(nn.Module):
    def __init__(self, channel, num_class, window_size, use_data_bn=False):
        super(Model, self).__init__()
        self.num_class = num_class
        self.use_data_bn = use_data_bn
        self.data_bn = nn.BatchNorm1d(channel)
        self.conv0 = nn.Conv1d(channel, 64, kernel_size=9, padding=4)
        conv_init(self.conv0)

        self.unit1 = TCN_unit(64, 64)
        self.unit2 = TCN_unit(64, 64)
        self.unit3 = TCN_unit(64, 64)
        self.unit4 = TCN_unit(64, 128, stride=2)
        self.unit5 = TCN_unit(128, 128)
        self.unit6 = TCN_unit(128, 128)
        self.unit7 = TCN_unit(128, 256, stride=2)
        self.unit8 = TCN_unit(256, 256)
        self.unit9 = TCN_unit(256, 256)
        self.bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.fcn = nn.Conv1d(256, num_class, kernel_size=1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)

        if self.use_data_bn:
            x = self.data_bn(x)

        x = self.conv0(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.unit3(x)
        x = self.unit4(x)
        x = self.unit5(x)
        x = self.unit6(x)
        x = self.unit7(x)
        x = self.unit8(x)
        x = self.unit9(x)
        x = self.bn(x)
        x = self.relu(x)

        x = F.avg_pool1d(x, kernel_size=x.size()[2])

        x = self.fcn(x)
        x = x.view(-1, self.num_class)

        return x
