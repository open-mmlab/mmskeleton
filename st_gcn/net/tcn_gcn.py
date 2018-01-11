import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from net import Unit2D, conv_init, import_class

default_backbone = [(64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128,
                                                            2), (128, 128, 1),
                    (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)]


class unit_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 use_global_bn=False,
                 kernel_size=1,
                 stride=1,
                 mask_learning=False):
        super(unit_gcn, self).__init__()
        self.V = A.size()[-1]
        self.A = Variable(
            A.clone(), requires_grad=False).view(-1, self.V, self.V)
        self.in_channels = in_channels
        self.mask_learning = mask_learning
        self.out_channels = out_channels

        self.num_A = self.A.size()[0]
        self.conv_list = nn.ModuleList([
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=(kernel_size, 1),
                padding=((kernel_size - 1) / 2, 0),
                stride=(stride, 1)) for i in range(self.num_A)
        ])

        if mask_learning:
            self.mask = nn.Parameter(torch.ones(self.A.size()))
        if use_global_bn:
            self.global_bn = nn.BatchNorm1d(self.out_channels * 25)  #NTU
        else:
            self.bn = nn.BatchNorm2d(self.out_channels)

        self.relu = nn.ReLU()
        self.use_global_bn = use_global_bn

        # initialize
        for conv in self.conv_list:
            conv_init(conv)

    def forward(self, x):

        N, C, T, V = x.size()
        self.A = self.A.cuda()
        A = self.A
        if self.mask_learning:
            A = A * self.mask

        for i, a in enumerate(A):
            xa = x.view(-1, V).mm(a).view(N, C, T, V)

            if i == 0:
                y = self.conv_list[i](xa)
            else:
                y = y + self.conv_list[i](xa)

        if self.use_global_bn:
            y = y.permute(0, 1, 3, 2).contiguous().view(
                N, self.out_channels * V, T)
            y = self.global_bn(y)
            y = y.view(N, self.out_channels, V, T).permute(0, 1, 3, 2)
        else:
            y = self.bn(y)

        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 A,
                 kernel_size=9,
                 stride=1,
                 dropout=0.5,
                 use_global_bn=False,
                 mask_learning=False):
        super(TCN_GCN_unit, self).__init__()
        half_out_channel = out_channel / 2
        self.A = A
        self.V = A.size()[-1]
        self.C = in_channel

        self.gcn1 = unit_gcn(
            in_channel,
            out_channel,
            A,
            use_global_bn=use_global_bn,
            mask_learning=mask_learning)
        self.tcn1 = Unit2D(
            out_channel,
            out_channel,
            kernel_size=kernel_size,
            dropout=dropout,
            stride=stride)
        if (in_channel != out_channel) or (stride != 1):
            self.down1 = Unit2D(
                in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.down1 = None

    def forward(self, x):
        # N, C, T, V = x.size()
        x = self.tcn1(self.gcn1(x)) + (x if (self.down1 is None) else
                                       self.down1(x))
        return x


class Model(nn.Module):
    def __init__(self,
                 channel,
                 num_class,
                 window_size,
                 num_point,
                 graph=None,
                 graph_args=dict(),
                 backbone_config=None,
                 use_data_bn=False,
                 use_global_bn=False,
                 temporal_kernel_size=9,
                 mask_learning=False):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
            self.A = torch.from_numpy(self.graph.A).float().cuda(0)

        self.num_class = num_class
        self.use_data_bn = use_data_bn
        self.data_bn = nn.BatchNorm1d(channel * num_point)

        kwargs = dict(
            A=self.A,
            mask_learning=mask_learning,
            use_global_bn=use_global_bn,
            kernel_size=temporal_kernel_size)
        unit = TCN_GCN_unit

        # backbone
        if backbone_config is None:
            backbone_config = default_backbone
        self.backbone = nn.ModuleList([
            unit(in_c, out_c, stride=stride, **kwargs)
            for in_c, out_c, stride in backbone_config
        ])
        backbone_in_c = backbone_config[0][0]
        backbone_out_c = backbone_config[-1][1]
        backbone_out_t = window_size
        backbone = []
        for in_c, out_c, stride in backbone_config:
            backbone.append(unit(in_c, out_c, stride=stride, **kwargs))
            if backbone_out_t % stride == 0:
                backbone_out_t = backbone_out_t // stride
            else:
                backbone_out_t = backbone_out_t // stride + 1
        self.backbone = nn.ModuleList(backbone)

        # head
        self.gcn0 = unit_gcn(
            channel,
            backbone_in_c,
            self.A,
            mask_learning=mask_learning,
            use_global_bn=use_global_bn)
        self.tcn0 = Unit2D(backbone_in_c, backbone_in_c, kernel_size=9)

        # tail
        self.person_bn = nn.BatchNorm1d(backbone_out_c)
        self.gap_size = backbone_out_t
        self.fcn = nn.Conv1d(
            backbone_out_c, num_class, kernel_size=self.gap_size)
        # self.fcn = nn.Conv1d(256, num_class, kernel_size=1)
        conv_init(self.fcn)

    def forward(self, x):
        N, C, T, V, M = x.size()  # Batch
        M_dim_bn = False

        # data bn
        if self.use_data_bn:
            if M_dim_bn:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            else:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)

            x = self.data_bn(x)
            # to (N*M, C, T, V)
            x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
                N * M, C, T, V)
        else:
            # from (N, C, T, V, M) to (N*M, C, T, V)
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        # model
        x = self.gcn0(x)
        x = self.tcn0(x)
        for  m in self.backbone:
            x = m(x)

        # V pooling
        x = F.avg_pool2d(x, kernel_size=(1, V))
        x = x.view(N, M, x.size(1), x.size(2))

        # M pooling
        x = x.mean(dim=1)

        # T pooling
        # x = F.avg_pool1d(x, kernel_size=self.gap_size)

        # C fcn
        x = self.fcn(x)
        x = F.avg_pool1d(x, x.size()[2:])
        x = x.view(N, self.num_class)

        return x
