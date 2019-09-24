# The based unit of graph convolutional networks.

import torch
import torch.nn as nn


class GraphConvND(nn.Module):
    def __init__(self, N, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, padding_mode):

        graph_kernel_size = kernel_size[0]
        graph_stride = stride[0]
        graph_padding = padding[0]
        graph_dilation = dilation[0]

        if graph_stride != 1 or graph_padding != 0 or graph_dilation != 1:
            raise NotImplementedError

        if N == 1:
            conv_type = nn.Conv1d
            self.einsum_func = 'nkcv,kvw->ncw'
        elif N == 2:
            conv_type = nn.Conv2d
            self.einsum_func = 'nkcvx,kvw->ncwx'
        elif N == 3:
            conv_type = nn.Conv3d
            self.einsum_func = 'nkcvxy,kvw->ncwxy'

        self.out_channels = out_channels
        self.graph_kernel_size = graph_kernel_size
        self.conv = conv_type(in_channels,
                              out_channels * graph_kernel_size,
                              kernel_size=[1] + kernel_size[1:],
                              stride=[1] + stride[1:],
                              padding=[0] + padding[1:],
                              dilation=[1] + dilation[1:],
                              groups=groups,
                              bias=bias,
                              padding_mode=padding_mode)

    def forward(self, x, graph):

        # graph is an adjacency matrix
        if graph.dim() == 2:
            A, out_graph = self.normalize_adjacency_matrix(graph)

        # graph is a weight matrix
        elif graph.dim() == 3:
            A, out_graph = graph, None

        else:
            raise ValueError('input[1].dim() should be 2 or 3.')

        x = self.conv(x)
        x = x.view((x.size(0), self.graph_kernel_size, self.out_channels) +
                   x.size()[2:])
        x = torch.einsum(self.einsum_func, (x, A))

        return x.contiguous(), out_graph

    def normalize_adjacency_matrix(self, graph):
        raise NotImplementedError
        return None, graph


class GraphConv(GraphConvND):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):

        super().__init__(1, in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)


class GraphConv2D(GraphConvND):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=(1, 1),
                 padding=(0, 0),
                 dilation=(1, 1),
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):

        super().__init__(2, in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)


class GraphConv3D(GraphConvND):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=(1, 1, 1),
                 padding=(0, 0, 0),
                 dilation=(1, 1, 1),
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):

        super().__init__(3, in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)