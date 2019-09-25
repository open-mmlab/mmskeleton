# The based unit of graph convolutional networks.
# This is the original implementation for ST-GCN papers.

import torch
import torch.nn as nn


class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


class Gconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        if isinstance(kernel_size, int):
            gcn_kernel_size = kernel_size
            feature_dim = 0
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            gcn_kernel_size = kernel_size[0]
            cnn_kernel_size = [1] + kernel_size[1:]
            feature_dim = len(kernel_size) - 1
        else:
            raise ValueError(
                'The type of kernel_size should be int, list or tuple.')

        if feature_dim == 1:
            self.conv = nn.Conv1d(in_channels,
                                  out_channels * gcn_kernel_size,
                                  kernel_size=cnn_kernel_size)
        elif feature_dim == 2:
            pass
        elif feature_dim == 3:
            pass
        elif feature_dim == 0:
            pass
        else:
            raise ValueError(
                'The length of kernel_size should be 1, 2, 3, or 4')

    def forward(self, X, A):
        pass