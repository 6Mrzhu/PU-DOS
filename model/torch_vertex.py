import torch
from torch import nn
from torch_nn import BasicConv, batched_index_select
from torch_edge import DenseDilatedKnnGraph, DilatedKnnGraph
import torch.nn.functional as F


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        return self.nn(torch.cat([x, x_j], dim=1))


def EdgeConv2d(x, edge_index):
    x_i = batched_index_select(x, edge_index[1])
    x_j = batched_index_select(x, edge_index[0])
    max_value = torch.cat([x_i, x_j - x_i], dim=1)
    return max_value

class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """

    def __init__(self, in_channels, out_channels,act, norm, bias):
        super(GraphConv2d, self).__init__()
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.ReLU())

    def forward(self, x, edge_index):
        x = EdgeConv2d(x, edge_index)
        x = self.conv1(x)
        x, _ = torch.max(x, -1, keepdim=False)
        return x



#静态图卷积
# class GraphConv2d(nn.Module):
#     """
#     Static graph convolution layer
#     """
#
#     def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
#         super(GraphConv2d, self).__init__()
#         if conv == 'edge':
#             self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
#         elif conv == 'mr':
#             self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
#         else:
#             raise NotImplementedError('conv:{} is not supported'.format(conv))
#
#     def forward(self, x, edge_index):
#         return self.gconv(x, edge_index)


class DynConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """

    def __init__(self, in_channels, out_channels, k=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DynConv2d, self).__init__(in_channels, out_channels, conv, act, norm)
        self.k = k
        self.d = dilation
        if knn == 'matrix':
            self.dilated_knn_graph = DenseDilatedKnnGraph(k, dilation, stochastic, epsilon)
        else:
            self.dilated_knn_graph = DilatedKnnGraph(k, dilation, stochastic, epsilon)

    def forward(self, x, edge_index=None):
        if edge_index is None:
            edge_index = self.dilated_knn_graph(x)
        return super(DynConv2d, self).forward(x, edge_index)


class PlainDynBlock2d(nn.Module):
    """
    Plain Dynamic graph convolution block
    """

    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(PlainDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x, edge_index=None):
        return self.body(x, edge_index)


class ResDynBlock2d(nn.Module):
    """
    Residual Dynamic graph convolution block
    """

    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, stochastic=False, epsilon=0.0, knn='matrix', res_scale=1):
        super(ResDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)
        self.res_scale = res_scale

    def forward(self, x, edge_index=None):
        return self.body(x, edge_index) + x * self.res_scale


class DenseDynBlock2d(nn.Module):
    """
    Dense Dynamic graph convolution block
    """

    def __init__(self, in_channels, out_channels=24, kernel_size=9, dilation=1, conv='edge',
                 act='relu', norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DenseDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, out_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x, edge_index=None):
        dense = self.body(x, edge_index)
        #return torch.cat((x, dense), 1)
        return dense
