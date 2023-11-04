import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple

def consistent_padding_with_dilation(padding, dilation, dim=2):
    assert  dim==2 or dim==3, 'Convolution layer only support 2D and 3D'
    if dim == 2:
        padding = _pair(padding)
        dilation = _pair(dilation)
    else: # dim == 3
        padding = _triple(padding)
        dilation = _triple(dilation)

    padding = list(padding)
    for d in range(dim):
        padding[d] = dilation[d] if dilation[d] > 1 else padding[d]
    padding = tuple(padding)

    return padding, dilation

def conv_bn_relu(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, batchNorm=True):
    #kernel_size=3, stride=1, padding=1
    # padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=2)
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias),
            nn.ReLU(inplace=True),
        )

# class ConfidenceEstimation(nn.Module):
#     """
#         Args:
#             in_planes, (int): usually cost volume used to calculate confidence map with $in_planes$ in Channel Dimension
#             batchNorm, (bool): whether use batch normalization layer, default True
#         Inputs:
#             cost, (Tensor): cost volume in (BatchSize, in_planes, Height, Width) layout
#         Outputs:
#             confCost, (Tensor): in (BatchSize, 1, Height, Width) layout
#     """

#     def __init__(self, in_planes, batchNorm=True):
#         super(ConfidenceEstimation, self).__init__()

#         self.in_planes = in_planes
#         self.sec_in_planes = int(self.in_planes//3)    #取整除(192除3次？？？)
#         self.sec_in_planes  = self.sec_in_planes if self.sec_in_planes > 0 else 1

#         self.conf_net = nn.Sequential(conv_bn_relu(self.in_planes, self.sec_in_planes, 3, 1, 1, bias=False,batchNorm),
#                                       nn.Conv2d(self.sec_in_planes, 1, 1, 1, 0, bias=False))    #two layers

#     def forward(self, cost):
#         assert cost.shape[1] == self.in_planes

#         confCost = self.conf_net(cost)

#         return confCost