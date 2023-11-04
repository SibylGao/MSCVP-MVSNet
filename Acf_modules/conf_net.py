import torch
import torch.nn as nn
import torch.nn.functional as F

from .Acf_module import conv_bn_relu

class ConfidenceEstimation(nn.Module):
    """
        Args:
            in_planes, (int): usually cost volume used to calculate confidence map with $in_planes$ in Channel Dimension
            batchNorm, (bool): whether use batch normalization layer, default True
        Inputs:
            cost, (Tensor): cost volume in (BatchSize, in_planes, Height, Width) layout
        Outputs:
            confCost, (Tensor): in (BatchSize, 1, Height, Width) layout
    """

    def __init__(self, in_planes, batchNorm=True):
        super(ConfidenceEstimation,self).__init__()
        
        self.in_planes = in_planes
        self.sec_in_planes = int(self.in_planes//3)
        self.sec_in_planes  = self.sec_in_planes if self.sec_in_planes > 0 else 1

        self.conf_net = nn.Sequential(conv_bn_relu(self.in_planes, self.sec_in_planes, bias=False,batchNorm=batchNorm),
                                        nn.Conv2d(self.sec_in_planes, 1, 1, 1, 0, bias=False))    #two layers

    def forward(self, cost):
        assert cost.shape[1] == self.in_planes

        confCost = self.conf_net(cost)

        return confCost