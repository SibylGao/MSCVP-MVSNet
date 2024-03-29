# CVP-MVSNet
# By: Jiayu Yang
# Date: 2019-08-05

# Note: This file use part of the code from the following projects.
#       Thanks for the authors for the great code.
#       MVSNet: https://github.com/YoYo000/MVSNet
#       MVSNet_pytorch: https://github.com/xy-guo/MVSNet_pytorch

import torch
import torch.nn as nn
from models.modules import *
from Acf_modules.conf_net import ConfidenceEstimation
import os
from Acf_modules.local_soft_argmin import LocalSoftArgmin
from torch.autograd import Variable
# Debug:
# import pdb,time
# import matplotlib.pyplot as plt
# from verifications import *

def convtext(in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):
    
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )


# fast propagation
class FastPropagation(nn.Module):
    def __init__(self):
        super(FastPropagation,self).__init__()
        self.depth_propagation_layer = nn.Sequential(
            nn.Conv2d(17 , 16 * 2, 3, 1, padding=1),
            nn.BatchNorm2d(16 * 2, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(16 * 2, 9, 3, padding=1, bias=False)
        )
        self.unfold = nn.Unfold(kernel_size=(3,3), stride=1, padding=0)
    def forward(self, depth, img_feature, confidence):
        confidence = confidence.permute(1,0,2,3)
        img_feature_cat = torch.cat((img_feature,confidence),1)     #在第一个维度cat，用confidence来做guidance
        img_filter = self.depth_propagation_layer(img_feature_cat)
        img_filter = F.softmax(img_filter, dim=1)    #正则化之后转变为概率  [B,9,H,W]

        depth_pad = F.pad(depth, (1, 1, 1, 1), mode='replicate')    #填充depth，每两个参数为一组,填出来周围的一个边[]
        depth_pad = depth_pad.permute(1, 0, 2, 3)    #[B,1,H,W]
        depth_unfold = self.unfold(depth_pad)

        b, c, h, w = img_filter.size()
        prob = img_filter.view(b, 9, h*w)   # 3*3=9

        result_depth = torch.sum(depth_unfold * prob, dim=1)   #unfold操作只卷，不做积，加在一起相当于卷积的操作
        result_depth = result_depth.view(b, 1, h, w).squeeze(1)
        return result_depth

# 用depth+confidence作为输入，直接进行卷积求置信度加权下的depth map
class CNNPropagation(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积网络的定义，fast原网络里面也用了BN和relu，但是我把kernal改成了5,padding2
        self.confidencerefinenet = nn.Sequential(
            nn.Conv2d(18 , 16*2 , 5, 1, padding=2),
            # # 加快收敛速度，避免梯度消失；提升模型泛化能力，强行归一化会减少模型的非线性程度
            # # 有效识别一些对网络贡献不大的神经元，learning rate不能太大
            nn.BatchNorm2d(16*2, momentum=0.01),    
            nn.ReLU(inplace=True),          # 将所有负值设为0，同样有利于快速收敛，缺点和BN一样，learning rate太大会直接导致神经元失活
            nn.Conv2d(16*2, 1, 5, padding=2, bias=False)
        )
    def forward(self, depth,confidence,feature):
        depth = depth.permute(1, 0, 2, 3)    #[B,1,H,W]
        depth_cat = torch.cat((depth,confidence,feature),1)
        result_map = self.confidencerefinenet(depth_cat)
        result_map = result_map.squeeze(1)
        return result_map


# 用depth+confidence作为输入，直接进行卷积求置信度加权下的depth map
class CNNPropagation2(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积网络的定义，fast原网络里面也用了BN和relu，但是我把kernal改成了5,padding2
        self.confidencerefinenet = nn.Sequential(
            nn.Conv2d(2 , 16 , 5, 1, padding=2),
            # # 加快收敛速度，避免梯度消失；提升模型泛化能力，强行归一化会减少模型的非线性程度
            # # 有效识别一些对网络贡献不大的神经元，learning rate不能太大
            nn.BatchNorm2d(16, momentum=0.01),    
            nn.ReLU(inplace=True),          # 将所有负值设为0，同样有利于快速收敛，缺点和BN一样，learning rate太大会直接导致神经元失活
            nn.Conv2d(16, 1, 5, padding=2, bias=False)
        )
    def forward(self, depth,confidence,feature):
        depth = depth.permute(1, 0, 2, 3)    #[B,1,H,W]
        depth_cat = torch.cat((depth,confidence),1)
        result_map = self.confidencerefinenet(depth_cat)
        result_map = result_map.squeeze(1)
        return result_map



# Feature pyramid
class FeaturePyramid(nn.Module):
    def __init__(self):
        super(FeaturePyramid, self).__init__()
        self.conv0aa = conv(3, 64, kernel_size=3, stride=1)
        self.conv0ba = conv(64,64, kernel_size=3, stride=1)
        self.conv0bb = conv(64,64, kernel_size=3, stride=1)
        self.conv0bc = conv(64,32, kernel_size=3, stride=1)
        self.conv0bd = conv(32,32, kernel_size=3, stride=1)
        self.conv0be = conv(32,32, kernel_size=3, stride=1)
        self.conv0bf = conv(32,16, kernel_size=3, stride=1)
        self.conv0bg = conv(16,16, kernel_size=3, stride=1)
        self.conv0bh = conv(16,16, kernel_size=3, stride=1)

    def forward(self, img, scales=5):
        fp = []
        f = self.conv0aa(img)
        f = self.conv0bh(self.conv0bg(self.conv0bf(self.conv0be(self.conv0bd(self.conv0bc(self.conv0bb(self.conv0ba(f))))))))
        fp.append(f)
        for scale in range(scales-1):
            img = nn.functional.interpolate(img,scale_factor=0.5,mode='bilinear',align_corners=None).detach()
            f = self.conv0aa(img)
            f = self.conv0bh(self.conv0bg(self.conv0bf(self.conv0be(self.conv0bd(self.conv0bc(self.conv0bb(self.conv0ba(f))))))))
            fp.append(f)

        return fp

class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()

        self.conv0 = ConvBnReLU3D(4, 16, kernel_size=3, pad=1)
        self.conv0a = ConvBnReLU3D(16, 16, kernel_size=3, pad=1)

        self.conv1 = ConvBnReLU3D(16, 32,stride=2, kernel_size=3, pad=1)
        self.conv2 = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)
        self.conv2a = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)
        self.conv3 = ConvBnReLU3D(32, 64, kernel_size=3, pad=1)
        self.conv4 = ConvBnReLU3D(64, 64, kernel_size=3, pad=1)
        self.conv4a = ConvBnReLU3D(64, 64, kernel_size=3, pad=1)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=0, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.prob0 = nn.Conv3d(16, 1, 3, stride=1, padding=1)

    def forward(self, x):

        conv0 = self.conv0a(self.conv0(x))

        conv2 = self.conv2a(self.conv2(self.conv1(conv0)))
        conv4 = self.conv4a(self.conv4(self.conv3(conv2)))

        conv5 = conv2+self.conv5(conv4)

        conv6 = conv0+self.conv6(conv5)
        prob = self.prob0(conv6).squeeze(1)

        return prob

class network(nn.Module):
    def __init__(self, args):
        super(network, self).__init__()
        self.featurePyramid = FeaturePyramid()
        self.cost_reg_refine = CostRegNet()
        self.args = args

        self.num_depth = 48
        self.refine_num_depth = 8
        self.batchNorm = True

        self.confidence_coefficient = 1.0      #s
        self.confidence_init_value = 1.0       #epsilon
        self.focal_coefficient = 5.0         #alpha
        self.lamda_confidence = 8.0
        self.lamda_regression = 0.1

        self.cost_save_index = 0
        
        self.DTU_variance_scale = 13.3

        #用两个不同的confidence estimation网络是因为inplanes数目不一样
        self.conf_est_net = ConfidenceEstimation(in_planes=self.num_depth, batchNorm=self.batchNorm)
        self.refine_conf_est_net = ConfidenceEstimation(in_planes=self.refine_num_depth,batchNorm=self.batchNorm)
        self.test_local_softmax = LocalSoftArgmin(max_disp=self.num_depth, depth_interval=self.DTU_variance_scale, radius=2)

        self.Group = 4
        # # DPS refine network
        # self.dps_convs = nn.Sequential(
        #     convtext(17, 128, 3, 1, 1),
        #     convtext(128, 128, 3, 1, 2),
        #     convtext(128, 128, 3, 1, 4),
        #     convtext(128, 96, 3, 1, 8),
        #     convtext(96, 64, 3, 1, 16),
        #     convtext(64, 32, 3, 1, 1),
        #     convtext(32, 1, 3, 1, 1)
        # )
        
        ## BatchNorm
        # self.confidence_estimation = nn.Sequential(
        #     nn.Conv2d(1 , 16, 3, 1, padding=1),
        #     nn.BatchNorm2d(16 , momentum=0.01),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16 , 1, 3, padding=1, bias=False)
        # )
        
        ## 两层卷积，没有BN
        self.confidence_estimation = nn.Sequential(
            conv(1, 16, kernel_size=3, stride=1),
            conv(16, 1, kernel_size=3, stride=1)
        )

        # fast depth propagation
        # self.depth_propagation_layer = nn.Sequential(
        #     nn.Conv2d(16 , 16 * 2, 3, 1, padding=1),
        #     nn.BatchNorm2d(16 * 2, momentum=0.01),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16 * 2, 9, 3, padding=1, bias=False)
        # )
        # self.unfold = nn.Unfold(kernel_size=(3,3), stride=1, padding=0)
        self.fast_propagation = FastPropagation()
        self.CNNPropagation = CNNPropagation2()

    def forward(self, ref_img, src_imgs, ref_in, src_in, ref_ex, src_ex, \
                    depth_min, depth_max):

        ## Initialize output list for loss
        depth_est_list = []
        output = {}

        ## Feature extraction
        ref_feature_pyramid = self.featurePyramid(ref_img,self.args.nscale)

        src_feature_pyramids = []
        for i in range(self.args.nsrc):
            src_feature_pyramids.append(self.featurePyramid(src_imgs[:,i,:,:,:],self.args.nscale))

        # Pre-conditioning corresponding multi-scale intrinsics for the feature:
        ref_in_multiscales = conditionIntrinsics(ref_in,ref_img.shape,[feature.shape for feature in ref_feature_pyramid])
        src_in_multiscales = []
        for i in range(self.args.nsrc):
            src_in_multiscales.append(conditionIntrinsics(src_in[:,i],ref_img.shape, [feature.shape for feature in src_feature_pyramids[i]]))
        src_in_multiscales = torch.stack(src_in_multiscales).permute(1,0,2,3,4)

        ## Estimate initial coarse depth map
        depth_hypos = calSweepingDepthHypo(ref_in_multiscales[:,-1],src_in_multiscales[:,0,-1],ref_ex,src_ex,depth_min, depth_max)

        ref_volume = ref_feature_pyramid[-1].unsqueeze(2).repeat(1, 1, len(depth_hypos[0]), 1, 1)
        B, C, H, W = src_feature_pyramids[0][-1].shape
        V = self.args.nsrc
        # Kwea3 implementation as reference
        ref_volume = ref_volume.view(B, self.Group, C // self.Group, *ref_volume.shape[-3:])   #[B,4,8,H,W]
        volume_sum = 0

        warp_volumes = None
        for src_idx in range(self.args.nsrc):
            # warpped features
            warped_volume = homo_warping(src_feature_pyramids[src_idx][-1], ref_in_multiscales[:, -1],
                                         src_in_multiscales[:, src_idx, -1, :, :],
                                         ref_ex, src_ex[:, src_idx], depth_hypos)
            ## regular solution
            warped_volume = warped_volume.view(*ref_volume.shape)
            if self.args.mode == "train":
                # (B, Groups, C//Groups, D, h, w)
                volume_sum = volume_sum + warped_volume
            else:
                volume_sum += warped_volume
            del warped_volume

        ## Aggregate multiple feature volumes by Similarity
        ## The parameter V is a little different with that in implementation of Kwea123
        ## V = nsrc here, while V denotes the quantity of all the input images in the implementation of Kwea123.
        cost_volume = (volume_sum * ref_volume).mean(2).div_(V)    #这一步是group wise correlation
        # Regularize cost volume
        cost_reg = self.cost_reg_refine(cost_volume)    #[B,D,H,W]
        

#####################################################gao_added##########################################
        ####这一段计算出confidence和正太分布相关的variance
        confidences = []
        variances = []
        max_depthmap_list = []
        min_depthmap_list = []
        cost_reg_list = []

        confidence_costs = self.conf_est_net(cost_reg)     #(B,1,H,W)
        confidence = torch.sigmoid(confidence_costs)     #(B,1,H,W)
        confidences.append(confidence)

        #第一层train
        if self.args.mode =="train":

            prob_volume = F.softmax(cost_reg, dim=1)
            depth = depth_regression(prob_volume, depth_values=depth_hypos)
            depth_est_list.append(depth) 

            # confidence_costs = self.conf_est_net(cost_reg)     #(B,1,H,W)
            # confidence = torch.sigmoid(confidence_costs)     #(B,1,H,W)
            variance = self.confidence_coefficient * (1 - confidence) + self.confidence_init_value   ##s*(1-conf)+epsilon  正态分布的方差
            variance = variance * self.DTU_variance_scale
            variances.append(variance)
            cost_reg_list.append(cost_reg)
            # confidences.append(confidence)
#####################################################gao_added##########################################
        # # 第一层test
        # if self.args.mode == "test":
        #     confidence_costs = self.conf_est_net(cost_reg)     #(B,1,H,W)
        #     confidence = torch.sigmoid(confidence_costs)     #(B,1,H,W)
        #     confidences.append(confidence)
        #     cost_reg_list.append(cost_reg)
            
        # prob_volume = F.softmax(cost_reg, dim=1)
        # depth = depth_regression(prob_volume, depth_values=depth_hypos)
        # depth_est_list.append(depth)

        if self.args.mode == "test":
            # prob_volume = F.softmax(cost_reg, dim=1)
            # depth = depth_regression(prob_volume, depth_values=depth_hypos)
            # depth_est_list.append(depth)
            # first_layer_pro_test = prob_volume
            # del prob_volume
            # # 用local softmax
            # depth = self.test_local_softmax(cost_reg)    #[B,1,H,W]
            # depth = depth.squeeze(1)
            # depth_est_list.append(depth)
            prob_volume = F.softmax(cost_reg, dim=1)
            depth = depth_regression(prob_volume, depth_values=depth_hypos)
            depth_est_list.append(depth)

       
        depthmap_confidences = []
        ## Upsample depth map and refine along feature pyramid
        for level in range(self.args.nscale-2,-1,-1):    #是倒序金字塔

            # Upsample 
            depth_up = nn.functional.interpolate(depth[None,:],size=None,scale_factor=2,mode='bicubic',align_corners=None)  #[1,B,H,W]
            # depth_up = depth_up.squeeze(0)  #[B,H,W]

            # depthmap_confidence = self.confidence_estimation(depth_up.permute(1, 0, 2, 3))     ##是估计上采样之后的深度图的confidence,注意交换一下
            # depthmap_confidences.append(depthmap_confidence)
            
            # fast depth propagation
            # 对confidence也做了一个上采样
            confidence = nn.functional.interpolate(depth[None,:],size=None,scale_factor=2,mode='bicubic',align_corners=None)
            depth_up = self.fast_propagation(depth_up, ref_feature_pyramid[level],confidence)   #[b, h, w]

            # depth_up = self.CNNPropagation(depth_up,depthmap_confidence,ref_feature_pyramid[level])

            # Generate depth hypothesis
            depth_hypos = calDepthHypo(self.args,depth_up,ref_in_multiscales[:,level,:,:], src_in_multiscales[:,:,level,:,:],ref_ex,src_ex,depth_min, depth_max,level)

            cost_volume = proj_cost_AACVP(Group=self.Group, settings=self.args, ref_feature=ref_feature_pyramid[level],
                                          src_feature=src_feature_pyramids,
                                          level=level, ref_in=ref_in_multiscales[:, level, :, :],
                                          src_in=src_in_multiscales[:, :, level, :, :], ref_ex=ref_ex,
                                          src_ex=src_ex[:, :], depth_hypos=depth_hypos)

            cost_reg2 = self.cost_reg_refine(cost_volume)

            #加上了对confidence估计的部分
            confidence_costs = self.refine_conf_est_net(cost_reg2)     #(B,1,H,W)
            confidence = torch.sigmoid(confidence_costs)     #(B,1,H,W)
            confidences.append(confidence)

            #第二层test
            if self.args.mode == "test":
                # confidence_costs = self.refine_conf_est_net(cost_reg2)     #(B,1,H,W)
                # confidence = torch.sigmoid(confidence_costs)     #(B,1,H,W)
                # confidences.append(confidence)
                cost_reg_list.append(cost_reg2)

            if self.args.mode == "test":
                del cost_volume
            
            prob_volume = F.softmax(cost_reg2, dim=1)
            if self.args.mode == "test":
                del cost_reg2

            # Depth regression
            depth = depth_regression_refine(prob_volume, depth_hypos)

            depth_est_list.append(depth)

        # Photometric confidence  
        with torch.no_grad():
            num_depth = prob_volume.shape[1]    #[B,D,H,W]
            #这一部分平均池化之后取一个均值，再乘以4，表示周围四个像素点的和
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)   #[B,D,H,W]
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()   #[B,H,W]
            prob_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        if self.args.mode == "test":
            del prob_volume

        ## Return
        depth_est_list.reverse() # Reverse the list so that depth_est_list[0] is the largest scale.
        cost_reg_list.reverse()
        max_depthmap_list.reverse()
        min_depthmap_list.reverse()
        confidences.reverse()
        variances.reverse()
        depthmap_confidences.reverse()

        output["depth_est_list"] = depth_est_list
        output["prob_confidence"] = prob_confidence

        ## Return confidence gao for loss
        if self.args.mode == "train":
            output["confidences"] = confidences
            output["variances"] = variances
            output["cost_volumn"] = cost_reg_list
            output["max_depthmap"] = max_depthmap_list
            output["min_depthmap"] = min_depthmap_list
            output["depthmap_confidences"] = depthmap_confidences
            # output["depth_before_refine"] = depth_0
        
        # test debug
        if self.args.mode == "test":
            output["cost_volumn"] = cost_reg_list       #for output
            output["confidences"] = confidences
            # output["depth_low"] = depth_est_list
            # output["first_layer_pro_test"] = first_layer_pro_test

        return output

def sL1_loss(depth_est, depth_gt, mask):
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

def MSE_loss(depth_est, depth_gt, mask):
    return F.mse_loss(depth_est[mask], depth_gt[mask], size_average=True)


###### loss gao
def mvsnet_confidence_loss(confidence):
    loss = (-1.0 * F.logsigmoid(confidence)).mean()
    return loss

def mvsnet_smooth_l1_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)

##min_depth和max_depth应该是一张图
def mvsnet_focal_losses(costs, depth_gt, variances, mask, max_depth, min_depth, focal_coefficient):
    #costs:[B,D,H,W]
    #depth_gt[B,H,W]
    #variances[B,1,H,W]
    #max_depthmap[B,1,H,W]
    eps = 1e-40
    start_depth = min_depth
    max_depth = max_depth      
    mask = mask > 0.5   #mask for valid depth
    b, d, h, w = costs.shape


    h, w = depth_gt.shape[1], depth_gt.shape[2]
    depth_interval_mask = (depth_gt.unsqueeze(1) > min_depth) & (depth_gt.unsqueeze(1) < max_depth)
    depth_gt_mask = depth_gt*mask   #[B,1,H,W]
    depth_gt_mask = depth_gt_mask.unsqueeze(1).repeat(1,d,1,1)     #[B,D,H,W]

    # #这一部分加上深度范围的判断
    # depth_gt_mask = depth_gt_mask.unsqueeze(1)*depth_interval_mask

    # 将max depth和min depth作为图来输入的写法
    # depth_gt_mask = depth_gt_mask.repeat(1,d,1,1)     #[B,D,H,W]
    # interval_map = (max_depth - start_depth)/(d-1)  #[B,1,H,W]
    # index = start_depth.repeat(1,d,1,1)      #[B,D,H,W]
    # for depth_interval in range(d):
    #     index[:,depth_interval,:,:] += (interval_map*depth_interval).view(b,h,w)
    

    index = torch.linspace(start_depth, max_depth, d)
    index = index.to(depth_gt.device)
    index = index.repeat(b, h, w, 1).permute(0, 3, 1, 2).contiguous()
   

    scaled_distance = ((-torch.abs(index - depth_gt_mask)) / variances)   #[B,D,H,W]
    probability_gt = F.softmax(scaled_distance, dim=1)
    probability_gt = probability_gt*mask.unsqueeze(1) + eps    #[B,H,W]

    estProb_log = F.log_softmax(costs, dim=1)   #[B,H,W]
    weight = (1.0 - probability_gt).pow(-focal_coefficient).type_as(probability_gt)
    mask_tmp = mask.unsqueeze(1).repeat(1,d,1,1)
    loss = -((probability_gt * estProb_log) * weight * mask_tmp.float()).sum(dim=1, keepdim=True).mean()

    return loss

##min_depth和max_depth应该是一张图
def mvsnet_focal_losses_uniform(costs, depth_gt, mask, max_depth, min_depth, focal_coefficient):
    #costs:[B,D,H,W]
    #depth_gt[B,H,W]
    #variances[B,1,H,W]
    #max_depthmap[B,1,H,W]
    eps = 1e-40
    start_depth = min_depth
    max_depth = max_depth      
    mask = mask > 0.5   #mask for valid depth
    b, d, h, w = costs.shape
    variances = 1.2

    h, w = depth_gt.shape[1], depth_gt.shape[2]
    depth_interval_mask = (depth_gt.unsqueeze(1) > min_depth) & (depth_gt.unsqueeze(1) < max_depth)
    depth_gt_mask = depth_gt*mask   #[B,1,H,W]
    depth_gt_mask = depth_gt_mask.unsqueeze(1).repeat(1,d,1,1)     #[B,D,H,W]
    
    index = torch.linspace(start_depth, max_depth, d)
    index = index.to(depth_gt.device)
    index = index.repeat(b, h, w, 1).permute(0, 3, 1, 2).contiguous()
   

    scaled_distance = ((-torch.abs(index - depth_gt_mask)) / variances)   #[B,D,H,W]
    probability_gt = F.softmax(scaled_distance, dim=1)
    probability_gt = probability_gt*mask.unsqueeze(1) + eps    #[B,H,W]

    estProb_log = F.log_softmax(costs, dim=1)   #[B,H,W]
    weight = (1.0 - probability_gt).pow(-focal_coefficient).type_as(probability_gt)
    mask_tmp = mask.unsqueeze(1).repeat(1,d,1,1)
    loss = -((probability_gt * estProb_log) * weight * mask_tmp.float()).sum(dim=1, keepdim=True).mean()

    return loss

def mvsnet_sum_loss(costs, depth_est, depth_gt, variances, mask, max_depth, min_depth, confidence):
    focal_coefficient = 5.0    #alpha
    lamda_confidence = 8.0
    lamda_regression = 0.1

    gao_lamda_confidence = 50
    gao_lamda_focal = 80

    focal_loss = mvsnet_focal_losses(costs, depth_gt, variances, mask, max_depth, min_depth, focal_coefficient)
    smooth_l1_loss = mvsnet_smooth_l1_loss(depth_est, depth_gt, mask)
    confidence_loss = mvsnet_confidence_loss(confidence)

    # small loss
    loss = focal_loss + smooth_l1_loss*lamda_regression + confidence_loss*lamda_confidence
    # big loss
    # loss = focal_loss*gao_lamda_focal + smooth_l1_loss + confidence_loss*gao_lamda_confidence

    return loss

def mvsnet_sum_loss_confidence(costs, depth_est, depth_gt, variances, mask, max_depth, min_depth, confidence,depthmap_confidence):
    focal_coefficient = 5.0    #alpha
    lamda_confidence = 8.0
    lamda_regression = 0.1

    gao_lamda_confidence = 50
    gao_lamda_focal = 80

    focal_loss = mvsnet_focal_losses(costs, depth_gt, variances, mask, max_depth, min_depth, focal_coefficient)
    smooth_l1_loss = mvsnet_smooth_l1_loss(depth_est, depth_gt, mask)
    confidence_loss = mvsnet_confidence_loss(confidence)
    depthmap_confidence_loss = depthmap_confidence(depth_gt,depth_est,depthmap_confidence)

    loss = focal_loss + smooth_l1_loss*lamda_regression + confidence_loss*lamda_confidence + depthmap_confidence_loss

    return loss

# def depthmap_confidence(depth_gt,depth_est,depthmap_confidence):
#     error = abs(depth_gt - depth_est)
#     relative_error = error/depth_gt
#     confidence_gt = relative_error<0.01
#     # confidence_gt = float(confidence_gt == 'true')
#     return F.smooth_l1_loss(confidence_gt.float(), depthmap_confidence)

def depthmap_confidence(depth_gt,depth_est,depthmap_confidence):
    # BCELoss requires its input to be between 0 and 1.
    BCE = nn.BCEWithLogitsLoss()
    # mask = depth_gt >= 0
    # depth_gt = depth_gt[mask]
    # depth_est = depth_est[mask]
    error = abs(depth_gt - depth_est)
    relative_error = error/depth_gt
    confidence_gt = relative_error<0.01
    depthmap_confidence = depthmap_confidence.squeeze(1)
    loss = BCE(depthmap_confidence,confidence_gt.float())
    return loss