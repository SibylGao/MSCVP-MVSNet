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
# Debug:
# import pdb,time
# import matplotlib.pyplot as plt
# from verifications import *

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

# 这里跟AACVP好像是一样的,输入inplanes = group
class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()

        self.conv0 = ConvBnReLU3D(4, 16, kernel_size=3, pad=1)
        # self.conv0 = ConvBnReLU3D(16, 16, kernel_size=3, pad=1)
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

        # gwc原文用的8，AACVP用的4
        self.Group = 4

        #用两个不同的confidence estimation网络是因为inplanes数目不一样
        self.conf_est_net = ConfidenceEstimation(in_planes=self.num_depth, batchNorm=self.batchNorm)
        self.refine_conf_est_net = ConfidenceEstimation(in_planes=self.refine_num_depth,batchNorm=self.batchNorm)
        self.test_local_softmax = LocalSoftArgmin(max_disp=self.num_depth, depth_interval=self.DTU_variance_scale, radius=2)

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

        # ## 方差
        # ref_volume = ref_feature_pyramid[-1].unsqueeze(2).repeat(1, 1, len(depth_hypos[0]), 1, 1)

        # volume_sum = ref_volume
        # volume_sq_sum = ref_volume.pow_(2)
        # if self.args.mode == "test":
        #     del ref_volume
        # for src_idx in range(self.args.nsrc):
        #     # warpped features
        #     warped_volume = homo_warping(src_feature_pyramids[src_idx][-1], ref_in_multiscales[:,-1], src_in_multiscales[:,src_idx,-1,:,:], ref_ex, src_ex[:,src_idx], depth_hypos)


        #     if self.args.mode == "train":
        #         volume_sum = volume_sum + warped_volume
        #         volume_sq_sum = volume_sq_sum + warped_volume ** 2
        #     elif self.args.mode == "test":
        #         volume_sum = volume_sum + warped_volume
        #         volume_sq_sum = volume_sq_sum + warped_volume ** 2
        #         del warped_volume
        #     else: 
        #         print("Wrong!")
        #         pdb.set_trace()
                
        # # Aggregate multiple feature volumes by variance
        # cost_volume = volume_sq_sum.div_(self.args.nsrc+1).sub_(volume_sum.div_(self.args.nsrc+1).pow_(2))
        # if self.args.mode == "test":
        #     del volume_sum
        #     del volume_sq_sum

        ref_volume = ref_feature_pyramid[-1].unsqueeze(2).repeat(1, 1, len(depth_hypos[0]), 1, 1)
        B, C, H, W = src_feature_pyramids[0][0].shape
        V = self.args.nsrc
        # Kwea3 implementation as reference
        ref_volume = ref_volume.view(B, self.Group, C // self.Group, *ref_volume.shape[-3:])
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
        cost_reg = self.cost_reg_refine(cost_volume)    #[B,C,D,H,W]


#####################################################gao_added##########################################
        ####这一段计算出confidence和正太分布相关的variance
        confidences = []
        variances = []
        max_depthmap_list = []
        min_depthmap_list = []
        cost_reg_list = []
        #第一层train
        if self.args.mode =="train":
            confidence_costs = self.conf_est_net(cost_reg)     #(B,1,H,W)
            confidence = torch.sigmoid(confidence_costs)     #(B,1,H,W)
            variance = self.confidence_coefficient * (1 - confidence) + self.confidence_init_value   ##s*(1-conf)+epsilon  正态分布的方差
            variance = variance * self.DTU_variance_scale
            variances.append(variance)
            cost_reg_list.append(cost_reg)
            confidences.append(confidence)
            prob_volume = F.softmax(cost_reg, dim=1)
            depth = depth_regression(prob_volume, depth_values=depth_hypos)
            depth_est_list.append(depth)
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
            # 用local softmax
            depth = self.test_local_softmax(cost_reg)    #[B,1,H,W]
            depth = depth.squeeze(1)
            depth_est_list.append(depth)

       

        # #第一层train  深度值
        # if self.args.mode =="train":
        #     b,h,w = depth.shape
        #     min_depthmap = depth_hypos[:,0]     #[B,D]
        #     max_depthmap = depth_hypos[:,-1]
        #     max_depthmap = max_depthmap.view(b,1).unsqueeze(2).unsqueeze(3).repeat(1,1,h,w)
        #     min_depthmap = min_depthmap.view(b,1).unsqueeze(2).unsqueeze(3).repeat(1,1,h,w)
        #     max_depthmap_list.append(max_depthmap)
        #     min_depthmap_list.append(min_depthmap)     #[B,1,H,W]


        # if self.args.mode == "test":
        #     depth_save_path = "./cost_confidence/"
        #     if not os.path.exists(depth_save_path):
        #         os.makedirs(depth_save_path)
        #     cost_cpu = cost_reg.detach().clone().cpu().numpy()
        #     np.save(depth_save_path+"cost{}_confidence".format(self.cost_save_index),cost_cpu)
        #     del cost_cpu, cost_reg
        #     self.cost_save_index = self.cost_save_index + 1

        ## Upsample depth map and refine along feature pyramid
        for level in range(self.args.nscale-2,-1,-1):

            # Upsample
            depth_up = nn.functional.interpolate(depth[None,:],size=None,scale_factor=2,mode='bicubic',align_corners=None)
            depth_up = depth_up.squeeze(0)
            # Generate depth hypothesis
            depth_hypos = calDepthHypo(self.args,depth_up,ref_in_multiscales[:,level,:,:], src_in_multiscales[:,:,level,:,:],ref_ex,src_ex,depth_min, depth_max,level)

            # cost_volume = proj_cost(self.args,ref_feature_pyramid[level],src_feature_pyramids,level,ref_in_multiscales[:,level,:,:], src_in_multiscales[:,:,level,:,:], ref_ex, src_ex[:,:],depth_hypos)

            confidence_save_path = "./depth_range_cvpbaseline/"
            if not os.path.exists(confidence_save_path):
                os.makedirs(confidence_save_path)
            save_shape = depth_hypos.shape[3]
            save_fact = 160/save_shape
            cost_cpu = nn.functional.interpolate(depth_hypos,size=None,scale_factor=save_fact,mode='bicubic',align_corners=None)
            cost_cpu = cost_cpu.detach().clone().cpu().numpy()
            np.save(confidence_save_path+"depth_range_{}".format(save_fact),cost_cpu)
            del cost_cpu


            # AACVP version
            cost_volume = proj_cost_AACVP(Group=self.Group, settings=self.args, ref_feature=ref_feature_pyramid[level],
                                          src_feature=src_feature_pyramids,
                                          level=level, ref_in=ref_in_multiscales[:, level, :, :],
                                          src_in=src_in_multiscales[:, :, level, :, :], ref_ex=ref_ex,
                                          src_ex=src_ex[:, :], depth_hypos=depth_hypos)

            cost_reg2 = self.cost_reg_refine(cost_volume)

            # #第二层train
            # if self.args.mode =="train":
            #     b,h,w = depth_hypos.shape[0], depth_hypos.shape[2], depth_hypos.shape[3]
            #     confidence_costs = self.refine_conf_est_net(cost_reg2)     #(B,1,H,W)
            #     confidence = torch.sigmoid(confidence_costs)     #(B,1,H,W)
            #     variance = self.confidence_coefficient * (1 - confidence) + self.confidence_init_value   ##s*(1-conf)+epsilon  正态分布的方差
            #     variances.append(variance)

            #     min_depthmap = depth_hypos[:,0,:,:]       #[B,1,H,W]
            #     max_depthmap = depth_hypos[:,-1,:,:]
            #     max_depthmap = max_depthmap.view(b,1,h,w)
            #     min_depthmap = min_depthmap.view(b,1,h,w)
            #     max_depthmap_list.append(max_depthmap)
            #     min_depthmap_list.append(min_depthmap)
            #     cost_reg_list.append(cost_reg2)
            #     confidences.append(confidence)
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

        # depth range
        confidence_save_path = "./depth_range_cvpbaseline/"
        if not os.path.exists(confidence_save_path):
            os.makedirs(confidence_save_path)
        save_shape = prob_volume.shape[3]
        save_fact = 160/save_shape
        cost_cpu = nn.functional.interpolate(prob_volume,size=None,scale_factor=save_fact,mode='bicubic',align_corners=None)
        cost_cpu = cost_cpu.detach().clone().cpu().numpy()
        np.save(confidence_save_path+"pro_{}".format(save_fact),cost_cpu)
        del cost_cpu

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

        output["depth_est_list"] = depth_est_list
        output["prob_confidence"] = prob_confidence

        ## Return confidence gao for loss
        if self.args.mode == "train":
            output["confidences"] = confidences
            output["variances"] = variances
            output["cost_volumn"] = cost_reg_list
            output["max_depthmap"] = max_depthmap_list
            output["min_depthmap"] = min_depthmap_list
        
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

    loss = focal_loss + smooth_l1_loss*lamda_regression + confidence_loss*lamda_confidence
    # loss = focal_loss*gao_lamda_focal + lamda_regression + confidence_loss*gao_lamda_confidence
    return loss

def mvsnet_sum_loss_uniform(costs, depth_est, depth_gt, mask, max_depth, min_depth):
    focal_coefficient = 5.0    #alpha
    lamda_confidence = 8.0
    lamda_regression = 0.1

    focal_loss = mvsnet_focal_losses_uniform(costs, depth_gt, mask, max_depth, min_depth, focal_coefficient)
    smooth_l1_loss = mvsnet_smooth_l1_loss(depth_est, depth_gt, mask)

    loss = focal_loss + smooth_l1_loss*lamda_regression 
    return loss