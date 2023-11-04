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
            nn.Conv2d(16 , 16 * 2, 3, 1, padding=1),
            nn.BatchNorm2d(16 * 2, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(16 * 2, 9, 3, padding=1, bias=False)
        )
        self.unfold = nn.Unfold(kernel_size=(3,3), stride=1, padding=0)
    def forward(self, depth, img_feature):
        img_filter = self.depth_propagation_layer(img_feature)
        img_filter = F.softmax(img_filter, dim=1)    #正则化之后转变为概率  [B,9,H,W]

        depth_pad = F.pad(depth, (1, 1, 1, 1), mode='replicate')    #填充depth，每两个参数为一组,填出来周围的一个边[]
        depth_pad = depth_pad.permute(1, 0, 2, 3)    #[B,9,H,W]
        depth_unfold = self.unfold(depth_pad)

        b, c, h, w = img_filter.size()
        prob = img_filter.view(b, 9, h*w)   # 3*3=9

        result_depth = torch.sum(depth_unfold * prob, dim=1)   #unfold操作只卷，不做积，加在一起相当于卷积的操作
        result_depth = result_depth.view(b, 1, h, w).squeeze(1)
        return result_depth


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


class CostRegNet_gao_modified(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()

        self.conv0 = ConvBnReLU3D(4, 8, kernel_size=3, pad=1)
        self.conv0a = ConvBnReLU3D(8, 8, kernel_size=3, pad=1)

        self.conv1 = ConvBnReLU3D(8, 16,stride=2, kernel_size=3, pad=1)
        self.conv2 = ConvBnReLU3D(16, 16, kernel_size=3, pad=1)
        self.conv2a = ConvBnReLU3D(16, 16, kernel_size=3, pad=1)
        self.conv3 = ConvBnReLU3D(16, 32, kernel_size=3, pad=1)
        self.conv4 = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)
        self.conv4a = ConvBnReLU3D(32, 32, kernel_size=3, pad=1)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=0, stride=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.prob0 = nn.Conv3d(24, 1, 3, stride=1, padding=1)

    def forward(self, x):

        conv0 = self.conv0a(self.conv0(x))
        conv2 = self.conv2a(self.conv2(self.conv1(conv0)))
        conv4 = self.conv4a(self.conv4(self.conv3(conv2)))

        # conv5 = conv2+self.conv5(conv4)
        conv5 = torch.cat(self.conv5(conv4),conv2)   #dim=32

        # conv6 = conv0+self.conv6(conv5)
        conv6 = torch.cat(self.conv6(conv5),conv0) 

        prob = self.prob0(conv6).squeeze(1)

        return prob


class network(nn.Module):
    def __init__(self, args):
        super(network, self).__init__()
        self.featurePyramid = FeaturePyramid()
        self.cost_reg_refine = CostRegNet()
        self.cost_reg_longrange = CostRegNet()
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

        # self.DTU_variance_scale = 1

        #用两个不同的confidence estimation网络是因为inplanes数目不一样
        self.conf_est_net_stage2 = ConfidenceEstimation(in_planes=32, batchNorm=self.batchNorm)
        # self.refine_conf_est_net = ConfidenceEstimation(in_planes=self.refine_num_depth,batchNorm=self.batchNorm)
        # self.test_local_softmax = LocalSoftArgmin(max_disp=self.num_depth, depth_interval=self.DTU_variance_scale, radius=2)

        self.Group = 4
        
        # self.gamma_s3 = nn.Parameter(torch.zeros(1))
        # self.beta_s3 = nn.Parameter(torch.zeros(1))
        self.gamma_s2 = nn.Parameter(torch.zeros(1))
        self.beta_s2 = nn.Parameter(torch.zeros(1))

        self.training_stage = [48,32]
        # fast depth propagation
        # self.depth_propagation_layer = nn.Sequential(
        #     nn.Conv2d(16 , 16 * 2, 3, 1, padding=1),
        #     nn.BatchNorm2d(16 * 2, momentum=0.01),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16 * 2, 9, 3, padding=1, bias=False)
        # )
        # self.unfold = nn.Unfold(kernel_size=(3,3), stride=1, padding=0)
        # self.fast_propagation = FastPropagation()

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
        cost_reg = self.cost_reg_longrange(cost_volume)    #[B,D,H,W]
        # cost_reg = self.cost_reg_refine(cost_volume)

#####################################################gao_added##########################################
        ####这一段计算出confidence和正太分布相关的variance
        confidences = []
        variances = []
        max_depthmap_list = []
        min_depthmap_list = []
        cost_reg_list = []
        #第一层train
        if self.args.mode =="train":

            prob_volume = F.softmax(cost_reg, dim=1)
            depth = depth_regression(prob_volume, depth_values=depth_hypos)
            depth_est_list.append(depth) 

            # confidence_costs = self.conf_est_net(cost_reg)     #(B,1,H,W)
            # confidence = torch.sigmoid(confidence_costs)     #(B,1,H,W)
            # variance = self.confidence_coefficient * (1 - confidence) + self.confidence_init_value   ##s*(1-conf)+epsilon  正态分布的方差
            # variance = variance * self.DTU_variance_scale
            # variances.append(variance)
            # cost_reg_list.append(cost_reg)
            # confidences.append(confidence)


        if self.args.mode == "test":
            # confidence_costs = self.conf_est_net(cost_reg)     #(B,1,H,W)
            # confidence = torch.sigmoid(confidence_costs)     #(B,1,H,W)
            # confidences.append(confidence)
            prob_volume = F.softmax(cost_reg, dim=1)
            depth = depth_regression(prob_volume, depth_values=depth_hypos)
            depth_est_list.append(depth)

       

        ## Upsample depth map and refine along feature pyramid
        for level in range(self.args.nscale-2,-1,-1):    #是倒序金字塔

            # Upsample 
            depth_up = nn.functional.interpolate(depth[None,:],size=None,scale_factor=2,mode='bicubic',align_corners=None)  #[1,B,H,W]
            depth_up = depth_up.squeeze(0)  #[B,H,W]

            # fast depth propagation
            # depth_up = self.fast_propagation(depth_up, ref_feature_pyramid[level])   #[b, h, w]
            if level==self.args.nscale-2:
                lamb = 1.5
                d = depth_hypos.shape[1]
                h,w = depth.shape[1], depth.shape[2]
                samp_variance = (depth_hypos.unsqueeze(-1).unsqueeze(-1).repeat(1,1,h,w) - depth.unsqueeze(1).repeat(1,d,1,1)) ** 2
                exp_variance = lamb * torch.sum(samp_variance * prob_volume, dim=1, keepdim=False) ** 0.5

                exp_variance = nn.functional.interpolate(exp_variance[None,:],size=None,scale_factor=2,mode='bicubic',align_corners=None)  #[1,B,H,W]
                exp_variance = exp_variance.squeeze(0)  #[B,H,W]

                mindisparity_s3 = depth_up - (self.gamma_s2 + 1) * exp_variance - self.beta_s2
                maxdisparity_s3 = depth_up + (self.gamma_s2 + 1) * exp_variance + self.beta_s2
                depth_hypos = generate_depth_range(mindisparity_s3,maxdisparity_s3)
                
            else:
                # Generate depth hypothesis
                depth_hypos = calDepthHypo(self.args,depth_up,ref_in_multiscales[:,level,:,:], src_in_multiscales[:,:,level,:,:],ref_ex,src_ex,depth_min, depth_max,level)

            # depth range
            confidence_save_path = "./depth_range_ours/"
            if not os.path.exists(confidence_save_path):
                os.makedirs(confidence_save_path)
            save_shape = depth_hypos.shape[3]
            save_fact = 160/save_shape
            cost_cpu = nn.functional.interpolate(depth_hypos,size=None,scale_factor=save_fact,mode='bicubic',align_corners=None)
            cost_cpu = cost_cpu.detach().clone().cpu().numpy()
            np.save(confidence_save_path+"depth_range_{}".format(save_fact),cost_cpu)
            del cost_cpu

            cost_volume = proj_cost_AACVP(Group=self.Group, settings=self.args, ref_feature=ref_feature_pyramid[level],
                                          src_feature=src_feature_pyramids,
                                          level=level, ref_in=ref_in_multiscales[:, level, :, :],
                                          src_in=src_in_multiscales[:, :, level, :, :], ref_ex=ref_ex,
                                          src_ex=src_ex[:, :], depth_hypos=depth_hypos)

            if level==self.args.nscale-2:
                cost_reg2 = self.cost_reg_longrange(cost_volume)

                confidence_costs = self.conf_est_net_stage2(cost_reg2)     #(B,1,H,W)
                confidence = torch.sigmoid(confidence_costs)     #(B,1,H,W)
                variance = self.confidence_coefficient * (1 - confidence) + self.confidence_init_value   ##s*(1-conf)+epsilon  正态分布的方差
                variance = variance * self.DTU_variance_scale
                variances.append(variance)
                cost_reg_list.append(cost_reg2)
                confidences.append(confidence)
            else:
                cost_reg2 = self.cost_reg_refine(cost_volume)

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
        confidence_save_path = "./depth_range_ours/"
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

        # # test code: to show cost volumn
        # cost_reg = depth_est_list[1]
        # depth_save_path = "./depth_low/"
        # if not os.path.exists(depth_save_path):
        #     os.makedirs(depth_save_path)
        # cost_cpu = cost_reg.detach().clone().cpu().numpy()
        # np.save(depth_save_path+"cost{}_confidence".format(batch_idx),cost_cpu)
        # del cost_cpu, cost_reg

        ## Return confidence gao for loss
        if self.args.mode == "train":
            output["confidences"] = confidences
            output["variances"] = variances
            output["cost_volumn"] = cost_reg_list
            output["max_depthmap"] = max_depthmap_list
            output["min_depthmap"] = min_depthmap_list
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

    focal_loss = mvsnet_focal_losses(costs, depth_gt, variances, mask, max_depth, min_depth, focal_coefficient)
    smooth_l1_loss = mvsnet_smooth_l1_loss(depth_est, depth_gt, mask)
    confidence_loss = mvsnet_confidence_loss(confidence)

    # small loss
    loss = focal_loss + smooth_l1_loss*lamda_regression + confidence_loss*lamda_confidence

    return loss

def mvsnet_sum_loss_uniform(costs, depth_est, depth_gt, mask, max_depth, min_depth):
    focal_coefficient = 5.0    #alpha
    lamda_confidence = 8.0
    lamda_regression = 0.1

    focal_loss = mvsnet_focal_losses_uniform(costs, depth_gt, mask, max_depth, min_depth, focal_coefficient)
    smooth_l1_loss = mvsnet_smooth_l1_loss(depth_est, depth_gt, mask)

    loss = focal_loss + smooth_l1_loss*lamda_regression 
    return loss