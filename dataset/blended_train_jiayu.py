# Dataloader for the DTU dataset in Yaoyao's format.
# by: Jiayu Yang
# date: 2020-01-28

# Note: This file use part of the code from the following projects.
#       Thanks for the authors for the great code.
#       MVSNet: https://github.com/YoYo000/MVSNet
#       MVSNet_pytorch: https://github.com/xy-guo/MVSNet_pytorch

from dataset.utils import *
from dataset.blended_datapath import *
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import cv2
# For debug:
# import matplotlib.pyplot as plt
# import pdb

class MVSDataset(Dataset):
    def __init__(self, args, logger=None):
        # Initializing the dataloader
        super(MVSDataset, self).__init__()
        
        # Parse input
        self.args = args
        self.data_root = self.args.dataset_root
        self.scan_list_file = getScanListFile(self.data_root,self.args.mode)
        # self.pair_list_file = getPairListFile(self.data_root,self.args.mode)
        self.logger = logger
        if logger==None:
            import logger
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
            consoleHandler = logging.StreamHandler(sys.stdout)
            consoleHandler.setFormatter(formatter)
            self.logger.addHandler(consoleHandler)
            self.logger.info("File logger not configured, only writing logs to stdout.")
        self.logger.info("Initiating dataloader for our pre-processed DTU dataset.")
        self.logger.info("Using dataset:"+self.data_root+self.args.mode+"/")

        self.metas = self.build_list(self.args.mode)
        self.logger.info("Dataloader initialized.")

    def build_list(self,mode):
        # Build the item meta list
        metas = []

        # Read scan list
        scan_list = readScanList(self.scan_list_file,self.args.mode, self.logger)

        # Read pairs list
        for scan in scan_list:
            self.pair_list_file = self.data_root + scan + "/cams/pair.txt"
            with open(self.pair_list_file) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    if mode == "train":
                        for light_idx in range(7):
                            metas.append((scan, ref_view, src_views, light_idx))
                    elif mode == "test":
                        metas.append((scan, ref_view, src_views, 3))

        self.logger.info("Done. metas:"+str(len(metas)))
        return metas

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views, light_idx = meta
        #test
        if self.args.nsrc > len(src_views):
            print("warning! there is a bug")
            print(scan,src_views)
        assert self.args.nsrc <= len(src_views)

        self.logger.debug("Getting Item:\nscan:"+str(scan)+"\nref_view:"+str(ref_view)+"\nsrc_view:"+str(src_views)+"\nlight_idx"+str(light_idx))

        ref_img = [] 
        src_imgs = [] 
        ref_depths = [] 
        ref_depth_mask = [] 
        ref_intrinsics = [] 
        src_intrinsics = [] 
        ref_extrinsics = [] 
        src_extrinsics = [] 
        depth_min = [] 
        depth_max = [] 

        ## 1. Read images
        # ref image
        ref_img_file = getImageFile(self.data_root,self.args.mode,scan,ref_view,light_idx)
        ref_img = read_img_blended_train(ref_img_file)

        # src image(s)
        for i in range(self.args.nsrc):
            src_img_file = getImageFile(self.data_root,self.args.mode,scan,src_views[i],light_idx)
            src_img = read_img_blended_train(src_img_file)

            src_imgs.append(src_img)

        ## 2. Read camera parameters
        cam_file = getCameraFile(self.data_root,self.args.mode,scan,ref_view)
        ref_intrinsics, ref_extrinsics, depth_min, depth_max = read_cam_file_blended_train(cam_file)
        for i in range(self.args.nsrc):
            cam_file = getCameraFile(self.data_root,self.args.mode,scan,src_views[i])
            intrinsics, extrinsics, depth_min_tmp, depth_max_tmp = read_cam_file_blended_train(cam_file)
            src_intrinsics.append(intrinsics)
            src_extrinsics.append(extrinsics)

        ## 3. Read Depth Maps
        if self.args.mode == "train":
            imgsize = self.args.imgsize
            nscale = self.args.nscale

            # Read depth map of same size as input image first.
            depth_file = getDepthFile(self.data_root,self.args.mode,scan,ref_view)
            ref_depth = read_depth_blended_train(depth_file)
            depth_frame_size = (ref_depth.shape[0],ref_depth.shape[1])
            frame = np.zeros(depth_frame_size)
            frame[:ref_depth.shape[0],:ref_depth.shape[1]] = ref_depth
            ref_depths.append(frame)

            # Downsample the depth for each scale.
            ref_depth = Image.fromarray(ref_depth)
            original_size = np.array(ref_depth.size).astype(int)

            for scale in range(1,nscale):
                new_size = (original_size/(2**scale)).astype(int)
                down_depth = ref_depth.resize((new_size),Image.BICUBIC)
                frame = np.zeros(depth_frame_size)
                down_np_depth = np.array(down_depth)
                frame[:down_np_depth.shape[0],:down_np_depth.shape[1]] = down_np_depth
                ref_depths.append(frame)

        # Orgnize output and return
        sample = {}
        sample["ref_img"] = np.moveaxis(np.array(ref_img),2,0)
        sample["src_imgs"] = np.moveaxis(np.array(src_imgs),3,1)
        sample["ref_intrinsics"] = np.array(ref_intrinsics)
        sample["src_intrinsics"] = np.array(src_intrinsics)
        sample["ref_extrinsics"] = np.array(ref_extrinsics)
        sample["src_extrinsics"] = np.array(src_extrinsics)
        sample["depth_min"] = depth_min
        sample["depth_max"] = depth_max

        # print(sample)

        if self.args.mode == "train":
            sample["ref_depths"] = np.array(ref_depths,dtype=float)
            sample["ref_depth_mask"] = np.array(ref_depth_mask)
        elif self.args.mode == "test":
            sample["filename"] = scan + '/{}/' + '{:0>8}'.format(ref_view) + "{}"

        return sample


def read_depth_blended_train(filename):
    depth = np.array(read_pfm(filename)[0], dtype=np.float32)
    h,w = depth.shape[0],depth.shape[1]
    depth = cv2.resize(depth, (192,144), interpolation=cv2.INTER_NEAREST)
    # read pfm depth file
    return depth   

def read_img_blended_train(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    img = np.array(img, dtype=np.float32) / 255.
    h,w = img.shape[0],img.shape[1]
    
    img = cv2.resize(img, (192,144), interpolation=cv2.INTER_NEAREST)

    return img

def read_cam_file_blended_train(filename):

    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    intrinsics2 = intrinsics.copy()
    intrinsics2[:2, :3] = intrinsics[:2, :3] / 4.   #  

    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_interval = float(lines[11].split()[1])/2
    depth_max = depth_min+(256*depth_interval)
    return intrinsics2, extrinsics, depth_min, depth_max