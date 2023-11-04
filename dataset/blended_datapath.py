# Fetching file path and name for dataloader on our DTU dataset. 
# by: Jiayu Yang
# date: 2019-08-01

import os
import numpy as np
from PIL import Image

# DTU:
# 2020-01-31 14:20:42: Modified to read original yao's format.
def getScanListFile(data_root,mode):
    if mode == "train":
        scan_list_file = data_root + "training_list.txt"
    else:
        scan_list_file = data_root + "validation_list.txt"
    return scan_list_file

def getPairListFile(data_root,mode):
    pair_list_file = data_root+"Cameras/pair.txt"
    return pair_list_file

# def getDepthFile(data_root,mode,scan,view):
#     depth_name = "depth_map_"+str(view).zfill(4)+".pfm"
#     if mode == "train":
#         scan_path = "Depths/"+scan+"_train/"
#     else:
#         scan_path = "Depths/"+scan+"/"
#     depth_file = os.path.join(data_root,scan_path,depth_name)
#     return depth_file

def getDepthFile(data_root,mode,scan,view):
    depth_file = data_root + scan + "/rendered_depth_maps/" + str(view).zfill(8) + ".pfm"
    return depth_file

# def getImageFile(data_root,mode,scan,view,light):
#     image_name = "rect_"+str(view+1).zfill(3)+"_"+str(light)+"_r5000.png"
#     if mode == "train":
#         scan_path = "Rectified/"+scan+"_train/"
#     else:
#         scan_path = "Rectified/"+scan+"/"
#     image_file = os.path.join(data_root,scan_path,image_name)
#     return image_file

def getImageFile(data_root,mode,scan,view,light):
    image_file = data_root + scan + "/blended_images/" + str(view).zfill(8) + ".jpg"
    return image_file

# def getCameraFile(data_root,mode,view):
#     cam_name = str(view).zfill(8)+"_cam.txt"
#     cam_path = "Cameras/"
#     cam_file = os.path.join(data_root,cam_path,cam_name)
#     return cam_file

def getCameraFile(data_root,mode,scan,view):
    cam_file = data_root + scan + "/cams/" + str(view).zfill(8) + "_cam.txt"
    return cam_file


def read_img_blended(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    img = np.array(img, dtype=np.float32) / 255.
    if img.shape[0] == 2048:
        img = img[:1536,:2048,:]
    
    if img.shape[0] == 600:
        img = img[:592,:800,:]

    return img

def read_cam_file_blended(filename):

    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))


    # intrinsics2 = intrinsics.copy()
    # intrinsics2[:2, :3] = intrinsics[:2, :3] * 2.   #  

    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_interval = float(lines[11].split()[1])/2
    depth_max = depth_min+(256*depth_interval)
    return intrinsics, extrinsics, depth_min, depth_max