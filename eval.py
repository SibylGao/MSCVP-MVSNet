# Evaluate CVP-MVSNet
# by: Jiayu Yang
# date: 2019-08-29

import os,sys,time,logging,argparse,datetime,re

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

# from dataset import dtu_jiayu
# from dataset import dtu_satge3_jiayu as dtu_jiayu
from dataset import dtu_stage3_jiayu_train_demo as dtu_jiayu
# from dataset import blended_jiayu as dtu_jiayu
# from models import net
# from models import cvp_gwc_fast_confidence as net
# from models import cvp_gwc_sarpn as net
# from models import cvp_gwc_sarpn as net
# from models import cvp_uas as net
from models import cvp_stage3 as net
# from models import cvp_stage3_acf_longrange as net
# from models import cvp_satge3_acf_range2 as net
# from models import DHS1 as net
# from models import dhs1_dhs2_dhs3 as net
# from models import dhs1_dhs2 as net
# from models import cvp_gwc_acf_range2_final as net
# from models import cvp_gwc_short_long_range as net
# from models import cvp_stage3_acf_rang2_unet as net

# from models import cvp_shortlongrange_adaptive_acf as net

# from models import cvp_gwc_acf as net
# from models import cvp_gwc_fast_confidence as net
from models.modules import *
from utils import *
from PIL import Image
from argsParser import getArgsParser
from plyfile import PlyData, PlyElement
from tqdm import tqdm

# Debug import
import pdb
import matplotlib.pyplot as plt

import numpy as np
# ##cuda path
# ORIGINAL_CUDA_HOME=$CUDA_HOME
# ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
# export CUDA_HOME=/usr/local/cuda-10.2
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


cudnn.benchmark = True

# Arg parser
parser = getArgsParser()
args = parser.parse_args()
assert args.mode == "test"

# logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
curTime = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
log_path = args.loggingdir+args.info.replace(" ","_")+"/"
if not os.path.isdir(args.loggingdir):
    os.mkdir(args.loggingdir)
if not os.path.isdir(log_path):
    os.mkdir(log_path)
log_name = log_path + curTime + '.log'
logfile = log_name
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fileHandler = logging.FileHandler(logfile, mode='a')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
logger.info("Logger initialized.")
logger.info("Writing logs to file:"+logfile)

settings_str = "All settings:\n"
line_width = 30
for k,v in vars(args).items(): 
    settings_str += '{0}: {1}\n'.format(k,v)
logger.info(settings_str)


# occumpy gpu
occumpy_gpu = [4]
occumpy_gpu_list_str = ','.join(map(str, occumpy_gpu)) 


# Run CVP-MVSNet to save depth maps and confidence maps
def save_depth():
    # dataset, dataloader
    test_dataset = dtu_jiayu.MVSDataset(args, logger)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=args.eval_shuffle, num_workers=8, drop_last=True)

    # model
    model = net.network(args)
    
    # GPU parallel
    gpu_list = [4]
    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu") 
    model = nn.DataParallel(model, device_ids=[0])

    # model = nn.DataParallel(model)  


    model.cuda()

    # load checkpoint file specified by args.loadckpt
    logger.info("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'],strict=False)

    with torch.no_grad():

        for batch_idx, sample in enumerate(test_loader):

            start_time = time.time()

            sample_cuda = tocuda(sample)

            torch.cuda.empty_cache()

            outputs = model(\
                sample_cuda["ref_img"].float(), \
                sample_cuda["src_imgs"].float(), \
                sample_cuda["ref_intrinsics"], \
                sample_cuda["src_intrinsics"], \
                sample_cuda["ref_extrinsics"], \
                sample_cuda["src_extrinsics"], \
                sample_cuda["depth_min"], \
                sample_cuda["depth_max"])

            # # test code: to show cost volumn
            # cost_reg = outputs["cost_volumn"]
            # depth_save_path = "./cost_confidence/"
            # if not os.path.exists(depth_save_path):
            #     os.makedirs(depth_save_path)
            # cost_cpu = cost_reg.detach().clone().cpu().numpy()
            # np.save(depth_save_path+"cost{}_confidence".format(batch_idx),cost_cpu)
            # del cost_cpu, cost_reg
            
            # # test code: to show cost volumn
            # cost_reg = outputs["depth_est_list"][1]
            # depth_save_path = "./depth_low/"
            # if not os.path.exists(depth_save_path):
            #     os.makedirs(depth_save_path)
            # cost_cpu = cost_reg.detach().clone().cpu().numpy()
            # np.save(depth_save_path+"cost{}_confidence".format(batch_idx),cost_cpu)
            # del cost_cpu, cost_reg

            # # test code: to show cost volumn
            # cost_reg = outputs["first_layer_pro_test"]
            # depth_save_path = "./probability/"
            # if not os.path.exists(depth_save_path):
            #     os.makedirs(depth_save_path)
            # cost_cpu = cost_reg.detach().clone().cpu().numpy()
            # np.save(depth_save_path+"cost{}_probability".format(batch_idx),cost_cpu)
            # del cost_cpu, cost_reg

            # #test code : to show confidences ## no acf no confidence
            # confidences = outputs["confidences"]
            # depth_save_path = "./confidences/"
            # if not os.path.exists(depth_save_path):
            #     os.makedirs(depth_save_path)
            # cost_cpu = confidences[0].detach().clone().cpu().numpy()
            # np.save(depth_save_path+"confidence{}".format(batch_idx),cost_cpu)
            # del cost_cpu, confidences

            

            # depth_searching_range_list = outputs["depth_searching_range_list"] 
            # confidence_save_path = "./depth_range_no_acf/"
            # if not os.path.exists(confidence_save_path):
            #     os.makedirs(confidence_save_path)
            # stage1 = depth_searching_range_list[0]
            # stage1 = stage1.detach().clone().cpu().numpy()
            # np.save(confidence_save_path+"depth_range{}_1".format(batch_idx),stage1)
            # del stage1
            # depth_range_test = torch.stack(depth_searching_range_list[1:],dim=0)
            # depth_range_test = depth_range_test.detach().clone().cpu().numpy()
            # np.save(confidence_save_path+"depth_range{}_2".format(batch_idx),depth_range_test)
            # del depth_range_test

            # Parse output
            depth_est_list = outputs["depth_est_list"]
            depth_est = depth_est_list[0].data.cpu().numpy()
            prob_confidence = outputs["prob_confidence"].data.cpu().numpy()


            # depth_save_path_gao = "./estimated_depth/"
            # if not os.path.exists(depth_save_path_gao):
            #     os.makedirs(depth_save_path_gao)
            # depth_est_list.reverse()
            # for i in range(5):
            #     depth1 = depth_est_list[i]
            #     depth1 = depth1.detach().clone().cpu().numpy()
            #     np.save(depth_save_path_gao+"depth_id{}_stage{}".format(batch_idx,i),depth1)
            #     del depth1
            # del depth_est_list

            # # 输出第一层的cost volume和confidence看看
            # # test code: to show cost volumn
            # cost_reg = outputs["cost_volumn"][0]
            # depth_save_path = "./cost_volume_xiuzheng_traindemo/"
            # if not os.path.exists(depth_save_path):
            #     os.makedirs(depth_save_path)
            # cost_cpu = cost_reg.detach().clone().cpu().numpy()
            # np.save(depth_save_path+"cost{}_confidence".format(batch_idx),cost_cpu)
            # del cost_cpu, cost_reg


            del sample_cuda
            filenames = sample["filename"]
            logger.info('Iter {}/{}, time = {:.3f}'.format(batch_idx, len(test_loader),time.time() - start_time))

            # save depth maps and confidence maps
            for filename, est_depth, photometric_confidence in zip(filenames, depth_est, prob_confidence):
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                save_pfm(depth_filename, est_depth)
                write_depth_img(depth_filename+".png", est_depth)
                # Save prob maps
                save_pfm(confidence_filename, photometric_confidence)

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    return intrinsics, extrinsics


def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data

# read an image
def read_img(filename):
    img = Image.open(filename)
    # Crop image (Hard code dtu image size here)
    left = 0
    top = 0
    right = 1600
    bottom = 1184
    img = img.crop((left, top, right, bottom))

    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.uint8)
    return np_img

def read_downsample_img(filename):
    img = Image.open(filename)
    # Crop image (Hard code dtu image size here)
    left = 0
    top = 0
    right = 800
    bottom = 592
    img = img.crop((left, top, right, bottom))

    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.uint8)
    return np_img

# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)

def save_pfm(filename, image, scale=1):

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()

def write_depth_img(filename,depth):

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    image = Image.fromarray((depth-500)/2).convert("L")
    image.save(filename)
    return 1

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)    # 将srcimg的深度投射到ref_img坐标中
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < 0.5, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(dataset_root, scan, out_folder, plyfilename):

    print("Starting fusion for:"+out_folder)

    # the pair file
    pair_file = os.path.join(dataset_root,'Cameras/pair.txt')
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(dataset_root, 'Cameras/{:0>8}_cam.txt'.format(ref_view)))

        # load the reference image
        # ref_img = read_downsample_img(os.path.join(dataset_root, "Rectified",scan, 'rect_{:03d}_3_r5000.png'.format(ref_view+1))) # Image start from 1.
        ref_img = read_img(os.path.join(dataset_root, "Rectified",scan, 'rect_{:03d}_3_r5000.png'.format(ref_view+1))) # Image start from 1.
        
        # load the estimated depth of the reference view
        ref_depth_est, scale = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))
        # load the photometric mask of the reference view
        confidence, scale = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))
        # 这里的confidence来自于prob_confidence，避免其影响，将其设为1
        photo_mask = confidence > 0.9      #光度一致性假设的confidence
        # photo_mask = np.ones_like(confidence)   #设为1
        # photo_mask = photo_mask > 0

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(dataset_root, 'Cameras/{:0>8}_cam.txt'.format(src_view)))

            # the estimated depth of the source view
            src_depth_est, scale = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                      src_depth_est,
                                                                      src_intrinsics, src_extrinsics)
            
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # at least 3 source views matched
        geo_mask = geo_mask_sum >= 3
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan, ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))


        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        ref_img = np.array(ref_img)
 
        color = ref_img[valid_points]

        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]     # [x,y,z]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)      #每个视角作为ref_img计算出来的三维点
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    print("Saving the final model to", plyfilename)
    PlyData([el], comments=['Model created by CVP-MVSNet.']).write(plyfilename)
    print("Model saved.")


# occumpy gpu
def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    # total, used = devices_info[int(cuda_device)].split(',')
    total, used = devices_info[(cuda_device)].split(',')

    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x


if __name__ == '__main__':


    os.environ["CUDA_VISIBLE_DEVICES"] = occumpy_gpu_list_str
    for i in occumpy_gpu:
        occumpy_gpu_tmp = i
        occumpy_mem(occumpy_gpu_tmp)
        for _ in tqdm(range(3)):
            time.sleep(1)
    print('Memory Occumpy Done')

    # step1. save all the depth maps and the masks in outputs directory
    save_depth()

   
    # Decomment following if you want to try the fusion provided by Xiaoyang Guo

    # step2. fusion
    # testlist = os.path.join(args.dataset_root,"validation_list.txt")
    # testlist = os.path.join(args.dataset_root,"scan_gao_test.txt")
    # # testlist = os.path.join(args.dataset_root,"train_demo.txt")
    # with open(testlist) as f:
    #     scans = f.readlines()
    #     scans = [line.rstrip() for line in scans]

    # for scan in scans:
    #     scan_id = int(scan[4:])
    #     outdir = args.outdir
    #     out_folder = os.path.join(outdir, scan)
    #     print("Start fusion for scan "+str(scan_id))
    #     # filter_depth(args.dataset_root, scan, out_folder, os.path.join(outdir, scan,'.ply'))
    #     filter_depth(args.dataset_root, scan, out_folder, os.path.join(outdir, 'cvpmvsnet{:0>3}_l3.ply'.format(scan_id)))

    # print("completed.")

    # os.environ["CUDA_VISIBLE_DEVICES"] = occumpy_gpu_list_str
    # for i in occumpy_gpu:
    #     occumpy_gpu_tmp = i
    #     occumpy_mem(occumpy_gpu_tmp)
    #     for _ in tqdm(range(4)):
    #         time.sleep(1)
    # print('Memory Occumpy Done')
    
    # CUDA_HOME=$ORIGINAL_CUDA_HOME
    # export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH
    # unset ORIGINAL_CUDA_HOME
    # unset ORIGINAL_LD_LIBRARY_PATH
