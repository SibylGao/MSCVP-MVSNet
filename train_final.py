# Train the CVP-MVSNet
# by: Jiayu Yang
# date: 2019-08-13

import os,sys,time,logging,datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

# from dataset import dtu_jiayu
from dataset import dtu_satge3_jiayu as dtu_jiayu
# from dataset import blended_jiayu as dtu_jiayu
# from dataset import blended_train_jiayu as dtu_jiayu

# 在这里选择跑哪个模型
# from models import net
# from models import cvp_gwc_acf as net
# from models import cvp_uas as net
# from models import cvp_stage3 as net
# from models import cvp_gwc_short_long_range as net
# from models import cvp_stage3_acf_rang2_unet as net
# from models import cvp_satge3_acf_range2 as net
from models import cvp_gwc_acf_range2_final as net
# from models import cvp_stage3_acf_longrange as net
# from models import cvp_shortlongrange_adaptive_acf as net
# from models import cvp_gwc_fast_confidence as net

from utils import *
from argsParser import getArgsParser,checkArgs
import torch.utils
import torch.utils.checkpoint
from tqdm import tqdm

torch.set_num_threads(20)
# Debug import
# import pdb
# import matplotlib.pyplot as plt

# Arg parser
parser = getArgsParser()
args = parser.parse_args()
assert args.mode == "train"
checkArgs(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark=True

# Check checkpoint directory
if not os.path.exists(args.logckptdir+args.info.replace(" ","_")):
    try:
        os.makedirs(args.logckptdir+args.info.replace(" ","_"))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

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
occumpy_gpu = [0]
occumpy_gpu_list_str = ','.join(map(str, occumpy_gpu)) 

# Dataset
train_dataset = dtu_jiayu.MVSDataset(args, logger)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=16, drop_last=True)

# Network
model = net.network(args)
logger.info(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
# model = nn.DataParallel(model)  

# GPU parallel
gpu_list = [0]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
model = nn.DataParallel(model, device_ids=[0])


model.cuda()
model.train()

# Loss
model_conf_loss = net.mvsnet_sum_loss_final      # confidence loss
# model_depth_conf_loss = net.depthmap_confidence    # depth map confidence only
# model_acf_confidencepropagation = net.mvsnet_sum_loss_confidence
# model_conf_loss_uniform = net.mvsnet_sum_loss_uniform
if args.loss_function == "sl1":
    logger.info("Using smoothed L1 loss")
    model_loss = net.sL1_loss
else: # MSE
    logger.info("Using MSE loss")
    model_loss = net.MSE_loss

optimizer = optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

# load network parameters
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    logger.info("Resuming or testing...")
    saved_models = [fn for fn in os.listdir(sw_path) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # use the latest checkpoint file
    loadckpt = os.path.join(sw_path, saved_models[-1])
    logger.info("Resuming "+loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    logger.info("loading model {}".format(args.loadckpt))

    # 正常load模型
    # state_dict = torch.load(args.loadckpt)
    # model.load_state_dict(state_dict['model'])

    ##load部分网络
    pretrained_dict=torch.load(args.loadckpt)
    model_dict=model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict['model'].items() if k in model_dict}
    #debug
    # for k, v in pretrained_dict.items():
    #     print(k,v)
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    #debug
    # print(model.state_dict())

# Start training
logger.info("start at epoch {}".format(start_epoch))
logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)      #step 
    last_loss = None
    this_loss = None
    for epoch_idx in range(start_epoch, args.epochs):
        logger.info('Epoch {}:'.format(epoch_idx))
        global_step = len(train_loader) * epoch_idx

        if last_loss is None:
            last_loss = 999999
        else:
            last_loss = this_loss
        this_loss = []

        for batch_idx, sample in enumerate(train_loader):

            start_time = time.time()
            global_step = len(train_loader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            
            loss = train_sample(sample, epoch_idx, batch_idx, detailed_summary=do_summary)
            this_loss.append(loss)

            # logger.info('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx,
            #                                                                          len(train_loader), loss,
            #                                                                          time.time() - start_time))

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logckptdir+args.info.replace(" ","_"), epoch_idx + 18  ))
            logger.info("model_{:0>6}.ckpt saved".format(epoch_idx))
        this_loss = np.mean(this_loss)
        logger.info("Epoch loss: {:.5f} --> {:.5f}".format(last_loss, this_loss))

        lr_scheduler.step()


def train_sample(sample, epoch_idx, batch_idx, detailed_summary=False):

    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    ref_depths = sample_cuda["ref_depths"]

    outputs = model(\
    sample_cuda["ref_img"].float(), \
    sample_cuda["src_imgs"].float(), \
    sample_cuda["ref_intrinsics"], \
    sample_cuda["src_intrinsics"], \
    sample_cuda["ref_extrinsics"], \
    sample_cuda["src_extrinsics"], \
    sample_cuda["depth_min"], \
    sample_cuda["depth_max"])

    depth_est_list = outputs["depth_est_list"]

    # variables for confidence loss
    # list版本的输出
    costs_list = outputs["cost_volumn"]
    confidences_list = outputs["confidences"]
    variances_list = outputs["variances"]
    cost_reg_list = outputs["cost_volumn"]
    # max_depthmap_list = outputs["max_depthmap"]
    # min_depthmap_list = outputs["min_depthmap"]


    # max_depth_i = sample_cuda["depth_max"][0]
    # min_depth_i = sample_cuda["depth_min"][0]
    # max_depth = max_depth_i.item()
    # min_depth = min_depth_i.item()

    depth_index = outputs["depth_index"]
    max_depth = 1065.0
    min_depth = 425.0

    # depthmap_confidences = outputs["depthmap_confidences"]

    #  # test code: to show cost volumn
    # cost_reg = outputs["depth_est_list"][1]
    # depth_save_path = "./depth_low/"
    # if not os.path.exists(depth_save_path):
    #     os.makedirs(depth_save_path)
    # cost_cpu = cost_reg.detach().clone().cpu().numpy()
    # np.save(depth_save_path+"epoch{}_batch{}_depthmap".format(epoch_idx, batch_idx),cost_cpu)
    # del cost_cpu, cost_reg

    # cost_reg = outputs["confidences"][0]
    # confidence_save_path = "./confidence/"
    # if not os.path.exists(confidence_save_path):
    #     os.makedirs(confidence_save_path)
    # cost_cpu = cost_reg.detach().clone().cpu().numpy()
    # np.save(confidence_save_path+"epoch{}_batch{}_confidencemap".format(epoch_idx, batch_idx),cost_cpu)
    # del cost_cpu, cost_reg

    # # 初代版本的输出
    # max_depth_i = sample_cuda["depth_max"][0]
    # min_depth_i = sample_cuda["depth_min"][0]
    # max_depth = max_depth_i.item()
    # min_depth = min_depth_i.item()
    # costs = output["cost_volumn"]
    # variances = output["variances"]
    # confidences = output["confidences"]

    dHeight = ref_depths.shape[2]
    dWidth = ref_depths.shape[3]
    loss = []
    weight = [2.0, 1.0, 0.5]
    for i in range(0,args.nscale):
        depth_gt = ref_depths[:,i,:int(dHeight/2**i),:int(dWidth/2**i)]
        # mask = depth_gt>425
        mask = depth_gt>0

        #第二层 long range
        if i==1:
            # loss.append(model_conf_loss(costs_list[i], depth_est_list[i], depth_gt.float(), \
            # variances_list[i], mask, max_depthmap_list[i], min_depthmap_list[i], confidences_list[i]))

            # loss.append(model_conf_loss(costs_list[0], depth_est_list[i], depth_gt.float(), \
            # variances_list[0], mask, max_depth, min_depth, confidences_list[0])* weight[i])
            
            loss.append(model_conf_loss(costs_list[0], depth_est_list[i], depth_gt.float(), \
            variances_list[0], mask, depth_index, confidences_list[0]) * weight[i])


            # loss.append(model_loss(depth_est_list[i], depth_gt.float(), mask) * weight[i])   #只含有confidence上采样

            

            # # uniform
            # loss.append(model_conf_loss_uniform(costs_list[0], depth_est_list[i], depth_gt.float(), \
            # mask, max_depth, min_depth))

        #第二层
        else:
            gao_secend_layer_loss = model_loss(depth_est_list[i], depth_gt.float(), mask)
            # gao_secend_layer_loss = gao_secend_layer_loss * 0.3
            loss.append(gao_secend_layer_loss * weight[i])
            # secend_layer_confidence = model_depth_conf_loss(depth_gt.float(),depth_est_list[i],depthmap_confidences[0])
            # loss.append(secend_layer_confidence*100)  #深度图confidence相关的loss，只在第一层上采样之后有


        # #第二层
        # if i==0:
        #     loss.append(model_conf_loss(costs_list[i], depth_est_list[i], depth_gt.float(), \
        #     variances_list[i], mask, max_depthmap_list[i], min_depthmap_list[i], confidences_list[i]))
        # #第一层
        # else:
        #     loss.append(model_loss(depth_est_list[i], depth_gt.float(), mask))

        # loss.append(model_conf_loss(costs_list[i], depth_est_list[i], depth_gt.float(), \
        #     variances_list[i], mask, max_depthmap_list[i], min_depthmap_list[i], confidences_list[i]))


        # # no confidence loss
        # loss.append(model_loss(depth_est_list[i], depth_gt.float(), mask))

        # # confidence added
        # if i == args.nscale - 1:
        #     # mvsnet_sum_loss(costs, depth_est, depth_gt, variances, mask, max_depth, min_depth, confidence)
        #     loss.append(model_conf_loss(costs, depth_est_list[i], depth_gt.float(), \
        #         variances, mask, max_depth, min_depth, confidences))
        # else:
        #     # no confidence
        #     loss.append(model_loss(depth_est_list[i], depth_gt.float(), mask))


    loss = sum(loss)

    loss.backward()

    optimizer.step()

    return loss.data.cpu().item()

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
    if args.mode == "train":
        # # # occumpy memory
        # os.environ["CUDA_VISIBLE_DEVICES"] = occumpy_gpu_list_str
        # for i in occumpy_gpu:
        #     occumpy_gpu_tmp = i
        #     occumpy_mem(occumpy_gpu_tmp)
        #     for _ in tqdm(range(60)):
        #         time.sleep(1)
        # print('Memory Occumpy Done')

        train()

    if args.mode == "train":
        # occumpy mem
        os.environ["CUDA_VISIBLE_DEVICES"] = occumpy_gpu_list_str
        for i in occumpy_gpu:
            occumpy_gpu_tmp = i
            occumpy_mem(occumpy_gpu_tmp)
            for _ in tqdm(range(60)):
                time.sleep(1)
        print('Memory Occumpy Done')