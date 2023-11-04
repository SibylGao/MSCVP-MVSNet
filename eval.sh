# Shell script for evaluating the CVP-MVSNet
# by: Jiayu Yang
# date: 2019-08-29

# Dataset
DATASET_ROOT="/mnt/sharedisk/gaoshiyu/datasets/dtu-test-1200/"

# Checkpoint
# LOAD_CKPT_DIR="./checkpoints/pretrained/model_000027.ckpt"
LOAD_CKPT_DIR="/mnt/sharedisk/gaoshiyu/CVP2/checkpoints/cvp_stage3_acf_stage2/model_000039.ckpt"


# Logging
LOG_DIR="./logs/"

# Output dir
OUT_DIR="./outputs_pretrained_test/"

CUDA_VISIBLE_DEVICES=4 python3 eval.py \
\
--info="eval_pretrained_e27" \
--mode="test" \
\
--dataset_root=$DATASET_ROOT \
--imgsize=1200 \
--nsrc=4 \
--nscale=5 \
\
--batch_size=1 \
\
--loadckpt=$LOAD_CKPT_DIR \
--logckptdir=$CKPT_DIR \
--loggingdir=$LOG_DIR \
\
--outdir=$OUT_DIR