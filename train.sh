# Shell script for training the CVP-MVSNet
# by: Jiayu Yang
# date: 2019-08-13

# Dataset
# DATASET_ROOT="/mnt/sharedisk/gaoshiyu/dtu-train-128/"
DATASET_ROOT="/mnt/sharedisk/gaoshiyu/datasets/dtu-train-128/"


# Logging
CKPT_DIR="./checkpoints/"
LOG_DIR="./logs/"

python3 train_final.py \
\
--info="train_dtu_128" \
--mode="train" \
\
--dataset_root=$DATASET_ROOT \
--imgsize=128 \
--nsrc=2 \
--nscale=3 \
\
--epochs=40 \
--lr=0.001 \
--lrepochs="10,12,14,20:2" \
--batch_size=8 \
\
--loadckpt='model_000039.ckpt' \        # try this ckpt
--logckptdir=$CKPT_DIR \
--loggingdir=$LOG_DIR \
--resume=0 \
