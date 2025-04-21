#!/usr/bin/env bash
source scripts/data_path.sh


LOG_DIR="/content/drive/MyDrive/MVS/GeoMVSNet/log"
CKPT_FILE="/content/drive/MyDrive/MVS/GeoMVSNet/model_geomvsnet_release.ckpt"
TNT_OUT_DIR="/content/drive/MyDrive/MVS/GeoMVSNet/outputs"
TNT_ROOT="/content/drive/MyDrive/MVS/tankandtemples"
# Intermediate
# CUDA_VISIBLE_DEVICES=0 python3 test.py ${@} \
#     --which_dataset="tnt" --loadckpt=$CKPT_FILE --batch_size=1 \
#     --outdir=$TNT_OUT_DIR --logdir=$LOG_DIR --nolog \
#     --testpath=$TNT_ROOT --testlist="datasets/lists/tnt/intermediate.txt" --split="intermediate" \
#     \
#     --n_views="11" --img_mode="resize" --cam_mode="origin"

# Advanced
CUDA_VISIBLE_DEVICES=0 python3 /content/GeoMVSNet/test.py ${@} \
    --which_dataset="tnt" --loadckpt=$CKPT_FILE --batch_size=1 \
    --outdir=$TNT_OUT_DIR --logdir=$LOG_DIR --nolog \
    --testpath=$TNT_ROOT --testlist="/content/GeoMVSNet/datasets/lists/tnt/advanced.txt" --split="advanced" \
    \
    --n_views="5" --img_mode="resize" --cam_mode="origin"
