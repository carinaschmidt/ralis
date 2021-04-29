#!/bin/bash
seed=1;
budget=480;
ckpt_path="/home/baumgartner/$USER/ckpt_seg/";
data_path="/mnt/qb/baumgartner/cschmidt77_data/";
code_path="/home/baumgartner/$USER/devel/ralis/";

sif_path="/home/baumgartner/$USER/deeplearning.sif"

exec_command="singularity exec --nv --bind $data_path $sif_path"

srun --pty --partition=gpu-2080ti-dev --gres=gpu:1 --mem=50G $exec_command \
        python3 -u $code_path/run.py \
        --exp-name 'RALIS_cs_train_debug' \
        --full-res \
        --region-size 128 128 \
        --snapshot 'best_jaccard_val.pth' \
        --al-algorithm 'ralis' \
        --ckpt-path $ckpt_path \
        --data-path $data_path \
        --dataset 'cityscapes' \
        --lr 0.0001 \
        --train-batch-size 16 \
        --val-batch-size 1 \
        --patience 10 \
        --input-size 256 512 \
        --only-last-labeled \
        --budget-labels 3840  \
        --num-each-iter 256 \
        --rl-pool 20 \
        --seed 1
