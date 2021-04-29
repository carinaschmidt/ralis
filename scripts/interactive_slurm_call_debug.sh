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
           --exp-name 'RALIS_camvid_test_seed'$seed \
           --al-algorithm 'ralis' \
           --checkpointer  \
           --region-size 80 90 \
           --ckpt-path $ckpt_path \
           --data-path $data_path \
           --load-weights \
           --dataset 'camvid' \
           --lr 0.001 \
           --train-batch-size 32 \
           --val-batch-size 4 \
           --patience 150 \
           --input-size 224 224 \
           --only-last-labeled \
           --budget-labels $budget \
           --num-each-iter 24 \
           --rl-pool 10 --seed $seed \
           --train \
           --test \
           --final-test \
           --exp-name-toload-rl 'RALIS_camvid_train_seed'$seed


