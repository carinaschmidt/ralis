#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-00:05            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti-dev # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=logs/hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=<your-email>  # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID 

# insert your commands here

ckpt_path='/home/baumgartner/cbaumgartner/ckpt_seg'
data_path='/mnt/qb/baumgartner/cschmidt77_data'
code_path='/home/baumgartner/cbaumgartner/devel/ralis/'

for budget in 480 #720 960 1200 1440 1920
    do
    for seed in 20 #50 82 12 4560
        do
        singularity exec --nv --bind /mnt/qb/baumgartner /home/baumgartner/cbaumgartner/deeplearning.sif \
        python3 -u $code_path/run.py \
        --exp-name 'RALIS_camvid_test_seed'$seed \
        --al-algorithm 'ralis' \
        --checkpointer \
        --region-size 80 90 \
        --ckpt-path $ckpt_path \
        --data-path $data_path \
        --load-weights \
        --exp-name-toload 'camvid_pretrained_dt' \
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
        done
    done
