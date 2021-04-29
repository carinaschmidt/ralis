#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=logs/hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=<your-email>  # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID 

# insert your commands here

ckpt_path="/home/baumgartner/$USER/ckpt_seg/";
data_path="/mnt/qb/baumgartner/cschmidt77_data/";
code_path="/home/baumgartner/$USER/devel/ralis/";

sif_path="/home/baumgartner/$USER/deeplearning.sif"

exec_command="singularity exec --nv --bind $data_path $sif_path"


## GTA pretraining for Cityscapes
$exec_command python3 $code_path/train_supervised.py --exp-name 'gta_pretraining_cs' --checkpointer \
--ckpt-path $ckpt_path --data-path $data_path \
--input-size 256 512 --dataset 'gta' \
--epoch-num 1500 --lr 0.0001 --train-batch-size 16 --val-batch-size 1 --patience 30 \
--test --snapshot 'best_jaccard_val.pth'

## Pretraining GTA + Cityscapes D_t (lower bound, what is used as a starting point for the active learning algorithm)
$exec_command python3 $code_path/train_supervised.py --exp-name 'cityscapes_pretrained_dt' --checkpointer \
--ckpt-path $ckpt_path --data-path $data_path \
--load-weights --exp-name-toload 'gta_fpn_baseline_lr1e-3_fullres_finetune' \
--input-size 256 512 --dataset 'cityscapes_subset' \
--epoch-num 1500 --lr 0.0001 --train-batch-size 16 --val-batch-size 1 --patience 50 \
--test --snapshot 'best_jaccard_val.pth'

#Upper bound Cityscapes
$exec_command python3 $code_path/train_supervised.py --exp-name 'cityscapes_upperbound' --checkpointer \
--ckpt-path $ckpt_path --data-path $data_path \
--load-weights --exp-name-toload 'gta_fpn_baseline_lr1e-3_fullres_finetune' \
--input-size 256 512 --dataset 'cs_upper_bound' \
--epoch-num 1500 --lr 0.0001 --train-batch-size 16 --val-batch-size 1 --patience 50 \
--test --snapshot 'best_jaccard_val.pth'

## GTA pretraining for Camvid
$exec_command python3 $code_path/train_supervised.py --exp-name 'gta_pretraining_camvid' --checkpointer \
--ckpt-path $ckpt_path --data-path $data_path \
--input-size 224 224 --dataset 'gta_for_camvid' \
--epoch-num 1500 --lr 0.005 --train-batch-size 32 --val-batch-size 8 --patience 30 --scale-size 480 \
--test --snapshot 'best_jaccard_val.pth'

## Pretraining GTA + Camvid D_t (lower bound, what is used as a starting point for the active learning algorithm)
$exec_command python3 $code_path/train_supervised.py --exp-name 'camvid_pretrained_dt' --checkpointer \
--ckpt-path $ckpt_path --data-path $data_path \
--load-weights --exp-name-toload 'gta_pretraining_camvid' \
--input-size 224 224 --dataset 'camvid_subset' \
--epoch-num 1500 --lr 0.0005 --train-batch-size 32 --val-batch-size 8 --patience 30 \
--test --snapshot 'best_jaccard_val.pth'

## Upper bound Camvid
$exec_command python3 $code_path/train_supervised.py --exp-name 'camvid_upperbound' --checkpointer \
--ckpt-path $ckpt_path --data-path $data_path \
--load-weights --exp-name-toload 'gta_pretraining_camvid' \
--input-size 224 224 --dataset 'camvid' \
--epoch-num 1500 --lr 0.0005 --train-batch-size 32 --val-batch-size 8 --patience 30 \
--test --snapshot 'best_jaccard_val.pth'

