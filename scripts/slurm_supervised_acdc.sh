#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/job_%j.out  # File to which STDOUT will be written
#SBATCH --error=logs/job_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=END,FAIL      # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=123   # array of cityscapes random seeds

# print info about current job
scontrol show job $SLURM_JOB_ID 

# insert your commands here
ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/ckpt_seg_supervised/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

### Pretraining for ACDC ###

for lr_rate in 0.001 0.005 0.01 0.05
    do
    $exec_command python3 $code_path/train_supervised.py --exp-name "2021-08-01-supervised_ralis_acdc_ImageNetBackbone_seed123_input128_lr${lr_rate}" --checkpointer \
    --ckpt-path $ckpt_path --data-path $data_path \
    --input-size 128 128 --dataset 'acdc' \
    --exp-name-toload "2021-07-17-train_acdc_ImageNetBackbone_budget_608_lr_0.01_seed_123" \
    --epoch-num 1500 --lr $lr_rate --train-batch-size 32 --val-batch-size 8 --patience 100 \
    --seed 123 \
    --test --final-test --snapshot 'best_jaccard_val.pth'
    done

for lr_rate in 0.001 0.005 0.01 0.05
    do
    $exec_command python3 $code_path/train_supervised.py --exp-name "2021-08-01-supervised_ralis_acdc_seed123_input256_lr${lr_rate}" --checkpointer \
    --ckpt-path $ckpt_path --data-path $data_path \
    --input-size 256 256 --dataset 'acdc' \
    --exp-name-toload '2021-07-18-supervised_msdHeart0.01' \
    --epoch-num 1500 --lr $lr_rate --train-batch-size 32 --val-batch-size 8 --patience 100 \
    --seed 123 \
    --test --final-test --snapshot 'best_jaccard_val.pth'
    done


for lr_rate in 0.001 0.005 0.01 0.05
    do
    $exec_command python3 $code_path/train_supervised.py --exp-name "2021-08-01-supervised_ralis_acdc_ImageNetBackbone_seed123_input256_lr${lr_rate}" --checkpointer \
    --ckpt-path $ckpt_path --data-path $data_path \
    --input-size 256 256 --dataset 'acdc' \
    --exp-name-toload "2021-07-17-train_acdc_ImageNetBackbone_budget_608_lr_0.01_seed_123" \
    --epoch-num 1500 --lr $lr_rate --train-batch-size 32 --val-batch-size 8 --patience 100 \
    --seed 123 \
    --test --final-test --snapshot 'best_jaccard_val.pth'
    done

for lr_rate in 0.001 0.005 0.01 0.05
    do
    $exec_command python3 $code_path/train_supervised.py --exp-name "2021-08-01-supervised_ralis_acdc_seed123_input128_lr${lr_rate}" --checkpointer \
    --ckpt-path $ckpt_path --data-path $data_path \
    --input-size 128 128 --dataset 'acdc' \
    --exp-name-toload '2021-07-18-supervised_msdHeart0.01' \
    --epoch-num 1500 --lr $lr_rate --train-batch-size 32 --val-batch-size 8 --patience 100 \
    --seed 123 \
    --test --final-test --snapshot 'best_jaccard_val.pth'
    done