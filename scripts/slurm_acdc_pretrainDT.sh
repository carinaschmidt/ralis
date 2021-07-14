#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/res_arr_%A_%a.out      # Standard output
#SBATCH -e logs/err_arr_%A_%a.err      # Standard error
#SBATCH --mail-type=END,FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=234   # array of cityscapes random seeds 50,234,77,12

scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/ckpt_seg_acdc/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

## Pretraining on ACDC D_t  sb
for lr in 0.0005 0.005 0.01 0.05
    do
    $exec_command python3 -u $code_path/train_supervised.py --exp-name "2021-07-08-acdc_pretrained_Dt_lr${lr}_inputsize_128128_noscheduler" --checkpointer \
    --al-algorithm 'ralis' --ckpt-path $ckpt_path --data-path $data_path \
    --input-size 128 128 --dataset 'acdc' \
    --epoch-num 1500 --lr $lr --train-batch-size 32 --val-batch-size 8 --patience 30 \
    --test --final-test --snapshot 'best_jaccard_val.pth'
    done