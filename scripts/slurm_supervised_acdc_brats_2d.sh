#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/brats2d_%j.out  # File to which STDOUT will be written
#SBATCH --error=logs/brats2d_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=ALL      # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=123   # array of cityscapes random seeds

# print info about current job
scontrol show job $SLURM_JOB_ID 

# insert your commands here
ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/ckpt_seg_supervised_brats_2d/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"
### Pretraining for Brats ###

for lr_rate in 0.01
    do
    $exec_command python3 $code_path/train_supervised.py --exp-name "2021-09-27-supervised-brats18_4ch_Aug_lr${lr_rate}" --checkpointer \
    --ckpt-path $ckpt_path --data-path $data_path \
    --dataset 'brats18' --modality '2D' --input-size 128 128 \
    --train-batch-size 32 --val-batch-size 8 --epoch-num 3000 --patience 300 \
    --test --final-test --snapshot 'best_jaccard_val.pth'
    done

for lr_rate in 0.01
    do
    $exec_command python3 $code_path/train_supervised.py --exp-name "2021-09-27-supervised-acdc_4ch_Aug_lr${lr_rate}" --checkpointer \
    --ckpt-path $ckpt_path --data-path $data_path \
    --dataset 'acdc' --modality '2D' --input-size 128 128 \
    --train-batch-size 32 --val-batch-size 8 --epoch-num 3000 --patience 300 \
    --test --final-test --snapshot 'best_jaccard_val.pth'
    done