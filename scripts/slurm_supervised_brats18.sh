#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:3              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/job_%j.out  # File to which STDOUT will be written
#SBATCH --error=logs/job_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=ALL      # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=123   # array of cityscapes random seeds

# print info about current job
scontrol show job $SLURM_JOB_ID 

# insert your commands here
ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/ckpt_seg_supervised_brats/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

### Pretraining for ACDC ###

for lr_rate in 0.01
    do
    $exec_command python3 $code_path/train_supervised.py --exp-name "2021-09-07-supervised-brats18" --checkpointer \
    --ckpt-path $ckpt_path --data-path $data_path \
    --dataset 'brats18' --modality '3D' \
    --train-batch-size 2 --val-batch-size 1 \
    --test --final-test --snapshot 'best_jaccard_val.pth'
    done