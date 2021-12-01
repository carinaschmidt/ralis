#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/exp1b_supDT_%A_%a.out      # Standard output
#SBATCH -e logs/exp1b_supDT_%A_%a.err      # Standard error
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=55,123,234  # random seeds: 20,55,77,123,234 
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/exp1b_brats_baselines'  
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

for lr_rate in 0.01 # std aug, DT=3
   do
   $exec_command python3 $code_path/train_supervised.py --exp-name "2021-11-05-supervised-brats18_DT3_RIRD_ImageNetBackbone_lr_${lr_rate}_${SLURM_ARRAY_TASK_ID}" --checkpointer \
   --checkpointer --ckpt-path $ckpt_path --data-path $data_path \
   --dataset 'brats18' --modality '2D' --input-size 128 128 --lr $lr_rate \
   --train-batch-size 8 --val-batch-size 2 --epoch-num 1500 --patience 40 \
   --test --final-test --snapshot 'best_jaccard_val.pth' --seed ${SLURM_ARRAY_TASK_ID}
   done
