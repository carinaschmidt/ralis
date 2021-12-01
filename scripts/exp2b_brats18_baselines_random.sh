#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1             # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/exp2_baseline_%A_%a.out      # Standard output
#SBATCH -e logs/exp2_baseline_%A_%a.err      # Standard error
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=20,55,77,123,234    # random seeds: 20,55,77,123,234 
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/exp1b_brats_baselines'  
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

for al_algorithm in 'random'
   do
   for budget in 35584 #64 384 1792 3584 17792 
       do
       $exec_command python3 -u $code_path/run.py --exp-name "2021-11-06-brats18_ImageNetBackbone_baseline_${al_algorithm}_budget_${budget}_seed_${SLURM_ARRAY_TASK_ID}" \
       --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
       --ckpt-path $ckpt_path --data-path $data_path \
       --input-size 128 128 --only-last-labeled --dataset 'brats18' --lr 0.01 --train-batch-size 8 --val-batch-size 2 \
       --patience 40 --region-size 40 48 --epoch-num 49 \
       --exp-name-toload "2021-11-03-supervised-brats18_DT3_stdAug_ImageNetBackbone_lr_0.01_234" --load-weights \
       --budget-labels $budget --num-each-iter 64 --al-algorithm $al_algorithm --rl-pool 200 --train --test --final-test
       done
   done