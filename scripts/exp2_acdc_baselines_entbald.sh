#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=10-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti-long    # Partition to submit to
#SBATCH --gres=gpu:1             # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/exp2_baseline_%A_%a.out      # Standard output
#SBATCH -e logs/exp2_baseline_%A_%a.err      # Standard error
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=20,55,77,123    # random seeds: 20,55,77,123,234 
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/exp2_acdc_baselines_lr05'  
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

for al_algorithm in 'entropy' 'bald'
   do
   for budget in 64 128 592 960 1184 1424 1904 2384 3568
       do
       $exec_command python3 -u $code_path/run.py --exp-name "2021-10-20-acdc_ImageNetBackbone_baseline_"${al_algorithm}"_seed"${SLURM_ARRAY_TASK_ID}"_budget_${budget}" \
       --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
       --ckpt-path $ckpt_path --data-path $data_path \
       --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr 0.05 --train-batch-size 32 --val-batch-size 4 \
       --patience 100 --region-size 64 64 --epoch-num 49 \
       --budget-labels $budget --num-each-iter 16 --al-algorithm $al_algorithm --rl-pool 30 --train --test --final-test
       done
   done

