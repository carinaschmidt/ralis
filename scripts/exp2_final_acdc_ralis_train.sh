#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/exp4_adam_%A_%a.out      # Standard output
#SBATCH -e logs/exp4_adam_%A_%a.err      # Standard error
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=20,55,77,123,234     # random seeds: 20,55,77,123,234 
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/exp4_acdc_train_DT2_adamW'  
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

for lrdqn in 0.001 0.0001
    do
    for budget in 64 #40*16=640 regions, 128 is 20%
        do
        $exec_command python3 -u $code_path/run.py --exp-name "2021-11-06-train_acdc_AdamW_ImageNetBackbone_budget_${budget}_lrdqn_${lrdqn}_2patients_${SLURM_ARRAY_TASK_ID}" \
        --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
        --ckpt-path $ckpt_path --data-path $data_path \
        --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr 0.05 --train-batch-size 32 --val-batch-size 8 \
        --patience 10 --region-size 64 64 --optimizer 'AdamW' --epoch-num 50 \
        --rl-episodes 100 --rl-buffer 600 --lr-dqn ${lrdqn} --dqn-epochs 10 \
        --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --snapshot 'best_jaccard_val.pth' --rl-pool 30 
        done
    done