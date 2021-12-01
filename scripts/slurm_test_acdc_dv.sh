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

# print info about current job
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/ckpt_seg_acdc/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

### ACDC ###
for lr in 0.05
    do
    for budget in 16
        do
        $exec_command python3 -u $code_path/run.py --exp-name "2021-07-28-test_acdc_lr_${lr}_budget_${budget}" \
        --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
        --load-weights --exp-name-toload "camvid_pretrained_dt" \
        --ckpt-path $ckpt_path --data-path $data_path \
        --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr $lr --train-batch-size 32 --val-batch-size 8 \
        --patience 100 --region-size 64 64 --epoch-num 49 \
        --rl-episodes 5 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 9 \
        --budget-labels $budget --num-each-iter 8 --al-algorithm 'ralis' --rl-pool 30 \
        --train --test --final-test
        done
    done