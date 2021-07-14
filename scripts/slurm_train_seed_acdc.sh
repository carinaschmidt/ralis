#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-v100   # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/res_arr_%A_%a.out      # Standard output
#SBATCH -e logs/err_arr_%A_%a.err      # Standard error
#SBATCH --mail-type=END,FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/ckpt_seg_new/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

### ACDC ###
$exec_command python3 -u $code_path/acdc.py --exp-name 'RALIS_acdc_train_seed'$seed --full-res --region-size 64 64 \
--snapshot 'best_jaccard_val.pth' --al-algorithm 'ralis' \
--ckpt-path $ckpt_path --data-path $data_path \
--rl-episodes 100 --rl-buffer 600 --lr-dqn 0.001 \
--load-weights \ 
--dataset 'acdc' --lr 0.001 --train-batch-size 32 --val-batch-size 4 --patience 10 \
--input-size 256 256 --only-last-labeled --budget-labels 300  --num-each-iter 20  --rl-pool 20


scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/ckpt_seg_new/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

#### ACDC ####

for al_algorithm in 'random'
    do
    for lr in 0.001 0.05
        do
        for budget in 11744
            do
            $exec_command python3 -u $code_path/run.py --exp-name '2021-06-30-baseline_acdc_'$al_algorithm'_lr_'$lr'_budget_'$budget'_seed'${SLURM_ARRAY_TASK_ID} \
            --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
            --ckpt-path $ckpt_path --data-path $data_path \
            --load-weights --exp-name-toload '2021-06-28-pretraining_acdc_input_128_lr_0.05_BestPerformance' \
            --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr $lr --train-batch-size 32 --val-batch-size 8 \
            --patience 100 --epoch-num 10 --region-size 64 64 \
            --budget-labels $budget --num-each-iter 16 --al-algorithm $al_algorithm --rl-pool 50 --train --test --final-test
            done
        done
    done