#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=03-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/res_arr_%A_%a.out      # Standard output
#SBATCH -e logs/err_arr_%A_%a.err      # Standard error
#SBATCH --mail-type=ALL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=123   # array of cityscapes random seeds 50,234,77,12

# print info about current job
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/ckpt_bestTraining/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

### ACDC ###
#### 3rd setting #####
# for lr in 0.01 
#     do
#     for budget in 1424 1904
#         do
#         $exec_command python3 -u $code_path/run.py --exp-name "2021-08-31-3rdsetting_test_acdc_ImageNetBackbone_budget112_seed_123_lr_${lr}_budget_${budget}" \
#         --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
#         --ckpt-path $ckpt_path --data-path $data_path \
#         --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr $lr --train-batch-size 32 --val-batch-size 8 \
#         --patience 50 --region-size 64 64 --epoch-num 500 \
#         --exp-name-toload-rl "2021-08-23-train_acdc_ImageNetBackbone_budget_112_lr0.01_2patients_newAug_3rdsetting" \
#         --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 30 \
#         --modality '2D' \
#         --train --test --final-test
#         done
#     done

# 4th setting
for lr in 0.01 
    do
    for budget in 1424 1904
        do
        $exec_command python3 -u $code_path/run.py --exp-name "2021-09-02-4thsetting_test_acdc_ImageNetBackbone_budget112_seed_123_lr_${lr}_budget_${budget}" \
        --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
        --ckpt-path $ckpt_path --data-path $data_path \
        --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr $lr --train-batch-size 32 --val-batch-size 8 \
        --patience 50 --region-size 64 64 --epoch-num 500 \
        --exp-name-toload '2021-08-19-supervised_ImageNetBackbone_msdHeart234' \
        --exp-name-toload-rl "2021-08-30-train_acdc_ImageNetBackbone_budget_112_lr0.01_2patients_4thsetting_rlep60" \
        --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 30 \
        --modality '2D' \
        --train --test --final-test
        done
    done

    # final test