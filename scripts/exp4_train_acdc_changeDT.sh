#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/exp4_arr_%A_%a.out      # Standard output
#SBATCH -e logs/exp4_arr_%A_%a.err      # Standard error
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=123    # random seeds: 20,55,77,123,234 
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/exp4_acdc_train_DT_small'  
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"


######## Try with standard and RIRD augmentations ! 
for lr in 0.05
    do
    for budget in 64 #20*16=320 regions, 64 is 20%
        do
        $exec_command python3 -u $code_path/run.py --exp-name "2021-10-26-train_acdc_ImageNetBackbone_budget_${budget}_lr_${lr}_1patient_seed_${SLURM_ARRAY_TASK_ID}" \
        --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
        --ckpt-path $ckpt_path --data-path $data_path \
        --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr $lr --train-batch-size 32 --val-batch-size 8 \
        --patience 50 --region-size 64 64 --epoch-num 99 \
        --rl-episodes 50 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 10 \
        --budget-labels $budget --num-each-iter 4 --al-algorithm 'ralis' --rl-pool 16 
        done
    done


# for lr in 0.05
#     do
#     for budget in 128 #40*16=640 regions, 128 is 20%
#         do
#         $exec_command python3 -u $code_path/run.py --exp-name "2021-10-26-train_acdc_ImageNetBackbone_budget_${budget}_lr_${lr}_2patients" \
#         --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
#         --ckpt-path $ckpt_path --data-path $data_path \
#         --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr $lr --train-batch-size 32 --val-batch-size 8 \
#         --patience 50 --region-size 64 64 --epoch-num 99 \
#         --rl-episodes 50 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 10 \
#         --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 30 
#         done
#     done


# for lr in 0.05
#     do
#     for budget in 192 #60*16=960 regions, 192 is 20%
#         do
#         $exec_command python3 -u $code_path/run.py --exp-name "2021-10-26-train_acdc_ImageNetBackbone_budget_${budget}_lr_${lr}_3patients" \
#         --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
#         --ckpt-path $ckpt_path --data-path $data_path \
#         --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr $lr --train-batch-size 32 --val-batch-size 8 \
#         --patience 50 --region-size 64 64 --epoch-num 99 \
#         --rl-episodes 50 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 10 \
#         --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 30 
#         done
#     done

# for lr in 0.05
#     do
#     for budget in 256 #80*16=1280 regions, 256 is 20%
#         do
#         $exec_command python3 -u $code_path/run.py --exp-name "2021-10-26-train_acdc_ImageNetBackbone_budget_${budget}_lr_${lr}_4patients" \
#         --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
#         --ckpt-path $ckpt_path --data-path $data_path \
#         --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr $lr --train-batch-size 32 --val-batch-size 8 \
#         --patience 50 --region-size 64 64 --epoch-num 99 \
#         --rl-episodes 50 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 10 \
#         --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 30 
#         done
#     done

# for lr in 0.05
#     do
#     for budget in 320 #100*16=1600, 320 is 20% 
#         do
#         $exec_command python3 -u $code_path/run.py --exp-name "2021-10-26-train_acdc_ImageNetBackbone_budget_${budget}_lr_${lr}_5patients" \
#         --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
#         --ckpt-path $ckpt_path --data-path $data_path \
#         --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr $lr --train-batch-size 32 --val-batch-size 8 \
#         --patience 50 --region-size 64 64 --epoch-num 99 \
#         --rl-episodes 50 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 10 \
#         --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 30 
#         done
#     done