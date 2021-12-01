#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:3              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/res_arr_%A_%a.out      # Standard output
#SBATCH -e logs/err_arr_%A_%a.err      # Standard error
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=234   # random seeds: 20,77,123,234
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/ckpt_bestTraining/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

### ACDC 2 patients, 38 images
### total number of regions: 608
### 122 regions are 18% labelled during training
### num-each-iter*rl-pool = 150
### total regions >= rl-pool*num_each_iter
### budget-labels <= num-each-iter*rl-pool (=num_regions) ?
### load pre-trained seg net on ImagNet and MSD Heart data

# for budget in 112
#     do
#     $exec_command python3 -u $code_path/run.py --exp-name "2021-08-22-train_acdc_budget_${budget}_lr0.01_2patients_newAug" \
#     --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
#     --ckpt-path $ckpt_path --data-path $data_path \
#     --load-weights --exp-name-toload '2021-08-19-supervised_ImageNetBackbone_msdHeart234' \
#     --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr 0.01 --train-batch-size 32 --val-batch-size 8 \
#     --patience 50 --region-size 64 64 --epoch-num 1 \
#     --rl-episodes 100 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 10 \
#     --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 30 --only-last-labeled
#     done


# using 10% of 2 patients -> 64 regions
# for budget in 112
#     do
#     $exec_command python3 -u $code_path/run.py --exp-name "2021-08-24-train_acdc_budget_${budget}_2patients_newAug_1stsetting" \
#     --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
#     --ckpt-path $ckpt_path --data-path $data_path  \
#     --load-weights --exp-name-toload '2021-08-19-supervised_ImageNetBackbone_msdHeart234' \
#     --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr 0.01 --train-batch-size 32 --val-batch-size 8 \
#     --patience 20 --region-size 64 64 --epoch-num 20 \
#     --rl-episodes 100 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 1 \
#     --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 20 --only-last-labeled --snapshot 'last_jaccard_val.pth'
#     done

# for budget in 112 
#     do
#     $exec_command python3 -u $code_path/run.py --exp-name "2021-08-25-train_acdc_ImageNetBackbone_budget_${budget}_lr0.01_2patients_3rdsetting" \
#     --seed ${SLURM_ARRAY_TASK_ID} --checkpointer --ckpt-path $ckpt_path --data-path $data_path \
#     --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr 0.01 --train-batch-size 32 --val-batch-size 8 \
#     --patience 50 --region-size 64 64 --epoch-num 20 --rl-episodes 100 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 10 \
#     --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 30 --only-last-labeled
#     done 

# for budget in 32
#     do
#     $exec_command python3 -u $code_path/run.py --exp-name "2021-08-25-train_acdc_ImageNetBackbone_budget_${budget}_lr0.01_1patient_3rdsetting" \
#     --seed ${SLURM_ARRAY_TASK_ID} --checkpointer --ckpt-path $ckpt_path --data-path $data_path \
#     --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr 0.01 --train-batch-size 32 --val-batch-size 8 \
#     --patience 30 --region-size 64 64 --epoch-num 10 --rl-episodes 100 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 10 \
#     --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 8 --only-last-labeled
#     done 

# for budget in 64
#     do
#     $exec_command python3 -u $code_path/run.py --exp-name "2021-08-25-2-train_acdc_ImageNetBackbone_budget_${budget}_lr0.01_2patients_3rdsetting" \
#     --seed ${SLURM_ARRAY_TASK_ID} --checkpointer --ckpt-path $ckpt_path --data-path $data_path \
#     --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr 0.01 --train-batch-size 16 --val-batch-size 4 \
#     --patience 30 --region-size 64 64 --epoch-num 10 --rl-episodes 60 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 10 \
#     --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 16 --only-last-labeled
#     done 

# for budget in 112
#     do
#     $exec_command python3 -u $code_path/run.py --exp-name "2021-09-02-5thsetting-train_acdc_ImageNetBackbone_agnostic_budget_112_lr0.01_2patients_rlep60" \
#     --seed ${SLURM_ARRAY_TASK_ID} --checkpointer --ckpt-path $ckpt_path --data-path $data_path \
#     --exp-name-toload '2021-08-19-supervised_ImageNetBackbone_msdHeart234' \
#     --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr 0.01 --train-batch-size 32 --val-batch-size 8 \
#     --patience 30 --region-size 64 64 --epoch-num 10 --rl-episodes 60 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 10 \
#     --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 16 --only-last-labeled
#     done 


for lr in 0.01 
    do
    for budget in 64 128 960 1424 1904
        do
        $exec_command python3 -u $code_path/run.py --exp-name "2021-09-02-5thsetting_test_agnostic_acdc_ImageNetBackbone_budget112_seed_123_lr_${lr}_budget_${budget}" \
        --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
        --ckpt-path $ckpt_path --data-path $data_path \
        --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr $lr --train-batch-size 32 --val-batch-size 8 \
        --patience 50 --region-size 64 64 --epoch-num 500 \
        --exp-name-toload '2021-08-19-supervised_ImageNetBackbone_msdHeart234' \
        --exp-name-toload-rl "2021-09-02-5thsetting-train_acdc_ImageNetBackbone_agnostic_budget_112_lr0.01_2patients_rlep60" \
        --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 30 \
        --modality '2D' \
        --train --test --final-test   
        done
    done
#### set train to false because already performed, final test missing


# for lr in 0.01 
#     do
#     for budget in 64 128 960
#         do
#         $exec_command python3 -u $code_path/run.py --exp-name "2021-08-30-4thsetting_test_acdc_ImageNetBackbone_budget112_seed_123_lr_${lr}_budget_${budget}" \
#         --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
#         --ckpt-path $ckpt_path --data-path $data_path \
#         --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr $lr --train-batch-size 32 --val-batch-size 8 \
#         --patience 50 --region-size 64 64 --epoch-num 1000 \
#         --exp-name-toload '2021-08-19-supervised_ImageNetBackbone_msdHeart234' \
#         --exp-name-toload-rl "2021-08-30-train_acdc_ImageNetBackbone_budget_112_lr0.01_2patients_4thsetting_rlep60" \
#         --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 30 \
#         --train --test --final-test
#         done
#     done



# rl-pool can't be larger than budget

# for budget in 112
#     do
#     $exec_command python3 -u $code_path/run.py --exp-name "2021-08-23-train_acdc_ImageNetBackbone_budget_${budget}_lr0.01_2patients_newAug" \
#     --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
#     --ckpt-path $ckpt_path --data-path $data_path  \
#     --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr 0.01 --train-batch-size 32 --val-batch-size 8 \
#     --patience 20 --region-size 64 64 --epoch-num 1 \
#     --rl-episodes 50 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 1 \
#     --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 30 --only-last-labeled
#     done