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
#SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=<your-email>  # Email to which notifications will be sent
#SBATCH --array=20   # array of cityscapes random seeds 50,234,77,12

# print info about current job
scontrol show job $SLURM_JOB_ID 

ckpt_path='/home/baumgartner/cschmidt77/ckpt_seg/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis/'
sif_path='/home/baumgartner/cschmidt77/deeplearning.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

#### Camvid ####

for al_algorithm in 'random'
    do
    for budget in 196 256 300 480 720 960 1200
        do
        $exec_command python3 -u $code_path/run.py --exp-name 'baseline_acdc_'$al_algorithm'_budget_'$budget'_seed'${SLURM_ARRAY_TASK_ID} \
        --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
        --ckpt-path $ckpt_path --data-path $data_path \
        --load-weights --exp-name-toload 'acdc_pretrained_dt' \
        --input-size 256 256 --only-last-labeled --dataset 'acdc' --lr 0.001 --train-batch-size 32 --val-batch-size 4 \
        --patience 100 --region-size 64 64 \
            --budget-labels $budget --num-each-iter 24 --al-algorithm $al_algorithm --rl-pool 50 --train --test --final-test
        done
    done

for al_algorithm in 'entropy' 'bald'
    do
    for budget in 196 256 300 480 720 960 1200
        do
        $exec_command python3 -u $code_path/run.py --exp-name 'baseline_acdc_'$al_algorithm'_budget_'$budget'_seed'${SLURM_ARRAY_TASK_ID} \
        --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
        --ckpt-path $ckpt_path --data-path $data_path \
        --load-weights --exp-name-toload 'acdc_pretrained_dt' \
        --input-size 256 256 --only-last-labeled --dataset 'acdc' --lr 0.001 --train-batch-size 32 --val-batch-size 4 \
        --patience 100 --region-size 64 64 \
            --budget-labels $budget --num-each-iter 24 --al-algorithm $al_algorithm --rl-pool 10 --train --test --final-test
        done
    done

    


for al_algorithm in 'random'
    do
    for lr in 0.001 0.05
        do
        for budget in 16 48 5000 10000
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

for al_algorithm in 'entropy' 'bald'
    do
    for lr in 0.001 0.05
        do
        for budget in 16 48 5000 10000
            do
            $exec_command python3 -u $code_path/run.py --exp-name '2021-06-30-baseline_acdc_'$al_algorithm'_lr_'$lr'_budget_'$budget'_seed'${SLURM_ARRAY_TASK_ID} \
            --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
            --ckpt-path $ckpt_path --data-path $data_path \
            --load-weights --exp-name-toload '2021-06-28-pretraining_acdc_input_128_lr_0.05_BestPerformance' \
            --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr $lr --train-batch-size 32 --val-batch-size 8 \
            --patience 100 --epoch-num 10 --region-size 64 64 \
            --budget-labels $budget --num-each-iter 16 --al-algorithm $al_algorithm --rl-pool 10 --train --test --final-test
            done
        done
    done








    for al_algorithm in 'random'
    do
    for lr in 0.001 0.05
        do
        for budget in 160 960 1440 2400 11520
            do
            $exec_command python3 -u $code_path/run.py --exp-name "2021-07-05-baseline_acdc_"$al_algorithm"_lr_"$lr"_budget_"$budget"_seed${SLURM_ARRAY_TASK_ID}" \
            --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
            --ckpt-path $ckpt_path --data-path $data_path \
            --load-weights \
            --input-size 128 128 --only-last-labeled --dataset "acdc" --lr $lr --train-batch-size 32 --val-batch-size 8 \
            --patience 100 --epoch-num 10 --region-size 64 64 \
            --budget-labels $budget --num-each-iter 16 --al-algorithm $al_algorithm --rl-pool 10 --test --final-test
            done
        done
    done