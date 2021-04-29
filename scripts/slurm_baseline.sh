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
#SBATCH --array=20,50,234,77,12   # array of cityscapes random seeds

# print info about current job
scontrol show job $SLURM_JOB_ID 

ckpt_path='/home/baumgartner/cbaumgartner/ckpt_seg/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cbaumgartner/devel/ralis/'
sif_path='/home/baumgartner/cbaumgartner/deeplearning.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

#### Citysapes ####

for al_algorithm in 'random'
    do
    for budget in 1920 3840 7680 11520 19200 30720
        do
        $exec_command python3 $code_path/run.py --exp-name 'baseline_cityscapes_'$al_algorithm'_budget_'$budget'_seed'${SLURM_ARRAY_TASK_ID} \
        --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
        --ckpt-path $ckpt_path --data-path $data_path \
        --load-weights --exp-name-toload 'cityscapes_pretrained_dt' \
        --input-size 256 512 --only-last-labeled --dataset 'cityscapes' \
            --budget-labels $budget --num-each-iter 256 --al-algorithm $al_algorithm --rl-pool 500 --train --test --final-test
        done
    done

for al_algorithm in 'entropy'
    do
    for budget in 1920 3840 7680 11520 19200 30720
        do
        $exec_command python3 $code_path/run.py --exp-name 'baseline_cityscapes_'$al_algorithm'_budget_'$budget'_seed'${SLURM_ARRAY_TASK_ID} \
        --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
        --ckpt-path $ckpt_path --data-path $data_path \
        --load-weights --exp-name-toload 'cityscapes_pretrained_dt' \
        --input-size 256 512 --only-last-labeled --dataset 'cityscapes' \
            --budget-labels $budget --num-each-iter 256 --al-algorithm $al_algorithm --rl-pool 200 --train --test --final-test
        done
    done

for al_algorithm in 'bald'
    do
    for budget in 1920 3840 7680 11520 19200 30720
        do
        $exec_command python3 $code_path/run.py --exp-name 'baseline_cityscapes_'$al_algorithm'_budget_'$budget'_seed'${SLURM_ARRAY_TASK_ID} \
        --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
        --ckpt-path $ckpt_path --data-path $data_path \
        --load-weights --exp-name-toload 'cityscapes_pretrained_dt' \
        --input-size 256 512 --only-last-labeled --dataset 'cityscapes' \
            --budget-labels $budget --num-each-iter 256 --al-algorithm $al_algorithm --rl-pool 200 --train --test --final-test
        done
    done



#### Camvid ####

for al_algorithm in 'random'
    do
    for budget in 480 720 960 1200 1440 1920
        do
        $exec_command python3 $code_path/run.py --exp-name 'baseline_camvid_'$al_algorithm'_budget_'$budget'_seed'${SLURM_ARRAY_TASK_ID} \
        --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
        --ckpt-path $ckpt_path --data-path $data_path \
        --load-weights --exp-name-toload 'camvid_pretrained_dt' \
        --input-size 224 224 --only-last-labeled --dataset 'camvid' --lr 0.001 --train-batch-size 32 --val-batch-size 4 \
        --patience 150 --region-size 80 90 \
            --budget-labels $budget --num-each-iter 24 --al-algorithm $al_algorithm --rl-pool 50 --train --test --final-test
        done
    done

for al_algorithm in 'entropy' 'bald'
    do
    for budget in 480 720 960 1200 1440 1920
        do
        $exec_command python3 $code_path/run.py --exp-name 'baseline_camvid_'$al_algorithm'_budget_'$budget'_seed'${SLURM_ARRAY_TASK_ID} \
        --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
        --ckpt-path $ckpt_path --data-path $data_path \
        --load-weights --exp-name-toload 'camvid_pretrained_dt' \
        --input-size 224 224 --only-last-labeled --dataset 'camvid' --lr 0.001 --train-batch-size 32 --val-batch-size 4 \
        --patience 150 --region-size 80 90 \
            --budget-labels $budget --num-each-iter 24 --al-algorithm $al_algorithm --rl-pool 10 --train --test --final-test
        done
    done

