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

ckpt_path='/home/baumgartner/cschmidt77/ckpt_seg/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis/'
sif_path='/home/baumgartner/cschmidt77/deeplearning.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

### Cityscapes ###
for budget in 1920 3840 7680 11520 19200 30720
    do
    $exec_command python3 -u $code_path/run.py --exp-name 'RALIS_cs_test_seed'${SLURM_ARRAY_TASK_ID} \
    --al-algorithm 'ralis' --checkpointer\
    --ckpt-path $ckpt_path --data-path $data_path \
    --load-weights --exp-name-toload 'cityscapes_pretrained_dt' \
    --dataset 'cityscapes' --lr 0.0001 --train-batch-size 16 --val-batch-size 1 --patience 60 \
    --input-size 256 512 --only-last-labeled --budget-labels $budget  --num-each-iter 256  --rl-pool 100 --seed ${SLURM_ARRAY_TASK_ID} \
    --train --test --final-test --exp-name-toload-rl 'RALIS_cs_train_seed'${SLURM_ARRAY_TASK_ID}
    done

### Camvid ###
for budget in 480 720 960 1200 1440 1920
    do
    $exec_command python3 -u $code_path/run.py --exp-name 'RALIS_camvid_test_seed'${SLURM_ARRAY_TASK_ID} \
    --al-algorithm 'ralis' --checkpointer --region-size 80 90 \
    --ckpt-path $ckpt_path --data-path $data_path \
    --load-weights --exp-name-toload 'camvid_pretrained_dt' \
    --dataset 'camvid' --lr 0.001 --train-batch-size 32 --val-batch-size 4 --patience 150 \
    --input-size 224 224 --only-last-labeled --budget-labels $budget  --num-each-iter 24  --rl-pool 10 --seed ${SLURM_ARRAY_TASK_ID} \
    --train --test --final-test --exp-name-toload-rl 'RALIS_camvid_train_seed'${SLURM_ARRAY_TASK_ID}
    done
