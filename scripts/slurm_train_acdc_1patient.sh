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
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=234   # random seeds: 20,77,123,234
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/ckpt_bestTraining/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

### ACDC 1 patient, 19 images, 304 regions, 32 are 10% labelled## 
### load pre-trained seg net on ImagNet and MSD Heart data

# for budget in 32
#     do
#     $exec_command python3 -u $code_path/run.py --exp-name "2021-08-20-train_acdc_budget_${budget}_lr_${lr}_1patient_newAug" \
#     --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
#     --ckpt-path $ckpt_path --data-path $data_path \
#     --load-weights --exp-name-toload '2021-08-19-supervised_ImageNetBackbone_msdHeart234' \
#     --input-size 128 128 --only-last-labeled --dataset 'acdc' --lr 0.01 --train-batch-size 32 --val-batch-size 8 \
#     --patience 50 --region-size 64 64 --epoch-num 1000 \
#     --rl-episodes 10 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 9 \
#     --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 16
#     done


for budget in 960
    do
    $exec_command python3 -u $code_path/run.py --exp-name "2021-09-14-train_brats18_budget_${budget}_lr_${lr}_3patients" \
    --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
    --ckpt-path $ckpt_path --data-path $data_path \
    --load-weights \
    --input-size 128 128 --only-last-labeled --dataset 'brats18' --modality '2D' --lr 0.01 --train-batch-size 32 --val-batch-size 8 \
    --patience 50 --region-size 64 64 --epoch-num 1000 \
    --rl-episodes 10 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 9 \
    --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --rl-pool 16
    done