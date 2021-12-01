#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/exp2_arr_%A_%a.out      # Standard output
#SBATCH -e logs/exp2_arr_%A_%a.err      # Standard error
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=20,55,77,123,234    # random seeds: 20,55,77,123,234 
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/exp2_acdc_train_msd'  
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"


for lr in 0.05 
   do
   for budget in 64 128 592 960 1184 1424 1904 2384 3568 # from D_V = 744*16 = 11904, epoch-num 9
      do
      $exec_command python3 -u $code_path/run.py --full-res --exp-name "2021-10-25-acdc_test_ep49_RIRD_msdHeart_agnostic_lr_${lr}_budget_${budget}_seed_${SLURM_ARRAY_TASK_ID}" \
      --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
      --ckpt-path $ckpt_path --data-path $data_path \
      --input-size 128 128 --only-last-labeled --dataset 'acdc' --modality '2D' \
      --lr $lr --train-batch-size 32 --val-batch-size 8 \
      --patience 30 --region-size 64 64 --epoch-num 49 \
      --rl-episodes 50 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 9 --rl-pool 30 \
      --exp-name-toload-rl "2021-10-13-acdc_train_RIRD_msdHeart_agnostic_budget_608_lr_${lr}_seed_${SLURM_ARRAY_TASK_ID}" \
      --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --newAugmentations --train --test --final-test
      done
   done

for lr in 0.05
   do
   for budget in 64 128 592 960 1184 1424 1904 2384 3568 # from D_V = 744*16 = 11904, epoch-num 9
      do
      $exec_command python3 -u $code_path/run.py --full-res --exp-name "2021-10-25-acdc_test_ep49_stdAug_msdHeart_agnostic_lr_${lr}_budget_${budget}_seed_${SLURM_ARRAY_TASK_ID}" \
      --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
      --ckpt-path $ckpt_path --data-path $data_path \
      --input-size 128 128 --only-last-labeled --dataset 'acdc' --modality '2D' \
      --lr $lr --train-batch-size 32 --val-batch-size 8 \
      --patience 30 --region-size 64 64 --epoch-num 49 \
      --rl-episodes 50 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 9 --rl-pool 30 \
      --exp-name-toload-rl "2021-10-09-acdc_train_stdAug_msdHeart_agnostic_budget_608_lr_${lr}_seed_${SLURM_ARRAY_TASK_ID}" \
      --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --train --test --final-test
      done
   done

# for lr in 0.01 0.05
#    do
#    for budget in 64 128 592 960 1184 1424 1904 2384 3568 # from D_V = 744*16 = 11904, epoch-num 9
#       do
#       $exec_command python3 -u $code_path/run.py --full-res --exp-name "2021-10-11-acdc_test_ep49_stdAug_ImageNetBackbone_lr_${lr}_budget_${budget}_seed_${SLURM_ARRAY_TASK_ID}" \
#       --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
#       --ckpt-path $ckpt_path --data-path $data_path \
#       --input-size 128 128 --only-last-labeled --dataset 'acdc' --modality '2D' \
#       --lr $lr --train-batch-size 32 --val-batch-size 8 \
#       --patience 30 --region-size 64 64 --epoch-num 49 \
#       --rl-episodes 50 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 9 --rl-pool 30 \
#       --exp-name-toload-rl "2021-10-08-acdc_train_stdAug_ImageNetBackbone_budget_608_lr_${lr}_seed_${SLURM_ARRAY_TASK_ID}" \
#       --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --train --test --final-test
#       done
#    done

# for lr in 0.01 0.05
#    do
#    for budget in 64 128 592 960 1184 1424 1904 2384 3568 # from D_V = 744*16 = 11904, epoch-num 9
#       do
#       $exec_command python3 -u $code_path/run.py --full-res --exp-name "2021-10-11-acdc_test_ep49_RIRD_ImageNetBackbone_lr_${lr}_budget_${budget}_seed_${SLURM_ARRAY_TASK_ID}" \
#       --seed ${SLURM_ARRAY_TASK_ID}  --checkpointer \
#       --ckpt-path $ckpt_path --data-path $data_path \
#       --input-size 128 128 --only-last-labeled --dataset 'acdc' --modality '2D' \
#       --lr $lr --train-batch-size 32 --val-batch-size 8 \
#       --patience 30 --region-size 64 64 --epoch-num 49 \
#       --rl-episodes 50 --rl-buffer 600 --lr-dqn 0.001 --dqn-epochs 9 --rl-pool 30 \
#       --exp-name-toload-rl "2021-10-08-acdc_train_RIRD_ImageNetBackbone_budget_608_lr_${lr}_seed_${SLURM_ARRAY_TASK_ID}" \
#       --budget-labels $budget --num-each-iter 16 --al-algorithm 'ralis' --newAugmentations --train --test --final-test
#       done
#    done