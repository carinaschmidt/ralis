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
#SBATCH --array=123   # array of cityscapes random seeds: 20 50 234 77 12

# print info about current job
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/ckpt_seg_acdc/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

### ACDC ###

for budget in 160 400 800 1600 6400 10000
    do
    $exec_command python3 -u $code_path/run.py --exp-name '2021-07-02-test_ralis_dt_seed123_budget'$budget \
    --al-algorithm 'ralis' --checkpointer --region-size 64 64 \
    --ckpt-path $ckpt_path --data-path $data_path \
    --load-weights --exp-name-toload '2021-06-28-pretraining_acdc_input_128_lr_0.05_BestPerformance' \
    --dataset 'acdc' --lr 0.001 --train-batch-size 32 --val-batch-size 8 --patience 100 \
    --snapshot 'best_jaccard_val.pth' --al-algorithm 'ralis' --checkpointer --region-size 64 64 \
    --input-size 128 128 --only-last-labeled --budget-labels $budget --num-each-iter 16  --rl-pool 10 --seed 123 \
    --train --test --final-test --exp-name-toload-rl '2021-07-01-train_ralis_acdc_DT_budget500_seed_20_lr_0.001'    
    done