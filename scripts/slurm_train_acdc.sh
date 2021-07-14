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
#SBATCH --mail-type=END,FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=234   # array of cityscapes random seeds: 20 50 234 77 12
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/ckpt_seg_acdc/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

### ACDC ###rl
for lr in 0.05
    do
    for lrdqn in 0.0001 0.001 0.01
        do
        $exec_command python3 -u $code_path/run.py --exp-name "2021-07-09-train_ralis_acdc_DT_seed_234_budget_1920_lr_0.05_lrdqn_${lrdqn}" --full-res --region-size 64 64 \
        --snapshot 'best_jaccard_val.pth' --al-algorithm 'ralis' \
        --ckpt-path $ckpt_path --data-path $data_path \
        --rl-episodes 100 --rl-buffer 600 --lr-dqn $lrdqn --epoch-num 9 \
        --dataset 'acdc' --lr $lr --train-batch-size 32 --val-batch-size 4 --patience 20 \
        --input-size 128 128 --only-last-labeled --budget-labels 480 --num-each-iter 16  --rl-pool 10 --seed ${SLURM_ARRAY_TASK_ID} \
        --final-test
        done
    done