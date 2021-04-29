#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=logs/hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=<your-email>  # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID 

ckpt_path='/home/baumgartner/cbaumgartner/ckpt_seg/'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cbaumgartner/devel/ralis/'
sif_path='/home/baumgartner/cbaumgartner/deeplearning.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

### Cityscapes ###
for seed in 20 50 234 77 12
    do
    $exec_command python3 -u $code_path/run.py --exp-name 'RALIS_cs_train_seed'$seed --full-res --region-size 128 128 \
     --snapshot 'best_jaccard_val.pth' --al-algorithm 'ralis' \
    --ckpt-path $ckpt_path --data-path $data_path \
    --dataset 'cityscapes' --lr 0.0001 --train-batch-size 16 --val-batch-size 1 --patience 10 \
    --input-size 256 512 --only-last-labeled --budget-labels 3840  --num-each-iter 256  --rl-pool 20 --seed $seed
    done


### Camvid ###
for seed in 20 50 82 12 4560
    do
    $exec_command python3 -u $code_path/run.py --exp-name 'RALIS_camvid_train_seed'$seed --full-res --region-size 80 90 \
    --snapshot 'best_jaccard_val.pth' --al-algorithm 'ralis' \
    --ckpt-path $ckpt_path --data-path $data_path \
    --rl-episodes 100 --rl-buffer 600 --lr-dqn 0.001\
    --load-weights --exp-name-toload 'gta_pretraining_camvid' \
    --dataset 'camvid' --lr 0.001 --train-batch-size 32 --val-batch-size 4 --patience 10 \
    --input-size 224 224 --only-last-labeled --budget-labels 480  --num-each-iter 24  --rl-pool 20 --seed $seed
    done

