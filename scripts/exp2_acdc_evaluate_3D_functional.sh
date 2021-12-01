#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/exp2_eval_%A_%a.out      # Standard output
#SBATCH -e logs/exp2_eval_%A_%a.err      # Standard error
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=20,55,77,123,234    # random seeds: 20,55,77,123,234 
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/exp2_acdc_baselines_lr05'
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

#evaluate 3D baselines
for alalgo in 'entropy' 'bald' 'random'
   do
   for budget in 2384 #20% from 3040
      do
      $exec_command python3 -u $code_path/evaluate_patients_acdc_originalImages.py --checkpointer \
      --exp-name "2021-10-20-acdc_ImageNetBackbone_baseline_${alalgo}_seed${SLURM_ARRAY_TASK_ID}_budget_${budget}" \
      --ckpt-path $ckpt_path --data-path $data_path \
      --dataset 'acdc' --al-algorithm ${alalgo}
      done
   done
