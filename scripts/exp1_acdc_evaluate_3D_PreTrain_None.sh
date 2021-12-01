#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-01:30            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/exp2_eval_%A_%a.out      # Standard output
#SBATCH -e logs/exp2_eval_%A_%a.err      # Standard error
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=20,55,77,123,234   # random seeds: 20,55,77,123,234 
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/exp2_acdc_train_ralis_None/'  # change ckpt path
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

# for budget in 64 128 592 960 1184 1424 1904 2384 3568
#     do     
#     $exec_command python3 -u $code_path/evaluate_patients_acdc_originalImages.py --exp-name "2021-11-08-acdc_test_RIRD_budget_${budget}_lr_0.05_seed_${SLURM_ARRAY_TASK_ID}" --checkpointer \
#     --ckpt-path $ckpt_path --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  \
#     --dataset 'acdc' --al-algorithm 'ralis'
#     done

for budget in 3568
    do         
    $exec_command python3 -u $code_path/evaluate_patients_acdc_originalImages.py --exp-name "2021-11-08-acdc_test_stdAug_budget_${budget}_lr_0.05_seed_${SLURM_ARRAY_TASK_ID}" --checkpointer \
    --ckpt-path $ckpt_path --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  \
    --dataset 'acdc' --al-algorithm 'ralis'
    done

for budget in 64 128 592 960 1184 1424 1904 2384 3568
    do     
    $exec_command python3 -u $code_path/calculate3Ddice_patientwise.py --exp-name "2021-11-08-acdc_test_RIRD_budget_${budget}_lr_0.05_seed_${SLURM_ARRAY_TASK_ID}" --checkpointer \
    --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/_FINAL_ACDC_exp1_NoPretrain/' --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  \
    --dataset 'acdc' --al-algorithm 'ralis'
    done


for budget in 64 128 592 960 1184 1424 1904 2384 3568
    do     
    $exec_command python3 -u $code_path/calculate3Ddice_patientwise.py --exp-name "2021-11-08-acdc_test_stdAug_budget_${budget}_lr_0.05_seed_${SLURM_ARRAY_TASK_ID}" --checkpointer \
    --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/_FINAL_ACDC_exp1_NoPretrain/' --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  \
    --dataset 'acdc' --al-algorithm 'ralis'
    done