#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=00-07:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/exp6_eval_%A_%a.out      # Standard output
#SBATCH -e logs/exp6_eval_%A_%a.err      # Standard error
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=20,55,77,123,234    # random seeds: 20,55,77,123,234 
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/exp2b_brats18_supervised/'  #change ckpt path
data_path='/mnt/qb/baumgartner/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"


# RALIS
for aug in 'stdAug' #60*16=960 regions, 192 is 20%
    do
    for lr in 0.001 0.01 0.05
        do                                                                           
        $exec_command python3 -u $code_path/evaluate_patients_brats18.py --exp-name "2021-10-20-supervised-brats18-allPatients_${aug}_ImageNetBackbone_lr${lr}_${SLURM_ARRAY_TASK_ID}" --checkpointer \
        --ckpt-path $ckpt_path --data-path $data_path \
        --dataset 'brats18' --al-algorithm 'ralis'
        done
    done

for aug in 'stdAug' #60*16=960 regions, 192 is 20%
    do
    for lr in 0.001 0.01 0.05
        do                                                                                                                                   
        $exec_command python3 -u $code_path/calculate3Ddice_patientwise_brats18.py --exp-name "2021-10-20-supervised-brats18-allPatients_${aug}_ImageNetBackbone_lr${lr}_${SLURM_ARRAY_TASK_ID}" --checkpointer \
        --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/_FINAL_brats18_exp6/' --data-path $data_path \
        --dataset 'brats18' --al-algorithm 'ralis'
        done
    done
