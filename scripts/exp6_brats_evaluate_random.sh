#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=01-00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/exp6_eval_%A_%a.out      # Standard output
#SBATCH -e logs/exp6_eval_%A_%a.err      # Standard error
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=20,55,77,123,234    # random seeds: 20,55,77,123,234 
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/exp1b_brats_baselines/'
data_path='/mnt/qb/baumgartner/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

# Random
for budget in 64 384 1792 3584 17792 #60*16=960 regions, 192 is 20%
    do                                                                                                                                   
    $exec_command python3 -u $code_path/evaluate_patients_brats18.py --exp-name "2021-11-04-brats18_ImageNetBackbone_baseline_random_budget_${budget}_seed_${SLURM_ARRAY_TASK_ID}" --checkpointer \
    --ckpt-path $ckpt_path --data-path $data_path \
    --dataset 'brats18' --al-algorithm 'ralis'
    done

for budget in 64 384 1792 3584 17792  #60*16=960 regions, 192 is 20%
    do                                                                              
    $exec_command python3 -u $code_path/calculate3Ddice_patientwise_brats18.py --exp-name "2021-11-04-brats18_ImageNetBackbone_baseline_random_budget_${budget}_seed_${SLURM_ARRAY_TASK_ID}" --checkpointer \
    --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/_FINAL_brats18_exp6/' --data-path $data_path \
    --dataset 'brats18' --al-algorithm 'ralis'
    done

for budget in 35584  #60*16=960 regions, 192 is 20%
    do                                                                                                                                   
    $exec_command python3 -u $code_path/evaluate_patients_brats18.py --exp-name "2021-11-06-brats18_ImageNetBackbone_baseline_random_budget_${budget}_seed_${SLURM_ARRAY_TASK_ID}" --checkpointer \
    --ckpt-path $ckpt_path --data-path $data_path \
    --dataset 'brats18' --al-algorithm 'ralis'
    done

for budget in 35584   #60*16=960 regions, 192 is 20%
    do                                                                              
    $exec_command python3 -u $code_path/calculate3Ddice_patientwise_brats18.py --exp-name "2021-11-06-brats18_ImageNetBackbone_baseline_random_budget_${budget}_seed_${SLURM_ARRAY_TASK_ID}" --checkpointer \
    --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/_FINAL_brats18_exp6/' --data-path $data_path \
    --dataset 'brats18' --al-algorithm 'ralis'
    done





# for augm in 'stdAug' 'RIRD'  #60*16=960 regions, 192 is 20%
#     do
#     $exec_command python3 -u $code_path/calculate3Ddice_patientwise_acdc.py \
#     --exp-name "2021-11-05-supervised-brats18_DT3_RIRD_ImageNetBackbone_lr_0.01_${SLURM_ARRAY_TASK_ID}" --checkpointer \
#     --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/_FINAL_brats18_exp6/' --data-path $data_path \
#     --dataset 'brats18' --al-algorithm 'ralis'
#     done


# singularity exec --nv --bind /mnt/qb/baumgartner ralis.sif python3 -u devel/ralis/evaluate_patients_brats18_originalImages.py --exp-name "2021-11-05-supervised-brats18_DT3_RIRD_ImageNetBackbone_lr_0.01_20" --checkpointer  --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/exp1b_brats_baselines/' --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'brats18' --al-algorithm 'ralis'

# singularity exec --nv --bind /mnt/qb/baumgartner tue-slurm-helloworld/ralis.sif python3 -u devel/ralis/evaluate_patients_brats18_originalImages.py --exp-name "2021-11-05-supervised-brats18_DT3_RIRD_ImageNetBackbone_lr_0.01_20" --checkpointer  --ckpt_path '/mnt/qb/baumgartner/cschmidt77_data/exp1b_brats_baselines/' --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'brats18' --al-algorithm 'ralis'