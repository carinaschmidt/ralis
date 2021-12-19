#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-03:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/exp4_eval_%A_%a.out      # Standard output
#SBATCH -e logs/exp4_eval_%A_%a.err      # Standard error
#SBATCH --mail-type=ALL          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=carina.schmidt@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=20,55,77,123,234    # random seeds: 20,55,77,123,234 
scontrol show job $SLURM_JOB_ID 

ckpt_path='/mnt/qb/baumgartner/cschmidt77_data/exp2_acdc_baselines_preTrainDT'  #change ckpt path
data_path='/mnt/qb/baumgartner/cschmidt77_data/'
code_path='/home/baumgartner/cschmidt77/devel/ralis'
sif_path='/home/baumgartner/cschmidt77/ralis.sif'

exec_command="singularity exec --nv --bind $data_path $sif_path"

# produce evaluation niftis
for algo in 'bald'
   do
   for budget in 64 128 592 960 1184 1424 1904 2384 3568  #80*16=1280 regions, 256 is 20%
      do
      $exec_command python3 -u $code_path/evaluate_patients_acdc_originalImages.py --exp-name "2021-11-04-acdc_ImageNetBackbone_baseline_${algo}_budget_${budget}_seed_${SLURM_ARRAY_TASK_ID}" \
      --ckpt-path $ckpt_path --data-path $data_path --checkpointer \
      --dataset 'acdc' --al-algorithm $algo
      done
   done


for algo in 'entropy' 'random' 
   do
   for budget in 64 128 592 960 1184 1424 1904 2384 3568  #80*16=1280 regions, 256 is 20%
      do                                                                       
      $exec_command python3 -u $code_path/evaluate_patients_acdc_originalImages.py --exp-name "2021-11-03-acdc_ImageNetBackbone_baseline_${algo}_budget_${budget}_seed_${SLURM_ARRAY_TASK_ID}" \
      --ckpt-path $ckpt_path --data-path $data_path --checkpointer \
      --dataset 'acdc' --al-algorithm $algo
      done
   done

for algo in 'ralis'
   do
   for budget in 64 128 592 960 1184 1424 1904 2384 3568  #80*16=1280 regions, 256 is 20%
      do                           
      $exec_command python3 -u $code_path/evaluate_patients_acdc_originalImages.py --exp-name "2021-11-04-acdc_${algo}_ImageNetBackbone_budget_${budget}_seed_${SLURM_ARRAY_TASK_ID}" \
      --ckpt-path $ckpt_path --data-path $data_path --checkpointer \
      --dataset 'acdc' --al-algorithm $algo
      done
   done

# calculate mean dice scores
for algo in 'bald'
   do
   for budget in 64 128 592 960 1184 1424 1904 2384 3568  #80*16=1280 regions, 256 is 20%
      do
      $exec_command python3 -u $code_path/calculate3Ddice_patientwise.py --exp-name "2021-11-04-acdc_ImageNetBackbone_baseline_${algo}_budget_${budget}_seed_${SLURM_ARRAY_TASK_ID}" \
      --ckpt-path $ckpt_path --data-path $data_path --checkpointer \
      --dataset 'acdc' --al-algorithm $algo
      done
   done


for algo in 'entropy' 'random' 
   do
   for budget in 64 128 592 960 1184 1424 1904 2384 3568  #80*16=1280 regions, 256 is 20%
      do                                                                       
      $exec_command python3 -u $code_path/calculate3Ddice_patientwise.py --exp-name "2021-11-03-acdc_ImageNetBackbone_baseline_${algo}_budget_${budget}_seed_${SLURM_ARRAY_TASK_ID}" \
      --ckpt-path $ckpt_path --data-path $data_path --checkpointer \
      --dataset 'acdc' --al-algorithm $algo
      done
   done

for algo in 'ralis'
   do
   for budget in 64 128 592 960 1184 1424 1904 2384 3568  #80*16=1280 regions, 256 is 20%
      do                           
      $exec_command python3 -u $code_path/calculate3Ddice_patientwise.py --exp-name "2021-11-04-acdc_${algo}_ImageNetBackbone_budget_${budget}_seed_${SLURM_ARRAY_TASK_ID}" \
      --ckpt-path $ckpt_path --data-path $data_path --checkpointer \
      --dataset 'acdc' --al-algorithm $algo
      done
   done

   

# singularity exec --nv --bind /mnt/qb/baumgartner tue-slurm-helloworld/ralis.sif python3 -u ralis/calculate3Ddice_patientwise_acdc.py --exp-name '2021-10-11-acdc_test_ep49_RIRD_ImageNetBackbone_lr_0.01_budget_128_seed_77'
#  --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'acdc' --al-algo 'ralis'