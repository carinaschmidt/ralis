
# Reinforced Active Learning for Medical Image Segmentation (RALMIS)
Code (https://github.com/ArantxaCasanova/ralis) from the paper [Reinforced Active Learning for Image Segmentation] (https://arxiv.org/abs/2002.06583) adapted to 3D medical images.

## Dependencies 
- python 3.9
- numpy 1.20.2
- scipy 1.6.2
- Pytorch 1.8.1
- Torchvision 0.9.1
- PIL 8.2.0

## Scripts
The folder 'scripts' contains the different bash scripts that could be used to train the same models used in the master thesis [Deep Reinforcement Learning for Interactive Training of Medical Image Segmentation Networks], for both anatomical cardiac (ACDC) and brain tumour (BraTS2018) segmentation datasets.
- Segmentation networks' backbone pre-trained on ImageNet 
  - a) on the ACDC cardiac dataset: 
    - exp2_final_acdc_baselines_*.sh to train the baseline active learning methods 'random', 'entropy' and 'bald' and train/test the segmentation network
    - exp2_final_acdc_ralis_train.sh to train the reinforcement agent for the 'ralis' model
    - exp2_final_acdc_ralis_test.sh to test the reinforcement agent for the 'ralis' model and train/test the segmentation network
  - b) on the BraTS2018 brain tumour dataset:
    - exp2b_brats_baselines_*.sh to train the baseline active learning methods 'random', 'entropy' and 'bald' and train/test the segmentation network
    - exp2b_brats18_train.sh to train the reinforcement agent for the 'ralis' model
    - exp2b_brats18_test.sh to test the reinforcement agent for the 'ralis' model and train/test the segmentation network
    
Furthermore the folder 'scripts' contains the bash scripts that could be used to train the same models used in the paper from Casanova et al., for both Camvid and Cityscapes datasets. 
- launch_supervised.sh: To train the pretrained segmentation models. 
- launch_baseline.sh: To train the baselines 'random', 'entropy' and 'bald'.
- launch_train_ralis.sh: To train the 'ralis' model.
- launch_test_ralis.sh: To test the 'ralis' model. 

## Datasets
Our investigated datasets:
- Anatomical cardiac structure segmentation dataset [ACDC]: https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html
- Pathological brain tumour segmentation dataset [BraTS18]: https://www.med.upenn.edu/sbia/brats2018/data.html

## Trained models
To download the trained reinforcement learning agent on the ACDC and BraTS datasets: https://drive.google.com/drive/folders/1SkBdh5HVZsO8Og8dgFSIVKvYiCASCGIY?usp=sharing

## Notes by C Baumgartner for use in Tuebingen ML Cloud

### Additions to base code:
 - add number of slurm job files to `scripts`
 - add a template for running an interactive training to `scripts`
 - make `code_path` an additional argument to prevent hard coded img_paths
 - remove `dataset` from all hardcoded `path` variables in dataset classes

### TODOs to make code run
 - Change paths in `utils/parser.py`
 - Change paths in `scripts/slurm_*` and `scripts/interative_slurm_call_debug.sh` to match system

### Run code on Tue ML Coud Slurm
To run ralis training execute 

````
sbatch devel/ralis/scripts/slurm_train_ralis.sh
````

To run stuff depending on the pre-trained models, don't forget to download them from the Google Drive link below and copy into your checkpoints folder (e.g. `ckpt_seg`)

## Citation
If you use this code, please cite the original paper:
```
@inproceedings{
Casanova2020Reinforced,
title={Reinforced active learning for image segmentation},
author={Arantxa Casanova and Pedro O. Pinheiro and Negar Rostamzadeh and Christopher J. Pal},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=SkgC6TNFvr}
}
```
