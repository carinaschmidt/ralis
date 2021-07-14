
# Reinforced Active Learning for Image Segmentation (RALIS)
Code for the paper [Reinforced Active Learning for Image Segmentation](https://arxiv.org/abs/2002.06583)

## Additional Notes by C Baumgartner for use in Tue ML Cloud

### Additions to base code:
 - add number of slurm job files to `scripts`
 - add a template for running an interactive training to `scripts`
 - make `code_path` an additional argument to prevent hard coded img_paths
 - remove `dataset` from all hardcoded `path` variables in dataset classes
 - Remove foor loop over seeds in scripts by a slurm array (for version without slurm arrays checkout commit `ee71c2e325165eef32768f62e5e79d81a237a7e2`)


### Todo's to make code run
 - Change paths in `utils/parser.py`
 - Change paths in `scripts/slurm_*` and `scripts/interative_slurm_call_debug.sh` to match system

### Run code on Tue ML Coud Slurm

To run ralis training with 5 random seeds execute 

````
sbatch devel/ralis/scripts/slurm_train_ralis.sh
````

This works on my system. 

Use the other `scripts/slurm_*` scripts to run other stuff. For some reason baselines didn't run on my system, without throwing an error. Jobs just got hung. Perhaps, not enough memory given in SBATCH settings?

To run stuff depending on the pre-trained models, don't forget to download them from the Google Drive link below and copy into your checkpoints folder (e.g. `ckpt_seg`)

## Dependencies 
- python 3.6.5
- numpy 1.14.5
- scipy 1.1.0
- Pytorch 0.4.0

## Scripts
The folder 'scripts' contains the different bash scripts that could be used to train the same models used in the paper, for both Camvid and Cityscapes datasets. 
- launch_supervised.sh: To train the pretrained segmentation models. 
- launch_baseline.sh: To train the baselines 'random', 'entropy' and 'bald'.
- launch_train_ralis.sh: To train the 'ralis' model.
- launch_test_ralis.sh: To test the 'ralis' model. 

## Datasets
Camvid: https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid

Cityscapes: https://www.cityscapes-dataset.com/

## Trained models
To download the trained RALIS models for Camvid and Cityscapes (as well as the pretrained segmentation model on GTA and D_T subsets): https://drive.google.com/file/d/13C4e0bWw6SEjTAD7JdAfLGVz7p7Veeb9/view?usp=sharing
## Citation
If you use this code, please cite:
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
