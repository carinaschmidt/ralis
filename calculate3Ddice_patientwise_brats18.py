# Author:
# Carina Schmidt

from posixpath import lexists
import torch
import numpy as np
import os
import glob
import utils.parser as parser
from models.model_utils import create_models
from torch.autograd import Variable
import nibabel as nib
from skimage import transform

from sklearn.metrics import f1_score
from utils.logger import Logger
import matplotlib.pyplot as plt
import time


def calculate_and_save_dice_score_3D(post_processing_dir):#, pred_dir):
    """calculate dice score between 3D volumes and save to file in given directory

    Args:
        post_processing_dir (str): directory where gt is located
        pred_dir (str): path to predicted slices
    Returns:
        path to log file
    """    ''''''
    log_columns = ['Patient frame', 'Mean Dice per volume', 'background', 'NCR/NET', 'ED', 'ET']
    log_file_path = os.path.join(post_processing_dir, '3D_dice_score.txt')
    logger = Logger(log_file_path)
    logger.set_names(log_columns)
    i = 0
    #iterate over all frames in predicted folder

    mean_dice_score_all_volumes = []
    dsc_background = []
    dsc_cl1 = []
    dsc_cl2 = []
    dsc_cl3 = []
    pred_dir = os.path.join(post_processing_dir, 'prediction')
    gt_dir = os.path.join(post_processing_dir, 'ground_truth')

    for pred_vol_path in glob.glob(os.path.join(pred_dir, 'pat_???_seg.nii.gz')):
        # load gt mask volume
        pred_frame = pred_vol_path.split("/")[-1]
        print("pred_frame: ", pred_frame)
        pat_id = pred_frame.split(".")[0].split("_")[1] #get pat id
        print("pat_id: ", pat_id)
        mask_vol_path = os.path.join(gt_dir, pred_frame)


        if os.path.isfile(mask_vol_path) and os.path.isfile(pred_vol_path):
            print("reading files")
            mask_vol, aff, header = load_nii(mask_vol_path)
            pred_vol, aff, header = load_nii(pred_vol_path)
            # read file
        else:
            print("file does not exist")
            raise ValueError("%s isn't a file!" % mask_vol_path)
           
        #calculate dice score        
        print("calculating dice score")
        # import ipdb
        # ipdb.set_trace()
        pred_vol = np.uint8(pred_vol)
        mask_vol = np.uint8(mask_vol)
        mask_unique = np.unique(mask_vol)
        print("unique mask values: ", mask_unique)
        pred_vol = np.squeeze(pred_vol)
        dice_score = calc_f1_score(pred_vol, mask_vol)

        if len(mask_unique) == 4:
            dsc_cl1.append(dice_score[1])
            dsc_cl2.append(dice_score[2])
            dsc_cl3.append(dice_score[3])
            cl1 = dice_score[1]
            cl2 = dice_score[2]
            cl3 = dice_score[3]
        
        # only background and one other class
        elif len(mask_unique) == 2:
            if 1 in mask_unique:
                cl1 = dice_score[1]
                dsc_cl1.append(cl1)
                cl2 = np.nan
                dsc_cl2.append(cl2)
                cl3 = np.nan
                dsc_cl3.append(cl3)
            elif 2 in mask_unique:
                cl2 = dice_score[1]
                dsc_cl2.append(cl2)
                cl1 = np.nan
                dsc_cl1.append(cl1)
                cl3 = np.nan
                dsc_cl3.append(cl3)
            elif 3 not in mask_unique:
                cl3 = dice_score[1]
                dsc_cl3.append(cl3)
                cl1 = np.nan
                dsc_cl1.append(cl1)
                cl2 = np.nan
                dsc_cl2.append(cl2)
            else:
                print("something went wrong in len==2! ")
        
        # only background in mask
        elif len(mask_unique) == 1:
            cl1 = np.nan
            cl2 = np.nan
            cl3 = np.nan
            dsc_cl1.append(cl1)
            dsc_cl2.append(cl2)
            dsc_cl3.append(cl3)

        elif len(mask_unique) == 3:
            if 1 not in mask_unique:
                cl1 = np.nan
                dsc_cl1.append(cl1)
                cl2 = dice_score[1]
                dsc_cl2.append(cl2)
                cl3 = dice_score[2]
                dsc_cl3.append(cl3)
            elif 2 not in mask_unique:
                cl2 = np.nan
                dsc_cl2.append(cl2)
                cl1 = dice_score[1]
                dsc_cl1.append(cl1)
                cl3 = dice_score[2]
                dsc_cl3.append(cl3)
            elif 4 not in mask_unique:
                cl3 = np.nan
                dsc_cl3.append(cl3)
                cl1 = dice_score[1]
                dsc_cl1.append(cl1)
                cl2 = dice_score[2]
                dsc_cl2.append(cl2)
            else:
                print("something went wrong in len==2! ")
        else:
            print("something went wrong! ")

        dsc_background.append(dice_score[0])
        dice_score = [dice_score[0], cl1, cl2, cl3]
        mean_dice_per_volume = np.nanmean(dice_score)
        #write score to log file
        print("dice_score: ", dice_score)
        mean_dice_score_all_volumes.append(mean_dice_per_volume)
        
        i+=1
        info = [int(pat_id), mean_dice_per_volume, dice_score[0], dice_score[1], dice_score[2], dice_score[3]]
        logger.append(info)
    logger.close()

    mean_mean_dice_score_all_volumes = np.nanmean(np.array(mean_dice_score_all_volumes))
    mean_dsc_background = np.nanmean(np.array(dsc_background))
    mean_dsc_cl1 = np.nanmean(np.array(dsc_cl1))
    mean_dsc_cl2 = np.nanmean(np.array(dsc_cl2))
    mean_dsc_cl3 = np.nanmean(np.array(dsc_cl3))

    #calculate means and save to separate file
    mean_dice_file_path = os.path.join(post_processing_dir, 'Mean_3D_DSC.txt')
    print("writing to log file")
    with open(mean_dice_file_path, 'w') as file_mean_dsc:
        file_mean_dsc.write('Mean DSC All volumes \t Background \t NCR/NET \t ED \t ET \n')
        file_mean_dsc.write(str(mean_mean_dice_score_all_volumes) + '\t' + str(mean_dsc_background) + '\t' + str(mean_dsc_cl1) + '\t' + str(mean_dsc_cl2) + '\t' + str(mean_dsc_cl3))

    return log_file_path

def load_nii(img_path):
    '''
    Shortcut to load a nifti file
    '''
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def calc_f1_score(predictions_mask,gt_mask):
    '''
    to compute f1/dice score
    input params:
        predictions_arr: predicted segmentation mask
        mask: ground truth mask
    returns:
        f1_val: f1/dice score
    '''
    y_pred= predictions_mask.flatten()
    print("y_pred: ", y_pred.shape)
    y_true= gt_mask.flatten()
    print("y_true: ", y_true.shape)
    f1_val= f1_score(y_true, y_pred, average=None)

    return f1_val

def _toEvaluationOneHot(self, labels):
    ### all labels not equal to 0 is whole tumour (WT)
    ### all labels not equal to 0 and not 2 is TC
    shape = labels.shape

    out = np.zeros([shape[0], shape[1], 3], dtype=np.float32)
    #print("out.shape: ", out.shape)
    out[:, :, 0] = (labels != 0) #RoI WT (TC and ED)
    out[:, :, 1] = (labels != 0) * (labels != 2) #TC (everything besides 0 and 2)
    out[:, :, 2] = (labels == 4) # ET
    return out


if __name__ == '__main__':
    args = parser.get_arguments()
    post_processing_dir = args.ckpt_path
    exp_name = args.exp_name
    post_processing = os.path.join(post_processing_dir, exp_name)
    print("post_processing path: ", post_processing)

    calculate_and_save_dice_score_3D(post_processing)

    # singularity exec --nv --bind /mnt/qb/baumgartner ralis.sif python3 -u devel/ralis/calculate3Ddice_patientwise_acdc.py --exp-name '2021-10-11-supervised-acdc_allPatients_stdAug_ImageNetBackbone_lr0.05_123' --checkpointer  --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/exp2_acdc_supervised_dice'  --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'acdc' --al-algo 'ralis'



    #singularity exec --nv --bind /mnt/qb/baumgartner tue-slurm-helloworld/ralis.sif python3 -u ralis/calculate3Ddice_patientwise_acdc.py --exp-name '2021-10-11-acdc_test_ep49_RIRD_ImageNetBackbone_lr_0.01_budget_128_seed_77'  --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'acdc' --al-algo 'ralis'

    #singularity exec --nv --bind /mnt/qb/baumgartner tue-slurm-helloworld/ralis.sif python3 -u ralis/calculate3Ddice_patientwise_acdc.py --exp-name '2021-11-07-test_acdc_ImageNetBackbone_budget_3568_lr_0.05_3patients_seed_123'  --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'acdc' --al-algo 'ralis'

        #### evaluation classes
