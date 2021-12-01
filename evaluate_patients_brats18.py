# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

# adapted by Carina for RALIS and BraTS data

import os
import glob
import numpy as np
import logging
import torch
from torch.autograd import Variable
from models.model_utils import create_models

import argparse
#import metrics_acdc
import time
from skimage import transform
import nibabel as nib

import utils
import image_utils
import utils.parser as parser
import h5py    
import pandas as pd

import torch.nn.functional as F

# test images 2D BraTS npy:  /mnt/qb/baumgartner/cschmidt77_data/BraTS2018/TrainingData/volumes_2D/test
# ground truth 3D BraTS npy: /mnt/qb/baumgartner/cschmidt77_data/BraTS2018/TrainingData/volumes/test

# ground truth 2D BraTS npy: /mnt/qb/baumgartner/cschmidt77_data/BraTS2018/TrainingData/gts_2D/test
# ground truth 3D BraTS npy:  /mnt/qb/baumgartner/cschmidt77_data/BraTS2018/TrainingData/gts/test

# flair.nii.gz: /mnt/qb/baumgartner/rawdata/BraTS2018/TrainingData/LGG/Brats18_2013_0_1/Brats18_2013_0_1_flair.nii.gz
# OR: der Reihe nach die 285 patients in: /mnt/qb/baumgartner/rawdata/BraTS2018_train/HGG_and_LGG/
# /Brats18_2013_0_1/Brats18_2013_0_1_flair.nii.gz
# segmentation: /mnt/qb/baumgartner/rawdata/BraTS2018/TrainingData/LGG/Brats18_2013_0_1/Brats18_2013_0_1_seg.nii.gz


# what goes in model: 2D images
# we get 2D predictions
# stack 2D predictions to 3D npy
# convert 3D npy to Nifti
# compare 3D prediction to 3D ground truth


def score_data(root,  pat3D_dir, pat3D_gt_dir,output_folder, exp_config):
    image_size = [160,192]
    test_patient_ids = []
    #258 patient folders
    for i in range(143,257):
        test_patient_ids.append('pat_%s'%i)

    kwargs_models = {"dataset": exp_config.dataset,
                    "al_algorithm": exp_config.al_algorithm,
                    "region_size": exp_config.region_size}
    net, _, _ = create_models(**kwargs_models)

    #load best model from checkpoint folder
    net_checkpoint_path = os.path.join(exp_config.ckpt_path, exp_config.exp_name, 'best_jaccard_val.pth') #ckpt_path and exp_name from parser

    if os.path.isfile(net_checkpoint_path):
        print("net_checkpoint_path: ", net_checkpoint_path)
        print('(Final test) Load best checkpoint for segmentation network!')
        net_dict = torch.load(net_checkpoint_path)
        if len([key for key, value in net_dict.items() if 'module' in key.lower()]) > 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in net_dict.items():
                name = k[7:]  # remove module.
                new_state_dict[name] = v
            net_dict = new_state_dict
        net.load_state_dict(net_dict)
    net.eval()

    # get segmentation ground truth nifti paths
    seg_niftis = []
    pat_niftis = []
    rootdir = '/mnt/qb/baumgartner/rawdata/BraTS2018_train/HGG_and_LGG/'
    print("os.getcwd()", os.getcwd())
    print("rootdir: ", rootdir)
    #print("os.isdir: ", os.path.isdir(rootdir))

    os.listdir('/mnt/qb/baumgartner/rawdata/BraTS2018_train/HGG_and_LGG/')

    print("len(os.listdir(rootdir): ", len(os.listdir(rootdir)))
    for folder in os.listdir(rootdir):
        #print("folder: ", folder)
        folder_dir = os.path.join(rootdir, folder)
        #print(folder_dir)
        for nifti in os.listdir(folder_dir):
            #print("nifti: ", nifti)
            #print("folder_dir ", folder_dir)
            if 'seg' in nifti:
                path_seg_nifti = os.path.join(folder_dir, nifti)
                seg_niftis.append(path_seg_nifti)
            else:
                path_pat_nifti = os.path.join(folder_dir, nifti)
                pat_niftis.append(path_pat_nifti)

    test_seg_niftis_path = seg_niftis[-160:]
    print("test segnifti path: ", len(test_seg_niftis_path))

    test_pat_niftis_path = pat_niftis[-640:]
    print("len(test_pat_niftis:", len(test_pat_niftis_path))

    print("pat3D_Dir: ", len(os.listdir(pat3D_dir)))
    for i, pat in enumerate(os.listdir(pat3D_dir)):
        patient_id = pat.split('.')[0]
        print("patient_id: ", patient_id)
        print("i: ", i)
        mask_dat = load_nii(test_seg_niftis_path[i])
        mask_affine = mask_dat[1]
        mask_header = mask_dat[2]

        # vol dat
        vol_dat = load_nii(test_pat_niftis_path[i])
        vol_affine = vol_dat[1]
        vol_header = vol_dat[2]

        pat_dir = os.path.join(pat3D_dir, pat)
        gt_dir = os.path.join(pat3D_gt_dir, pat)
        pat = np.load(pat_dir)
        gt = np.load(gt_dir)
        print(pat.shape)
        print("np.unique: ", np.unique(gt))
        # save input image as nifti
        out_file_name = os.path.join(output_folder, 'image', patient_id + '.nii.gz')
        print("out_file_name: ", out_file_name)
        save_nii(out_file_name, pat, vol_affine, vol_header)

        #save ground truth as nifti
        out_file_name = os.path.join(output_folder, 'ground_truth', patient_id + '_seg.nii.gz')
        save_nii(out_file_name, gt, mask_affine, mask_header)

        pred_slices = []
        entropies = []
        for sl in range(pat.shape[2]):
            # process into network
            # get prediction
            image = pat[:,:,sl,:]
            image = np.transpose(image, (2,0,1))
            input_img = torch.from_numpy(image) #torch.Size([4, 256, 256])
            #print("input_img: ", input_img)
            if input_img.dim() == 3:
                img_sz = input_img.size()
                input_img = input_img.view(1, img_sz[0], img_sz[1], img_sz[2])
                input_img = Variable(input_img).cuda()
            outputs, _ = net(input_img.float())
            #print("before output.data: ",outputs.min())
            print(outputs.min())
            print(outputs.max())
            #predictions_py = torch.squeeze(outputs) #removes dimension of size 1
            pred_cpu = outputs.cpu()
            pred = np.squeeze(pred_cpu.detach())#[1,4,160,192]

            # get pixel-wise class predictions
            pred_py = F.softmax(outputs, dim=1).data
            pred_py = pred_py.max(1)
            predictions_py = pred_py[1].squeeze_(1).cpu().type(torch.FloatTensor)
            # compute entropy:
            ent = compute_entropy_seg(args, input_img, net)
            entropies.append(ent)
            #prediction_cropped = np.transpose(prediction_cropped, (1,2,0)) #[256,256,4]

            pred_slices.append(predictions_py)
        
        # save entropy list
        entropy_pat = np.stack(entropies, axis=0)
        name = 'entropy_%s.npy' %(patient_id)
        out_file_name = os.path.join(output_folder, 'entropy', name)
        print("out_file_name: ", out_file_name)
        np.save(out_file_name, entropy_pat, allow_pickle=True, fix_imports=True)
            
        pred_vol = np.stack(pred_slices, axis=0)
        # Save prediced mask
        pred_vol = np.transpose(pred_vol,(2,3,0,1)) #[160,4,160,192] to [160,192,160,4]
        print("pred_vol_max: ", pred_vol.max())
        pred_vol = np.where(pred_vol>=3.0, 4.0, pred_vol) 
        print("pred_vol_max: ", pred_vol.max())

        print("pred_vol.shape: ", pred_vol.shape)
        out_file_name = os.path.join(output_folder, 'prediction', patient_id + '_seg.nii.gz')
        print("prediction dir: ", out_file_name)
        save_nii(out_file_name, pred_vol, mask_affine, mask_header)
        
        # calculate dice
        #pred_cpu = predictions_py.cpu()

        #logging.info('Average time per volume: %f' % (total_time/total_volumes))

    #return init_iteration

def load_nii(img_path):
    '''
    Shortcut to load a nifti file
    '''
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):
    '''
    Shortcut to save a nifty file
    '''

    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

def compute_entropy_seg(args, im_t, net):
    '''
    Compute entropy function
    :param args:
    :param im_t:
    :param net:
    :return:
    '''
    net.eval()
    if im_t.dim() == 3:
        im_t_sz = im_t.size()
        im_t = im_t.view(1, im_t_sz[0], im_t_sz[1], im_t_sz[2])

    out, _ = net(im_t)
    out_soft_log = F.log_softmax(out) 
    out_soft = torch.exp(out_soft_log) #max in softmax: exponential of softmax, increases the probability of the biggest score and decreases the probability of the lower score
    ent = - torch.sum(out_soft * out_soft_log, dim=1).detach().cpu()  # .data.numpy()
    del (out)
    del (out_soft_log)
    del (out_soft)
    del (im_t)

    return ent


if __name__ == '__main__':
    args = parser.get_arguments()
    root = '/mnt/qb/baumgartner/cschmidt77_data/BraTS2018/TrainingData/'
    #exp_name = '2021-10-11-supervised-acdc_allPatients_stdAug_ImageNetBackbone_lr0.05_123/' 
    #exp_name = '2021-10-11-acdc_test_ep49_stdAug_ImageNetBackbone_lr_0.05_budget_1904_seed_234/'
    #output_folder = '/mnt/qb/baumgartner/cschmidt77_data/evaluation_BraTS18/' + args.exp_name
    # where the gt volume masks are located
    #post_processing_dir = '/mnt/qb/baumgartner/cschmidt77_data/acdc_postproc/2021-12-15-3D_dice' 
    # where the predictions of loaded model are located
    #pred_dir = os.path.join(post_processing_dir, args.exp_name)

    # checkpoint path
    #seg_net_path = '/mnt/qb/baumgartner/cschmidt77_data/exp2b_brats18_supervised/'

    pat3D = 'volumes/test/'
    pat3D_labels = 'gts/test/'
    pat3D = os.path.join(root, pat3D)
    pat3D_gt = os.path.join(root, pat3D_labels)
    output_folder = '/mnt/qb/baumgartner/cschmidt77_data/_FINAL_brats18_exp6/' + args.exp_name
    # create folder for prediction, ground_truth, image and difference
    folders = ['prediction', 'ground_truth', 'image', 'entropy']
    for f in folders:
        if not os.path.exists(os.path.join(output_folder, f)):
            os.makedirs(os.path.join(output_folder, f))
    score_data(root, pat3D, pat3D_gt, output_folder, args)

# singularity exec --nv --bind /mnt/qb/baumgartner ralis.sif python3 -u devel/ralis/evaluate_patients_brats18_originalImages.py --exp-name '2021-10-20-supervised-brats18_allPatients_stdAug_ImageNetBackbone_lr0.01_123' --checkpointer  --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/exp2b_brats18_supervised'  --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'brats18' --al-algo 'ralis'

  #locally:
  #singularity exec --nv --bind /mnt/qb/baumgartner tue-slurm-helloworld/ralis.sif python3 -u ralis/evaluate_patients_brats18_originalImages.py --exp-name '2021-10-20-supervised-brats18_allPatients_stdAug_ImageNetBackbone_lr0.01_123'  --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'brats18' --al-algo 'ralis'


  #singularity exec --nv --bind /mnt/qb/baumgartner ralis.sif python3 -u devel/ralis/evaluate_patients_brats18.py --exp-name '2021-10-20-supervised-brats18_allPatients_stdAug_ImageNetBackbone_lr0.01_123' --checkpointer  --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/exp2b_brats18_supervised/'  --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'brats18' --al-algo 'ralis'

