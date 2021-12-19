# Author:
# Carina Schmidt (carina.schmidt@mail.de)

import os
import glob

import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import append
from scipy._lib.doccer import doc_replace
from skimage.util.dtype import dtype_limits
import torch 
from PIL import Image
from torch.utils import data 
import logging
import nibabel as nib
import gc
from skimage import transform

from pathlib import Path

import utils_acdc

# Dictionary to translate a diagnosis into a number
# NOR  - Normal
# MINF - Previous myiocardial infarction (EF < 40%)
# DCM  - Dialated Cardiomypopathy
# HCM  - Hypertrophic cardiomyopathy
# RV   - Abnormal right ventricle (high volume or low EF)
diagnosis_dict = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}

file_list = []
info_list = []
# save diagnoses of patients for splitting
diag_list = []
cardiac_phase_list = []

#containing frame images
img_list = []
#containing masks
mask_list = []


def load_patient_frames(input_folder, pat_train_test_val, size, target_resolution):
    '''
    Function to load frames and gt images in two folders, also store patient related information to list
    '''
    # iterate over all patients
    # save patient related infos in list respectively 
    # for each patient, load Info.cfg, patient###_4d.nii.gz, patient###_frame01.nii.gz, patient###_frame01_gt.nii.gz, patient###_frame12.nii.gz, patient###_frame12_gt.nii.gz
    #npy_file = np.save(input_folder)

    # iterate over all patient folders
    for folder in os.listdir(input_folder):
        pat_id = int(folder[-3:])
        # whole path to patient
        folder_path = os.path.join(input_folder, folder)

        if os.path.isdir(folder_path):
            # splitting patients into train, test, val (regarding diseases)
            infos = {}
            for line in open(os.path.join(folder_path, 'Info.cfg')):
                label, value = line.split(':')
                if label == 'Group':
                    # get diagnosis code from dict 
                    infos[label] = diagnosis_dict[value.rstrip('\n').lstrip(' ')]
                    # first 20 patients: diagnosis group2, next 20 diag 3, next diag 1, diag 0, diag 4
                    diag_list.append(infos[label])
                else:
                    infos[label] = value.rstrip('\n').lstrip(' ')
            # systole_frame = int(infos['ES'])
            # diastole_frame = int(infos['ED'])

            #patient_id = folder.lstrip('patient')
            infos['patientID'] = pat_id
            info_list.append(infos)

            #iterate over all frame files 
            #file contains full path to image
            for file in glob.glob(os.path.join(folder_path, 'patient???_frame??*.nii.gz')):
                file_list.append(file)
                # systole_frame = int(infos['ES'])
                # diastole_frame = int(infos['ED'])
                
                #directory and file name until dot
                file_dir_name = file.split('.')[0]
                # gives frame number (and gt)
                frame = file_dir_name.split('frame')[-1]
                nii_data = utils_acdc.load_nii(file)
                # nimg.get_data() is in nii_data[0]
                img_data = nii_data[0].copy()   

                # for frames we get (15625, 15625, 10) as pixel size
                pixel_size = (nii_data[2].structarr['pixdim'][1],
                            nii_data[2].structarr['pixdim'][2],
                            nii_data[2].structarr['pixdim'][3])           

                # only for ground truth masks
                if 'gt' in frame:
                    mask = img_data
                    mask_list.append(mask)
                else:
                    # normalise image IMAGES ARE ALL BLACK WITH NORMALISATION
                    # makes image zero mean and unit standard deviation
                    #img = utils_acdc.normalise_image(img_data)
                    img = img_data
                    img_list.append(img)

                ############# processing slice-by-slice 2D data##########
                #############saving images slice-by-slice################
                scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1]]
                nx, ny = size
                # iterate over # of third component of image shape (216,256,10) here 10 times
                # gives #z 2D images from 3D array
                for z in range(img.shape[2]):
                    # removes z axis along array, makes shape (216,256)         
                    if 'gt' in file:
                        #added as type uint8
                        slice_mask = np.squeeze(mask[:, :, z])
                        
                        #Scale image by a certain factor.Performs interpolation to up-scale or down-scale N-dimensional images.
                        mask_rescaled = transform.rescale(slice_mask,
                                                             scale_vector,
                                                             order=0, #check if order has to be 0 or 1 
                                                             preserve_range=True,
                                                             multichannel=False,
                                                             mode='constant') 

                        mask_cropped = crop_or_pad_slice_to_size(mask_rescaled, nx, ny).astype(np.uint8)
                        mask_list.append(mask_cropped)
                        data_file_name = 'pat_%s_diag_%s_frame_%s_slice_%s.npy' % (str(pat_id), infos['Group'], str(frame[0:2]), str(z))
                        # determines to which data split to save
                        data_split = pat_train_test_val[pat_id]
                        # save mask to gt folder
                        data_file_path = os.path.join(preprocessing_folder, 'gt', data_split, data_file_name)
                        # create new npy file to save data splits
                        np.save(data_file_path, mask_cropped)        


                    else: 
                        # makes shape (256,256)
                        slice_img = np.squeeze(img[:, :, z])
                        #Scale image by a certain factor.Performs interpolation to up-scale or down-scale N-dimensional images.
                        slice_rescaled = transform.rescale(slice_img,
                                                            scale_vector,
                                                            order=1,
                                                            preserve_range=True,
                                                            multichannel=False,
                                                            mode = 'constant')
                        slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
                        slice_cropped = utils_acdc.normalise_image(slice_cropped) #Normalisation after cropping! If not, we get 0 padding instead of smallest value
                        img_list.append(slice_cropped)
                        data_file_name = 'pat_%s_diag_%s_frame_%s_slice_%s.npy' % (str(pat_id), infos['Group'], str(frame), str(z))
                        # determines to which data split to save
                        data_split = pat_train_test_val[pat_id]
                        data_file_path = os.path.join(preprocessing_folder, 'slices', data_split, data_file_name)         
                        # save image to slices folder
                        #im.save(os.path.join(preprocessing_folder, 'slices', data_split, data_file_name))  
                        slice_cropped = np.float32(slice_cropped)
                        np.save(data_file_path, slice_cropped)
    return img_list, mask_list #, info_list, diagnoses
                                  

    ################################################################### 
    # Requirements for data splits: (similarly to CamVid/Cityscapes)
    # diagnoses have to be represented in train, test, val
    # train >> test > val
    # split train into: D_S, D_T, D_V (where D_S << D_T<<D_V)
    # use val set for D_R
    # report final results on test set
    ###################################################################


def train_test_val(data_dir):
    '''
    # iterate over all patients in raw dataset
    # define the train, test, val splits, such that diagnoses are represented in each split
    # patients are in 5 groups of diagnoses, 20 patients each 
    # create dictionary with patients as key and 'train', 'test' or 'val' als value
    '''
    patients = np.asarray([int(folder[-3:]) for folder in os.listdir(data_dir)])
    pat_train_test_val = {}
    
    # array containing index to patient for each data split
    d_s = []
    d_t = []
    d_r = []
    d_v = []
    # for all patient ids
    for pat in patients:
        # train test contains 50 patients, 10x each diagnosis
        if pat<=10 or (pat>=21 and pat<=30) or (pat>=41 and pat<=50) or (pat>=61 and pat<=70) or (pat>=81 and pat<=90):
            pat_train_test_val[pat] = 'train'
            if pat==61:
                # add healthy patient to state representation split
                pat_num = 'pat_' + str(pat) + '_'
                d_s.append(pat_num)
            # 10 patients (each diagnosis twice) for 2,10, 22,30, 42,50, 62,70, 82,90
            elif '2' in str(pat)[-1] or '0' in str(pat)[-1]:
                pat_num = 'pat_' + str(pat) + '_'
                d_t.append(pat_num)
            else:
                pat_num = 'pat_' + str(pat) + '_'
                d_v.append(pat_num)
        # test set containing 40 patients, each diagnosis 8x
        elif (pat>=11 and pat<=18) or (pat>=31 and pat<=38) or (pat>=51 and pat<=58) or (pat>=71 and pat<=78) or (pat>=91 and pat<=98):
            pat_train_test_val[pat] = 'test'
        # val set cntains 10 patients, each diagnosis twice, also represents d_r
        else:
            pat_train_test_val[pat] = 'val'
            pat_num = 'pat_' + str(pat) + '_'
            d_r.append(pat_num)
    
    keys = ['d_s', 'd_t', 'd_r', 'd_v']
    values = d_s, d_t, d_r, d_v
    data_splits = np.array(dict(zip(keys, values)))
    file_dir = os.getcwd()
    data_file_name = 'data/acdc_al_splits.npy' 
    npy_data_file_path = os.path.join(file_dir, data_file_name)
    #print(data_file_path)
    
    # create new npy file to save data splits
    np.save(npy_data_file_path, data_splits)
    return pat_train_test_val, npy_data_file_path
    

def crop_or_pad_slice_to_size(slice, nx, ny):
    x, y = slice.shape
    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]
    
    return slice_cropped


def create_final_patient_splits(preprocessing_folder, npy_data_file_path):
    # array containing all images/masks to patient
    d_s = []
    d_t = []
    d_r = []
    d_v = []
    patient_splits = np.load(npy_data_file_path, allow_pickle=True).item()
    # list with all images (.npy files)
    img_paths = glob.glob(preprocessing_folder + "/slices/*/*.npy", recursive = True)

    # iterate over all image_paths
    for img in img_paths:
        #iterate over dictionary
        for key,values in patient_splits.items():
            for v in values:
                if v in img:
                    if key == 'd_s':
                        d_s.append(os.path.basename(img))
                    elif key == 'd_t':
                        d_t.append(os.path.basename(img))
                    elif key == 'd_r':
                        d_r.append(os.path.basename(img))
                    elif key == 'd_v':
                        d_v.append(os.path.basename(img))
                    else:
                        print('should not occur') 
    

    keys = ['d_s', 'd_t', 'd_r', 'd_v']
    values = d_s, d_t, d_r, d_v
    data_splits = np.array(dict(zip(keys, values)))
    file_dir = os.getcwd()
    data_file_name = 'data/acdc_pat_img_splits.npy' 
    npy_data_file_path = os.path.join(file_dir, data_file_name)
    np.save(npy_data_file_path, data_splits)


if __name__ == '__main__':
    input_folder = '/mnt/qb/baumgartner/cschmidt77_data/acdc_challenge/train'
    preprocessing_folder = '/mnt/qb/baumgartner/cschmidt77_data/acdc'
    pat_train_test_val, npy_data_file_path = train_test_val(input_folder)
    image_list, mask_list = load_patient_frames(input_folder, pat_train_test_val, size=(256,256) , target_resolution=(1.36719, 1.36719)) #size=(216,256)

    # method that stores all names of patient related files in splits
    patient_file_splits = create_final_patient_splits(preprocessing_folder, npy_data_file_path)

    #split_patients_based_on_diagnosis(diagnoses)