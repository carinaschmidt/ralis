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

file_list = []
info_list = []
# save diagnoses of patients for splitting
diag_list = []
cardiac_phase_list = []

#containing frame images
img_list = []
#containing masks
mask_list = []


def load_images_masks(input_folder, size, target_resolution):
    '''
    Function to load frames and gt images in two folders, also store patient related information to list
    '''
    # iterate over all patients
    # save patient related infos in list respectively 
    # for each patient, load Info.cfg, patient###_4d.nii.gz, patient###_frame01.nii.gz, patient###_frame01_gt.nii.gz, patient###_frame12.nii.gz, patient###_frame12_gt.nii.gz
    #npy_file = np.save(input_folder)
    for folder in os.listdir(input_folder):
        if folder == 'imagesTr' or folder == 'labelsTr':
            file_path = input_folder + '/' + folder + '/'
            for file in os.listdir(file_path):
                #get patient id
                pat_id = file.split('_')[1].split('.')[0]

                img_path = os.path.join(file_path, file)
                # size (320,320, )
                nii_data = utils_acdc.load_nii(img_path)
                img = nii_data[0].copy()  

                # for frames we get (15625, 15625, 10) as pixel size
                pixel_size = (nii_data[2].structarr['pixdim'][1],
                            nii_data[2].structarr['pixdim'][2],
                            nii_data[2].structarr['pixdim'][3])   

                if folder == 'imagesTr':
                    img_list.append(img)
                else:
                    mask_list.append(img)
                

                ############# processing slice-by-slice 2D data##########
                #############saving images slice-by-slice################
                scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1]]
                nx, ny = size
                # iterate over # of third component of image shape (216,256,10) here 10 times
                # gives #z 2D images from 3D array
                for z in range(img.shape[2]):
                    # removes z axis along array, makes shape (216,256)         
                    if 'label' in folder:
                        #added as type uint8
                        slice_mask = np.squeeze(img[:, :, z])
                        
                        #Scale image by a certain factor.Performs interpolation to up-scale or down-scale N-dimensional images.
                        mask_rescaled = transform.rescale(slice_mask,
                                                             scale_vector,
                                                             order=0, #check if order has to be 0 or 1 
                                                             preserve_range=True,
                                                             multichannel=False,
                                                             mode='constant') 

                        mask_cropped = crop_or_pad_slice_to_size(mask_rescaled, nx, ny).astype(np.uint8)  
                        #mask_cropped = slice_mask
                        mask_list.append(mask_cropped)     

                        data_file_name = 'la_pat_%s_slice_%s.npy' % (str(pat_id), str(z))
                        # determines to which data split to save
                        #data_split = pat_train_test_val[pat_id]
                        # save mask to gt folder    
                        data_file_path = os.path.join(preprocessing_folder, 'gt', data_file_name) #data_split, data_file_name)
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
                       
                        data_file_name = 'la_pat_%s_slice_%s.npy' % (str(pat_id), str(z))
                        # determines to which data split to save
                        #data_split = pat_train_test_val[pat_id]
                        data_file_path = os.path.join(preprocessing_folder, 'slices', data_file_name) #data_split, data_file_name)         
                        # save image to slices folder
                        #im.save(os.path.join(preprocessing_folder, 'slices', data_split, data_file_name))  
                        slice_cropped = np.float32(slice_cropped)
                        np.save(data_file_path, slice_cropped)
    return img_list, mask_list
    

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


#def create_splits():
    # for each patient
    # randomly choose 3 patients
    # take slice 11, 21, 31, 41, 51, 61, 71, 81, 91 for D_t
    


if __name__ == '__main__':
    input_folder = '/mnt/qb/baumgartner/cschmidt77_data/msd_heart_raw'
    preprocessing_folder = '/mnt/qb/baumgartner/cschmidt77_data/msd_heart'
    img_list, mask_list = load_images_masks(input_folder, size=(320,320), target_resolution=(1.25, 1.25))
    # pat_train_test_val, npy_data_file_path = train_test_val(input_folder)
    # image_list, mask_list = load_images(input_folder, pat_train_test_val, size=(256,256) , target_resolution=(1.36719, 1.36719)) #size=(216,256)

    # # method that stores all names of patient related files in splits
    # patient_file_splits = create_final_patient_splits(preprocessing_folder, npy_data_file_path)

    #split_patients_based_on_diagnosis(diagnoses)