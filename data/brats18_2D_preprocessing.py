# Authors:
# Carina Schmidt

import os
import numpy as np

def transform_volumes_to2D(input_folder, output_folder):
    train_test_val = ['train', 'test', 'validation']
    for ttv in train_test_val:
        path = os.path.join(input_folder, ttv) 
        for file in os.listdir(path): # get all 4D patient files
            file_path = os.path.join(path, file)
            vol = np.load(file_path) # load 4D patient file

            len_z = len(vol[0,0,:,0])

            for z in range(0, len_z): #iterate over z component
                vol_x_y = vol[:, :, z, :]
                data_file_name = '%s_slice_%s.npy' % (str(file.split('.')[0]), str(z)) #patiend and slice
                vol_file_path = os.path.join(output_folder, ttv, data_file_name)
                np.save(vol_file_path, vol_x_y) #save 2D slices

def transform_gts_to2D(input_folder, output_folder):
    train_test_val = ['train', 'test', 'validation']
    for ttv in train_test_val:
        path = os.path.join(input_folder, ttv) 
        for file in os.listdir(path): # get all 3D mask files
            file_path = os.path.join(path, file)
            vol = np.load(file_path) # load 3D mask volume

            len_z = len(vol[0,0,:])
            for z in range(0, len_z): #iterate over z component
                vol_x_y = vol[:, :, z]
                data_file_name = '%s_slice_%s.npy' % (str(file.split('.')[0]), str(z)) #patiend and slice
                vol_file_path = os.path.join(output_folder, ttv, data_file_name)
                np.save(vol_file_path, vol_x_y) #save 2D slices


if __name__ == '__main__':
    preprocessing_folder = "/mnt/qb/baumgartner/cschmidt77_data/BraTS2018/TrainingData/"

    input_folder_vol = '/mnt/qb/baumgartner/cschmidt77_data/BraTS2018/TrainingData/volumes/' #train test val
    output_folder_vol = '/mnt/qb/baumgartner/cschmidt77_data/BraTS2018/TrainingData/volumes_2D'  #train, test, val
    #transform_volumes_to2D(input_folder_vol, output_folder_vol)

    input_folder_gt = '/mnt/qb/baumgartner/cschmidt77_data/BraTS2018/TrainingData/gts/' #train test val
    output_folder_gt = '/mnt/qb/baumgartner/cschmidt77_data/BraTS2018/TrainingData/gts_2D'
    transform_gts_to2D(input_folder_gt, output_folder_gt)