# Author:
# Carina Schmidt (carina.schmidt@mail.de)

import os, os.path
import numpy as np

def create_pat_splits(npy_vol_path):
    # npy_vol_path: '/mnt/qb/baumgartner/cschmidt77_data/BraTS2018/TrainingData/volumes/
    # path to train patients
    train_path = os.path.join(npy_vol_path, 'train')
    train = os.listdir(train_path) # 143
    d_s = np.random.choice(train, 1, replace=False) #randomly schoose unique 1 pat to build D_S
    train = np.setdiff1d(train, d_s) #removes d_s from train -> 142 left
    d_t = np.random.choice(train, 3, replace=False) # choose 3 pats to build D_T
    d_v = np.setdiff1d(train, d_t) #remaining 139 patients to build d_v
    
    val_path = os.path.join(npy_vol_path, 'validation') #28 patients for D_R
    d_r = os.listdir(val_path) #list of val files save as d_r
    # 114 in test

    keys = ['d_s', 'd_t', 'd_r', 'd_v']
    values = list(d_s), list(d_t), list(d_r), list(d_v)
    data_splits = np.array(dict(zip(keys, values))) #keys are split names, values are names 'pat_id.npy' file names
    file_dir = os.getcwd()
    data_file_name = 'data/brats18_pat_img_splits_DT3.npy' 
    npy_data_file_path = os.path.join(file_dir, data_file_name)
    np.save(npy_data_file_path, data_splits)
    # save dict as array({'d_s':['pat_119.npy',...], 'd_t':[...]})

if __name__ == '__main__':
    create_pat_splits(npy_vol_path = '/mnt/qb/baumgartner/cschmidt77_data/BraTS2018/TrainingData/volumes/')

