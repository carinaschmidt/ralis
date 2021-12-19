import os
import numpy as np

def create_splits(data_dir, dS_ratio, dT_ratio, dR_ratio):  
    '''divides patients in four data splits with given split ratio'''
    '''returns four splits with patient ids'''
    #create numpy array with patients
    patients = np.asarray([int(folder[-3:]) for folder in os.listdir(data_dir)])
    # random shuffle patients data set and create splits
    np.random.shuffle(patients)
    # define indices for np.split function
    dS_ind = int(dS_ratio*len(patients))
    dT_ind = dS_ind+int(dT_ratio*len(patients))
    dR_ind = dT_ind+int(dR_ratio*len(patients))
    #splits the random set in four splits, where dV is the rest
    dS, dT, dR, dV = np.split(patients, [dS_ind, dT_ind, dR_ind])
    return dS, dT, dR, dV

def create_splits_array(dS, dT, dR, dV):
    keys = ['d_S', 'd_T', 'd_R', 'd_V']
    values = [dS, dT, dR, dV]
    data_splits = np.array(dict(zip(keys, values)))
    return data_splits

def create_ref_data_splits_file(data_dir,
              preproc_dir, data_splits,
              mode,
              size,
              target_resolution,
              force_overwrite):
    '''Creates a file, where data splits will be defined '''
    # create folder for processed data with config naming
    size_str = '_'.join([str(s) for s in size])
    res_str = '_'.join([str(res) for res in target_resolution])

    data_file_name = 'data_%s_size_%s_res_%s_al_splits.npy' % (mode, size_str, res_str)
    data_file_path = os.path.join(preproc_dir, data_file_name)
    
    # create new npy file to save data
    npy_file = np.save(data_file_name, data_splits)  
    return npy_file

if __name__ == '__main__':
    data_dir = '/mnt/qb/baumgartner/cschmidt77_data/acdc_challenge/train'
    preproc_dir = '/home/baumgartner/cschmidt77/devel/ralis/acdc/preproc_data/'
    dS_ratio = 0.03
    dT_ratio = 0.09
    dR_ratio = 0.11
    dV_ratio = 0.77
    dS, dT, dR, dV = create_splits('/mnt/qb/baumgartner/cschmidt77_data/acdc_challenge/train', dS_ratio, dT_ratio, dR_ratio)
    data_splits = create_splits_array(dS, dT, dR, dV)
    npy_file = create_ref_data_splits_file(data_dir, preproc_dir, data_splits, '2D', (256, 256), (1.36719, 1.36719),force_overwrite=True)