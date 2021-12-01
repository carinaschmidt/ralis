import glob
import os

import numpy as np
import torch
from torch.utils import data
import torchio as tio

import utils.parser as parser
import data.augmentation as aug

import torchio as tio

num_classes = 4 # 5 for originial
ignore_label = 4
path = 'BraTS2018'

def make_dataset(mode, root):
    if mode == "train":
        img_path = os.path.join(root, "TrainingData", "volumes_2D", "train")
        mask_path = os.path.join(root, "TrainingData", "gts_2D", "train")
    elif mode == "val": #for validation set
        img_path = os.path.join(root, "TrainingData", "volumes_2D", "validation")
        mask_path = os.path.join(root, "TrainingData", "gts_2D", "validation")
    elif mode == "test":
        img_path = os.path.join(root, "TrainingData", "volumes_2D", "test")
        mask_path = os.path.join(root, "TrainingData", "gts_2D", "test")
    else:
        raise ValueError('Dataset split specified does not exist!')

    img_paths = [f for f in glob.glob(os.path.join(img_path, "*.npy"))]
    print('Length of image paths: ', len(img_paths))
    items = []
    for im_p in img_paths:
        item = (im_p, os.path.join(mask_path, im_p.split('/')[-1]), im_p.split('/')[-1])
        items.append(item)
    return items


class BraTS18_2D(data.Dataset):
    def __init__(self, quality, mode, data_path='', code_path='', joint_transform=None,
                 sliding_crop=None, transform=None, target_transform=None, subset=False):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.root = data_path + path
        print('in BraTS2D class')
        self.imgs = make_dataset(mode, self.root)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        d_t = np.load(
            os.path.join(code_path, 'data/brats18_pat_img_splits_DT3.npy'), 
            allow_pickle=True
            ).item()['d_t']

        # for train set of supervised, subset is true -> use d_t split, for validate use val data (see make_dataset)
        if subset:
            print("subset = True")
            self.imgs = [img for i, img in enumerate(self.imgs) if (img[-1].split('_slice')[0] + '.npy' in d_t)] # all slices from patients in split d_T

        print('Using ', str(len(self.imgs)) + ' images for mode:' + self.mode)


    def __getitem__(self, index):
        img_path, mask_path, im_name = self.imgs[index]
        img, mask = np.load(img_path), np.load(mask_path)
        #print("before one hot:", mask.shape)
        #print("mask max: ", mask.max())
        #mask = self._toEvaluationOneHot(mask)
        #print("mask max after eval: ", mask.max())
        #print("after to One Hot Encode mask.shape: ", mask.shape)
        #img = img[...,:3] #take only first three channels into account, since network expects 3 channels instead of 4
        mask = np.where(mask == 4, 3, mask)

        img, mask = torch.from_numpy(img).permute(2,0,1), torch.from_numpy(mask).unsqueeze(0) #add 1st channel dim

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)


        # if self.transform is not None: #torchvision standard transform
        #     if type(img) != torch.Tensor:
        #         img = self.transform(img) #self.transforms transforms ToTensor!! 
        
        if type(img) != torch.Tensor:
            img = torch.from_numpy(img.copy())
            mask = torch.from_numpy(mask.copy())
        img = torch.squeeze(img) #removes dimensions of size 1  
        mask = torch.squeeze(mask) 
        #print("img.shape before return: ", img.shape)
        #print("mask.shape before return: ", mask.shape)
        return img, mask.long(), (img_path, mask_path, im_name)


    # def _toEvaluationOneHot(self, labels):

    #     ### all labels not equal to 0 is whole tumour (WT)
    #     ### all labels not equal to 0 and not 2 is TC
    #     shape = labels.shape

    #     out = np.zeros([shape[0], shape[1], 3], dtype=np.float32)
    #     #print("out.shape: ", out.shape)
    #     out[:, :, 0] = (labels != 0) #RoI WT (TC and ED)
    #     out[:, :, 1] = (labels != 0) * (labels != 2) #TC (everything besides 0 and 2)
    #     out[:, :, 2] = (labels == 4) # ET
    #     return out
    
    # def _toEvaluationTensor(self, labels):
    #     if labels[labels==0]:
    #         print("zero labels")
    #     else:
    #         if labels[labels==0]



    def __len__(self):
        return len(self.imgs)

    
