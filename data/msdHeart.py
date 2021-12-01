import glob
import os

import numpy as np
import torch
from torch.utils import data
import torchio as tio

num_classes = 2
ignore_label = 2
path = 'msd_heart'

def make_dataset(mode, root):
    img_path = os.path.join(root, "slices")
    mask_path = os.path.join(root, "gt")

    img_paths = [f for f in glob.glob(os.path.join(img_path, "*.npy"))]
    print('Length of image paths: ', len(img_paths))
    items = []

    for im_p in img_paths:
        item = (im_p, os.path.join(mask_path, im_p.split('/')[-1]), im_p.split('/')[-1])
        items.append(item)
    if mode == "train":
        items = items[:int(len(items)*0.6)]
    elif mode=='test':
        items = items[int(len(items)*0.6):-int(len(items)*0.1)]
    elif mode == "val":
        items = items[-int(len(items)*0.1):]
    else:
        print("mode not specified!")  
    return items


class MSD_Heart(data.Dataset):
    def __init__(self, quality, mode, data_path='', code_path='', joint_transform=None,
                 sliding_crop=None, transform=None, target_transform=None, subset=False):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.root = data_path + path
        print('in MSD_Heart class')
        self.imgs = make_dataset(mode, self.root)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        # d_t = np.load(
        #     os.path.join(code_path, 'data/acdc_pat_img_splits.npy'), 
        #     allow_pickle=True
        #     ).item()['d_t']
        
        # for train set of supervised, subset is true -> use d_t split, for validate use val data (see make_dataset)
        # if subset:
        #      self.imgs = [img for i, img in enumerate(self.imgs) if (img[-1] in d_t)]
        #      print("Usind d_t from ACDC")

        print('Using ' + str(len(self.imgs)) + ' images for mode:' + self.mode)

    #@carina adapted from Camvid
    def __getitem__(self, index):
        img_path, mask_path, im_name = self.imgs[index]
        img, mask = np.load(img_path), np.load(mask_path)
        img, mask = torch.from_numpy(img), torch.from_numpy(mask)
        img_unsqueezed = img.unsqueeze(0).unsqueeze(3) #adds two additional dims (required from torch io) 
        mask_unsqueezed = mask.unsqueeze(0).unsqueeze(3)
        subject = tio.Subject(
            img=tio.ScalarImage(tensor=img_unsqueezed), 
            mask=tio.LabelMap(tensor=mask_unsqueezed)
        )

        if self.joint_transform is not None:
            transformed = self.joint_transform(subject)
            img_transf = transformed.img.numpy()
            mask_transf = transformed.mask.numpy()
            img_t = torch.from_numpy(img_transf)
            mask_t = torch.from_numpy(mask_transf)
            img = torch.squeeze(img_t) #removes dimensions of size 1
            mask = torch.squeeze(mask_t) 
        if self.transform is not None: #torchvision standard transform
            if type(img) != torch.Tensor:
                img = self.transform(img) #self.transforms transforms ToTensor!! 
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        img = torch.stack((img, img, img), dim=0)
        return img, mask.long(), (img_path, mask_path, im_name)


    def __len__(self):
        return len(self.imgs)

    
