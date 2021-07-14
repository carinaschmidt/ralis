import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data

num_classes = 4
ignore_label = 4
path = 'acdc'
palette = [128, 128, 128, 128, 0, 0, 192, 192, 128, 128, 64, 128, 0, 0, 192, 128, 128, 0, 192, 128, 128, 64, 64, 128,
           64, 0, 128, 64, 64, 0, 0, 128, 192, 0, 0, 0]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


# def colorize_mask(mask):
#     # mask: numpy array of the mask
#     new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
#     new_mask.putpalette(palette)
#     return new_mask


def make_dataset(mode, root):
    if mode == "train":
        img_path = os.path.join(root, "slices", "train")
        mask_path = os.path.join(root, "gt", "train")
    elif mode == "val": #for validation set
        img_path = os.path.join(root, "slices", "val")
        mask_path = os.path.join(root, "gt", "val")
        # img path: /mnt/qb/baumgartner/cschmidt77_data/acdc/slices/val
    elif mode == "test":
        img_path = os.path.join(root, "slices", "test")
        mask_path = os.path.join(root, "gt", "test")
    else:
        raise ValueError('Dataset split specified does not exist!')

    img_paths = [f for f in glob.glob(os.path.join(img_path, "*.npy"))]
    # length of image paths 190
    print('Length of image paths: ', len(img_paths))
    items = []
    #im_p: '/mnt/qb/baumgartner/cschmidt77_data/acdc/slices/val/pat_100_diag_4_frame_01_slice_0_size_(256, 256)_res_(256, 256).png'
    for im_p in img_paths:
        item = (im_p, os.path.join(mask_path, im_p.split('/')[-1]), im_p.split('/')[-1])
        items.append(item)
    # import ipdb
    # ipdb.set_trace()
    #items [ ('/mnt/qb/baumgartner/cschmidt77_data/acdc/slices/val/pat_99_diag_4_frame_09_slice_9.npy', 
    # '/mnt/qb/baumgartner/cschmidt77_data/acdc/gt/val/pat_99_diag_4_frame_09_slice_9.npy', 
    # 'pat_99_diag_4_frame_09_slice_9.npy'), ...]
    return items


class ACDC(data.Dataset):
    def __init__(self, quality, mode, data_path='', code_path='', joint_transform=None,
                 sliding_crop=None, transform=None, target_transform=None, subset=False):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.root = data_path + path
        print('in ACDC class')
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
            os.path.join(code_path, 'data/acdc_pat_img_splits.npy'), 
            allow_pickle=True
            ).item()['d_t']
        #d_t=['/mnt/qb/baumgartner/cschmidt77_data/acdc/slices/train/pat_90_diag_4_frame_11_slice_6.npy',..]
        # if subset:
        #     import ipdb 
        #     ipdb.set_trace()
        #     from matplotlib import pyplot as plt
        #     for i, img in enumerate(self.imgs): 
        #         if img[-1] in d_t:
        #             plt.imshow(img[-1], cmap='gray')
        #             plt.show()
        
        # for train set of supervised, subset is true -> use d_t split, for validate use val data (see make_dataset)
        if subset:
            self.imgs = [img for i, img in enumerate(self.imgs) if (img[-1] in d_t)]
            print("Usind d_t from ACDC")

        print('Using ' + str(len(self.imgs)) + ' images for mode:' + self.mode)

    #@carina adapted from Camvid
    def __getitem__(self, index):
        #import ipdb
        #ipdb.set_trace()
        img_path, mask_path, im_name = self.imgs[index]
        # @carina added 
        img, mask = np.load(img_path), np.load(mask_path)
        img, mask = torch.from_numpy(img), torch.from_numpy(mask)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.transform is not None: #torchvision standard transform
            if type(img) != torch.Tensor:
                img = self.transform(img) #self.transforms transforms ToTensor!! 
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        img = torch.stack((img, img, img), dim=0)
        return img, mask, (img_path, mask_path, im_name)


    def __len__(self):
        return len(self.imgs)

    
