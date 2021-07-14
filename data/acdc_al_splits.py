import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data

num_classes = 4
ignore_label = 4
path = 'acdc'
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153,
           153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

# def colorize_mask(mask):
#     # mask: numpy array of the mask
#     new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
#     new_mask.putpalette(palette)
#     return new_mask


def make_dataset(quality, mode, root):
    if mode == "train":
        # img_path='/mnt/qb/baumgartner/cschmidt77_data/acdc/slices/train'
        img_path = os.path.join(root, "slices", "train")
        mask_path = os.path.join(root, "gt", "train")
    elif mode == "val":
        img_path = os.path.join(root, "slices", "val")
        mask_path = os.path.join(root, "gt", "val")
    elif mode == "test":
        img_path = os.path.join(root, "slices", "test")
        mask_path = os.path.join(root, "gt", "test")
    else:
        raise ValueError('Dataset split specified does not exist!')
    # list with img paths '/mnt/qb/baumgartner/cschmidt77_data/acdc/slices/train/pat_9_diag_2_frame_13_slice_9_size_(256, 256)_res_(256, 256).png'
    img_paths = [f for f in glob.glob(os.path.join(img_path, "*.npy"))]
    items = []
    # im_p '/mnt/qb/baumgartner/cschmidt77_data/acdc/slices/train/pat_10_diag_2_frame_01_slice_0_size_(256, 256)_res_(256, 256).png'
    for im_p in img_paths:
        # item: ('/mnt/qb/baumgartner/cschmidt77_data/acdc/slices/train/pat_10_diag_2_frame_01_slice_0_size_(256, 256)_res_(256, 256).png', 
        # '/mnt/qb/baumgartner/cschmidt77_data/acdc/gt/train/pat_10_diag_2_frame_01_slice_0_size_(256, 256)_res_(256, 256).png', 
        # 'pat_10_diag_2_frame_01_slice_0_size_(256, 256)_res_(256, 256).png')
        item = (im_p, os.path.join(mask_path, im_p.split('/')[-1]), im_p.split('/')[-1])
        items.append(item)
    return items


class ACDC_al_splits(data.Dataset):
    def __init__(self, quality, mode, data_path='', code_path='', joint_transform=None,
                 sliding_crop=None, transform=None, target_transform=None, supervised=False, subset=False):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.root = data_path + path
        print('in ACDC_al_splits class')
        self.imgs = make_dataset(quality, mode, self.root)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label,
                              1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label,
                              5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label,
                              11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label,
                              16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9,
                              23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label,
                              31: 16, 32: 17, 33: 18}

        # item() gives dictionary inside array with the four splits as keys
        splits = np.load(
            os.path.join(code_path, 'data/acdc_pat_img_splits.npy'), allow_pickle=True
            ).item()
        # import ipdb
        # ipdb.set_trace()
        # d_t
        if subset:
            self.imgs = [img for i, img in enumerate(self.imgs) if (img[0] in splits['d_t'])]
        else:
            # d_t + d_v
            if supervised:
                self.imgs = [img for i, img in enumerate(self.imgs) if
                             (img[0] in splits['d_t'] or img[0] in splits['d_v'])]

            # d_r
            else: #in imgs: '/mnt/qb/baumgartner/cschmidt77_data/acdc/slices/val/pat_99_diag_4_frame_09_slice_6.npy', '/mnt/qb/baumgartner/cschmidt77_data/acdc/slices/val/pat_99_diag_4_frame_09_slice_7.npy', '/mnt/qb/baumgartner/cschmidt77_data/acdc/slices/val/pat_99_diag_4_frame_09_slice_8.npy', '/mnt/qb/baumgartner/cschmidt77_data/acdc/slices/val/pat_99_diag_4_frame_09_slice_9.npy'
                self.imgs = [img for i, img in enumerate(self.imgs) if (img in splits['d_r'])] #in d_r: '/mnt/qb/baumgartner/cschmidt77_data/acdc/slices/val/pat_99_diag_4_frame_09_slice_8.npy'
        print('Using splitting of ' + str(len(self.imgs)) + ' images.')

    def __getitem__(self, index):
        img_path, mask_path, im_name = self.imgs[index]
        img, mask = np.load(img_path), np.load(mask_path) 
        #img_stacked = np.stack((img,)*3, axis=-1)

        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        # import ipdb
        # ipdb.set_trace()
        #@carina 
        #img, mask = Image.fromarray(img_stacked.astype(np.uint8)), Image.fromarray(mask.astype(np.uint8))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info), im_name
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            mask = np.stack((img,)*3, axis=-1)
            return img, mask, (img_path, mask_path, im_name)

    def __len__(self):
        return len(self.imgs)
