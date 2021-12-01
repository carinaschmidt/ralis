import glob
import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils import data
import torchio as tio

import utils.parser as parser

num_classes = 4 #4 # 5 for originial
ignore_label = 4 # 4
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
    print("length of items: ", len(items))
    return items


class BraTS18_2D_al(data.Dataset):
    def __init__(self, quality, mode, data_path='', code_path='', joint_transform=None,
                 sliding_crop=None, transform=None, target_transform=None, candidates_option=False,
                 region_size=(64, 64),
                 num_each_iter=1, only_last_labeled=True, split='train'):
        self.num_each_iter = num_each_iter
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.root = data_path + path
        print('in Brats18_2D_al class')
        self.imgs = make_dataset(mode, self.root)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform #train transforms like for supervised
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

        splits = np.load(
            os.path.join(code_path, 'data/brats18_pat_img_splits_DT3.npy'),#'data/acdc_al_splits.npy'), 
            allow_pickle=True
            ).item()

        # img[0] is full path to image
        self.state_subset = [img for i, img in enumerate(self.imgs) if (img[-1].split('_slice')[0] + '.npy' in splits['d_s'])]    
        self.state_subset_regions = {}

        args = parser.get_arguments()
        input_size = args.input_size

        original_input = [160,192,4]
        #original_input = [128,128,4]

        print("len splits['d_s'] =", len(splits['d_s']))
        splits_d_s_slices = len(splits['d_s']) * 160 ### splits['d_s'] is smaller than all slices, so multiply with number of slices
        for i in range(splits_d_s_slices): 
            x_r1 = np.arange(0, original_input[0] - region_size[0] + 1, region_size[0]) #array([  0,  64, 128, 192]) 
            y_r1 = np.arange(0, original_input[1] - region_size[1] + 1, region_size[1])
            self.state_subset_regions.update({i: np.array(np.meshgrid(x_r1, y_r1)).T.reshape(-1, 2)})
        
        print("len splits_d_s_slices: ", splits_d_s_slices)
        print("len of all images in BraTS: ", len(self.imgs))
        if split == 'train':
            self.imgs = [img for i, img in enumerate(self.imgs) if (img[-1].split('_slice')[0] + '.npy' in splits['d_t'])] # all slices from patients in split d_T
            #print("split train: ", len(self.imgs))
            # args = parser.get_arguments()
            # if "1patient" in args.exp_name:
            #     self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,160)]
            # elif "2patients" in args.exp_name:
            #     self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,320)]
            # elif "3patients" in args.exp_name:
            #     self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,480)]
            # elif "4patients" in args.exp_name:
            #     self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,640)]
            # elif "5patients" in args.exp_name:
            #     self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,800)]
            # elif "6patients" in args.exp_name:
            #     self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,960)]
            # elif "7patients" in args.exp_name:
            #     self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,1120)]
            # elif "8patients" in args.exp_name:
            #     self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,1280)]
            # elif "9patients" in args.exp_name:
            #     self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,1440)]
            print("Using d_t from BraTS18: ", len(self.imgs))
        elif split == 'test':
            print("Using d_v")
            self.imgs = [img for i, img in enumerate(self.imgs) if (img[-1].split('_slice')[0] + '.npy'  in splits['d_v'])]
            print("Using d_v from BraTS18: ", len(self.imgs))
        else:
            print("define splits!")
            
        print('Using ' + str(split) + ' splitting of ' + str(len(self.imgs)) + ' images for mode:' + self.mode)

        self.end_al = False
        self.balance_cl = []
        self.only_last_labeled = only_last_labeled
        self.candidates = candidates_option
        self.selected_images = []
        self.selected_regions = dict()
        self.list_regions = []
        self.num_imgs = len(self.imgs)
        print("num_imgs: ", self.num_imgs)
        splitters_x = np.arange(0, original_input[0] - region_size[0] + 1, region_size[0]) #array([  0,  64, 128, 192])
        splitters_y = np.arange(0, original_input[1] - region_size[1] + 1, region_size[1]) ##array([  0,  64, 128, 192])
        print("splitters x: ", splitters_x) # [  0  40  80 120]
        print("splitters y: ", splitters_y) #  [  0  48  96 144]
        splitters_mesh = np.array(np.meshgrid(splitters_y, splitters_x)).T.reshape(-1, 2) #array([[  0,   0],
        prov_splitters = splitters_mesh.copy()
        prov_splitters_x = list(prov_splitters[:, 1])
        prov_splitters_y = list(prov_splitters[:, 0])
        self.unlabeled_regions_x = [deepcopy(prov_splitters_x) for _ in range(self.num_imgs)]
        self.unlabeled_regions_y = [deepcopy(prov_splitters_y) for _ in range(self.num_imgs)]
        self.num_unlabeled_regions_total = (original_input[1] // region_size[1]) * (original_input[0] // region_size[0]) * self.num_imgs
        print("split: ", str(split))
        print('Number of unlabeled regions total for split: ', self.num_unlabeled_regions_total) #@carina
        self.region_size = region_size 
        print("region size: ", self.region_size) #40 48
        print("original input: ", original_input) # [160, 192, 4]

    def get_subset_state(self, index):
        img_path, mask_path, im_name = self.state_subset[index]
        img, mask = np.load(img_path), np.load(mask_path)
        mask = np.where(mask == 4, 3, mask) ##replace label 4 by label 3

        img, mask = torch.from_numpy(img).permute(2,0,1), torch.from_numpy(mask).unsqueeze(0) #add 1st channel dim
       
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
       
        mask = torch.squeeze(mask)
        img = torch.squeeze(img)
        return img, mask.long(), None, (img_path, mask_path, im_name), self.state_subset_regions[index]

    def __getitem__(self, index):
        #print("in getitem")
        # Train with all labeled images, selecting a random region per image, and doing the random crop around it
        if self.candidates or self.end_al:
            #print("self.candidates")
            img_path, mask_path, im_name = self.imgs[self.selected_images[index]]
            # Select random region in the image to make sure there is a region in the crop
            selected_region_ind = np.random.choice(len(self.selected_regions[self.selected_images[index]]))
            selected_region = self.selected_regions[self.selected_images[index]][selected_region_ind]
            selected = [self.selected_images[index]]
        else:
            #print("else in get_item")
            # Train with just the last regions selected, random crop around the selected region
            if self.only_last_labeled:
                selected = self.list_regions[len(self.list_regions) - self.num_each_iter:][index]
            # Train with all labeled regions so far, random crop around the selected region
            else:
                selected = self.list_regions[index]
            img_path, mask_path, im_name = self.imgs[selected[0]]
            selected_region = selected[1]

        
        img, mask = np.load(img_path), np.load(mask_path)
        mask = np.where(mask == 4, 3, mask)
        img, mask = torch.from_numpy(img).permute(2,0,1), torch.from_numpy(mask).unsqueeze(0) 
        #print("in get_item: ", img.shape)
        if not self.candidates:
            #print("maskout unselected regions: ")
            mask = self.maskout_unselected_regions(mask, selected[0], self.region_size)
        mask = torch.from_numpy(mask) #changed mask to tensor 

        if self.joint_transform is not None:
            if not self.candidates:
                img, mask = self.joint_transform(img, mask, selected_region)
            else:
                img, mask = self.joint_transform(img, mask)
        mask = torch.squeeze(mask)
        img = torch.squeeze(img)
        return img, mask.long(), (img_path, mask_path, im_name), selected_region[0] if not self.candidates else \
            self.selected_images[index], 0

    def maskout_unselected_regions(self, mask, image, region_size=(128, 128)):
        #masked = np.full(mask.shape, ignore_label)
        masked = np.full(mask.shape, 0) #fill with 0 #@changed by carina 
        for region in self.selected_regions[image]:
            # Indexes reverted, because here width is the 2nd index.
            r_x = int(region[1])
            r_y = int(region[0])
            masked[r_x: r_x + region_size[1], r_y: r_y + region_size[0]] = mask[r_x: r_x + region_size[1],
                                                                           r_y: r_y + region_size[0]]
        #print("masked after maskout: ", masked.min(), masked.max())
        return masked

    def get_specific_item(self, path):
        img_path, mask_path, im_name = self.imgs[path]
        cost_img = None
        img, mask = np.load(img_path), np.load(mask_path)
        mask = np.where(mask == 4, 3, mask)
        ###### process only first channel
        img, mask = torch.from_numpy(img).permute(2,0,1), torch.from_numpy(mask).unsqueeze(0) # don't take channels into account
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        mask = torch.squeeze(mask) #removes dim 1 
        img = torch.squeeze(img)
        return img, mask, cost_img, (img_path, mask_path, im_name)
       

    def __len__(self):  ### for debugging purpose -ValueError: num_samples should be a positive integer value, but got num_samples=0
        if self.candidates: # and self.supervised:
            return len(self.imgs)
        elif self.candidates or self.end_al:
            return len(self.selected_images)
        else:
            if self.only_last_labeled:
                return self.num_each_iter
            else:
                return len(self.list_regions)

    def get_random_unlabeled_region_image(self, index):
        counter_i = int(np.random.choice(range(len(self.unlabeled_regions_x[index])), 1, replace=False))
        counter_x = self.unlabeled_regions_x[index].pop(counter_i)
        counter_y = self.unlabeled_regions_y[index].pop(counter_i)
        return counter_x, counter_y

    def labeled_set(self):
        return self.selected_images

    def get_labeled_regions(self):
        return self.selected_regions

    def get_unlabeled_regions(self):
        return deepcopy(self.unlabeled_regions_x), deepcopy(self.unlabeled_regions_y)

    def set_unlabeled_regions(self, rx, ry):
        self.unlabeled_regions_x = rx
        self.unlabeled_regions_y = ry

    def get_num_unlabeled_regions(self, region_size=128):
        return self.num_unlabeled_regions_total

    def get_num_unlabeled_regions_image(self, index):
        return len(self.unlabeled_regions_x[index])

    def get_num_labeled_regions(self):
        labeled_regions = 0
        for key, value in self.selected_regions.items():
            labeled_regions += len(value)
        return labeled_regions

    def get_candidates(self, num_regions_unlab=1000):
        """Take all regions from an image into account. We will take as many images as we need to get to the number of
        regions we want to have as candidates.
        :param num_regions_unlab: (int) number of unlabeled regions to form candidates.
        :return: list of images that have, in total, the target number of unlabeled regions (num_regions_unlab)
        """
        unlabeled_regions = 0
        candidates = []
        print("Number of images in get_candidates: ", self.num_imgs) #@carina
        images_list = list(range(self.num_imgs))
        while unlabeled_regions <= num_regions_unlab:
            if len(images_list) == 0:
                #import ipdb
                #ipdb.set_trace()
                raise ValueError('There is no more unlabeled regions to fullfill the amount we want!')
            index = np.random.choice(len(images_list))
            candidate = images_list.pop(index)
            num_regions_left = self.get_num_unlabeled_regions_image(int(candidate))
            if num_regions_left > 0:
                unlabeled_regions += num_regions_left
                candidates.append(candidate)
        return candidates

    def check_class_region(self, img, region, region_size=(128, 120), eps=1E-7):
        img_path, mask_path, im_name = self.imgs[img]
        #mask = Image.open(mask_path)
        mask = np.load(mask_path)
        mask = np.where(mask == 4, 3, mask)
        #mask = np.array(mask)
        r_x = int(region[1])
        r_y = int(region[0])
        region_classes = mask[r_x: r_x + region_size[1], r_y: r_y + region_size[0]]
        unique, counts = np.unique(region_classes, return_counts=True)
        balance = []
        for cl in range(0, self.num_classes + 1):
            if cl in unique:
                balance.append(counts[unique == cl].item() / counts.sum())
            else:
                balance.append(eps)
        self.balance_cl.append(balance)

    def add_index(self, paths, region=None):
        if isinstance(paths, list):
            for path in paths:
                if path not in self.selected_images:
                    self.selected_images.append(int(path))
                if region is not None:
                    if int(path) in self.selected_regions.keys():
                        if region not in self.selected_regions[int(path)]:
                            self.selected_regions[int(path)].append(region)
                            self.add_index_(path, region)
                    else:
                        self.selected_regions.update({int(path): [region]})
                        self.add_index_(path, region)

        else:
            if paths not in self.selected_images:
                self.selected_images.append(int(paths))
            if region is not None:
                if int(paths) in self.selected_regions.keys():
                    if region not in self.selected_regions[int(paths)]:
                        self.selected_regions[int(paths)].append(region)
                        self.add_index_(paths, region)

                    else:
                        print('Region already added!')
                else:
                    self.selected_regions.update({int(paths): [region]})
                    self.add_index_(paths, region)

    def add_index_(self, path, region):
        self.list_regions.append((int(path), region))
        self.num_unlabeled_regions_total -= 1

        self.check_class_region(int(path), (region[0], region[1]), self.region_size)
        for i in range(len(self.unlabeled_regions_x[int(path)])):
            if self.unlabeled_regions_x[int(path)][i] == region[0] and \
                    self.unlabeled_regions_y[int(path)][i] == region[1]:
                self.unlabeled_regions_x[int(path)].pop(i)
                self.unlabeled_regions_y[int(path)].pop(i)
                break

    def del_index(self, paths):
        self.selected_images.remove(paths)

    def reset(self):
        self.selected_images = []
