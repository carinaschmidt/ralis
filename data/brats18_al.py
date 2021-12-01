import glob
import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils import data
import utils.parser as parser
import torchio as tio
import h5py
import numpy as np
import random
import data.augmentation as aug
import itertools



num_classes = 3 # 5 for originial
ignore_label = 3
path = 'brats18'


def make_dataset(mode, root):
    if mode == "train":
        img_path = os.path.join(root, "TrainingData", "volumes", "train")
        mask_path = os.path.join(root, "TrainingData", "gts", "train")
    elif mode == "val": #for validation set
        img_path = os.path.join(root, "TrainingData", "volumes", "validation")
        mask_path = os.path.join(root, "TrainingData", "gts", "validation")
    elif mode == "test":
        img_path = os.path.join(root, "TrainingData", "volumes", "test")
        mask_path = os.path.join(root, "TrainingData", "gts", "test")
    else:
        raise ValueError('Dataset split specified does not exist!')

    img_paths = [f for f in glob.glob(os.path.join(img_path, "*.npy"))]
    print('Length of image paths: ', len(img_paths))
    items = []
    for im_p in img_paths:
        item = (im_p, os.path.join(mask_path, im_p.split('/')[-1]), im_p.split('/')[-1])
        items.append(item)
    return items


class BraTS18_al(data.Dataset):
    def __init__(self, quality, mode, data_path='', code_path='', joint_transform=None, joint_transform_region=None, joint_transform_acdc_al = None,
                 sliding_crop=None, transform=None, target_transform=None, candidates_option=False,
                 region_size=(80, 90),
                 num_each_iter=1, only_last_labeled=True, split='train'):
        self.mode = mode
        self.file = None
        # got values from noNewNet.py class
        self.trainOriginalClasses = False   #to get original classes or only binary classes
        self.randomCrop = [128, 128, 128]
        self.hasMasks = True
        self.returnOffsets = False
        #augmentation settings
        self.nnAugmentation = True
        self.softAugmentation = False
        self.doRotate = True
        self.rotDegrees =  20
        self.doScale = True
        self.scaleFactor = 1.1
        self.doFlip = True
        self.doElasticAug = True
        self.sigma = 10
        self.doIntensityShift = True
        self.maxIntensityShift = 0.1

        ##### from ACDC
        self.num_each_iter = num_each_iter
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.root = data_path + path
        print('in BraTS18_al class')
        self.imgs = make_dataset(mode, self.root)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform #train transforms like for supervised
        self.joint_transform_region = joint_transform_region #region crop
        self.joint_transform_acdc_al = joint_transform_acdc_al  #train transforms - crop

        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

        splits = np.load(
            os.path.join(code_path, 'data/brats18_pat_img_splits.npy'),#'data/acdc_al_splits.npy'), 
            allow_pickle=True
            ).item()

        # img[0] is full path to image
        self.state_subset = [img for i, img in enumerate(self.imgs) if (img[-1] in splits['d_s'])]    
        self.state_subset_regions = {}

        # len(splits['d_s']) is 36
        for i in range(len(splits['d_s'])):
            # region_size is here [80,90]
            x_r1 = np.arange(0, self.randomCrop[0] - region_size[0] + 1, region_size[0]) #array([  0,  64, 128, 192]) 
            y_r1 = np.arange(0, self.randomCrop[0] - region_size[1] + 1, region_size[1])
            self.state_subset_regions.update({i: np.array(np.meshgrid(x_r1, y_r1)).T.reshape(-1, 2)})

        if split == 'train':
            self.imgs = [img for i, img in enumerate(self.imgs) if (img[-1] in splits['d_t'])]
            args = parser.get_arguments()
            if "1patient" in args.exp_name:
                self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,19)]
            elif "2patients" in args.exp_name:
                self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,38)]
            elif "3patients" in args.exp_name:
                self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,57)]
            elif "4patients" in args.exp_name:
                self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,76)]
            elif "5patients" in args.exp_name:
                self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,95)]
            elif "6patients" in args.exp_name:
                self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,114)]
            elif "7patients" in args.exp_name:
                self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,133)]
            elif "8patients" in args.exp_name:
                self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,152)]
            elif "9patients" in args.exp_name:
                self.imgs = [img for i, img in enumerate(self.imgs) if i in range(0,171)]
            print("Using d_t")
        elif split == 'test':
            print("Using d_v")
            self.imgs = [img for i, img in enumerate(self.imgs) if (img[-1] in splits['d_v'])]

        print('Using ' + str(split) + ' splitting of ' + str(len(self.imgs)) + ' images for mode:' + self.mode)

        self.end_al = False
        self.balance_cl = []
        self.only_last_labeled = only_last_labeled
        self.candidates = candidates_option
        self.selected_images = []
        self.selected_regions = dict()
        self.list_regions = []

        # 3D specific
        self.num_volumes = len(self.imgs) #here loaded files are volumes
        self.num_slices_per_volume_per_dim = self.randomCrop[0]
        self.num_slices_per_volume = self.randomCrop[0] + self.randomCrop[1] + self.randomCrop[1] # here 3dimensional,
        self.num_slices_total = self.num_slices_per_volume * self.num_volumes

        splitters_x = np.arange(0, self.randomCrop[0] - region_size[0] + 1, region_size[0]) #array([  0,  64, 128, 192])
        splitters_y = np.arange(0, self.randomCrop[0] - region_size[1] + 1, region_size[1]) ##array([  0,  64, 128, 192])
        splitters_mesh = np.array(np.meshgrid(splitters_y, splitters_x)).T.reshape(-1, 2) #array([[  0,   0],
        prov_splitters = splitters_mesh.copy()
        prov_splitters_x = list(prov_splitters[:, 1])
        prov_splitters_y = list(prov_splitters[:, 0])

        # unlabeled regions for each direction (dim x, z, z)
        self.unlabeled_regions_x_dimx = [deepcopy(prov_splitters_x) for _ in range(self.self.num_slices_per_volume_per_dim)]
        self.unlabeled_regions_y_dimx = [deepcopy(prov_splitters_y) for _ in range(self.self.num_slices_per_volume_per_dim)]

        self.unlabeled_regions_x_dimy = [deepcopy(prov_splitters_x) for _ in range(self.self.num_slices_per_volume_per_dim)]
        self.unlabeled_regions_y_dimy = [deepcopy(prov_splitters_y) for _ in range(self.self.num_slices_per_volume_per_dim)]

        self.unlabeled_regions_x_dimz = [deepcopy(prov_splitters_x) for _ in range(self.self.num_slices_per_volume_per_dim)]
        self.unlabeled_regions_y_dimz = [deepcopy(prov_splitters_y) for _ in range(self.self.num_slices_per_volume_per_dim)]

        # 3D: 
        # e.g. for random crop 128 and region size 32: 16 regions per slice
        # e.g. randomCrop[x,y,z] number of slices in each dimension, here 128
        # here num_slices_total from all volumes
        self.num_unlabeled_regions_total = (self.randomCrop[0] // region_size[1]) * (self.randomCrop[0] // region_size[0]) * self.num_slices_total
        print('Number of unlabeled regions total: ', self.num_unlabeled_regions_total) #@carina

        self.region_size = region_size #[64,64]

    def get_subset_state(self, index):
        img_path, mask_path, im_name = self.state_subset[index]
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
        
        # if self.transform is not None:
        #     if type(img) != torch.Tensor:
        #         img = self.transform(img) #self.transforms transforms ToTensor!! but ToTensor returns image (0-255)
        # if self.target_transform is not None:
        #     mask = self.target_transform(mask) #after transform: torch.Size([256, 256])
        #img = np.stack((img,)*3, axis=-1) 
        img = torch.stack((img, img, img), dim=0)
        return img, mask.long(), None, (img_path, mask_path, im_name), self.state_subset_regions[index]

    def __getitem__(self, index):
        # Train with all labeled images, selecting a random region per image, and doing the random crop around it
        if self.candidates or self.end_al:
            img_path, mask_path, im_name = self.imgs[self.selected_images[index]]
            # Select random region in the image to make sure there is a region in the crop
            selected_region_ind = np.random.choice(len(self.selected_regions[self.selected_images[index]]))
            selected_region = self.selected_regions[self.selected_images[index]][selected_region_ind]
            selected = [self.selected_images[index]]
        else:
            # Train with just the last regions selected, random crop around the selected region
            if self.only_last_labeled:
                selected = self.list_regions[len(self.list_regions) - self.num_each_iter:][index]
            # Train with all labeled regions so far, random crop around the selected region
            else:
                selected = self.list_regions[index]
            img_path, mask_path, im_name = self.imgs[selected[0]]
            selected_region = selected[1]

        img, mask = np.load(img_path), np.load(mask_path)
        img, mask = torch.from_numpy(img), torch.from_numpy(mask)
        #print("tensor mask.max(): ", mask.max())
        if not self.candidates:
            #print("maskout unselected regions: ")
            mask = self.maskout_unselected_regions(mask, selected[0], self.region_size)
        mask = torch.from_numpy(mask) #changed mask to tensor    
      
        if self.joint_transform is not None:
            #print("selected region", selected_region)
            if not self.candidates:
                #img, mask = self.joint_transform(img, mask, selected_region)
                # first get selected region, then perform other transforms
                if self.joint_transform_region is not None:
                    # get selected region
                    img, mask = self.joint_transform_region(img, mask, selected_region)
                    img_unsqueezed = img.unsqueeze(0).unsqueeze(3) #adds two additional dims (required from torch io) 
                    mask_unsqueezed = mask.unsqueeze(0).unsqueeze(3)
                    subject = tio.Subject(
                        img=tio.ScalarImage(tensor=img_unsqueezed), 
                        mask=tio.LabelMap(tensor=mask_unsqueezed)
                    )
                    # performs other transforms
                    transformed = self.joint_transform_acdc_al(subject)
                    img_transf = transformed.img.numpy()
                    mask_transf = transformed.mask.numpy()
                    img_t = torch.from_numpy(img_transf)
                    mask_t = torch.from_numpy(mask_transf)
                    img = torch.squeeze(img_t) #removes dimensions of size 1
                    mask = torch.squeeze(mask_t) 

            else:
                img_unsqueezed = img.unsqueeze(0).unsqueeze(3) #adds two additional dims (required from torch io) 
                mask_unsqueezed = mask.unsqueeze(0).unsqueeze(3)
                subject = tio.Subject(
                    img=tio.ScalarImage(tensor=img_unsqueezed), 
                    mask=tio.LabelMap(tensor=mask_unsqueezed)
                    )
                transformed = self.joint_transform(subject)
                img_transf = transformed.img.numpy()
                mask_transf = transformed.mask.numpy()
                img_t = torch.from_numpy(img_transf)
                mask_t = torch.from_numpy(mask_transf)
                img = torch.squeeze(img_t) #removes dimensions of size 1
                mask = torch.squeeze(mask_t) 

        img = torch.stack((img, img, img), dim=0)
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

        img = torch.stack((img, img, img), dim=0)
        return img, mask.long(), cost_img, (img_path, mask_path, im_name)

    def __len__(self):
        if self.candidates and self.supervised:
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

    def get_num_unlabeled_regions_volume(self, index):
        return len(self.unlabeled_regions_x[index])

    def get_num_labeled_regions(self):
        labeled_regions = 0
        for key, value in self.selected_regions.items():
            labeled_regions += len(value)
        return labeled_regions

    
    def get_volume_candidates(self, num_regions_unlab=1000):
        """
        Take all slices from a volume into account. 
        Take all regions from a sliceinto account. We will take as many images as we need to get to the number of
        regions we want to have as candidates.
        :param num_regions_unlab: (int) number of unlabeled regions to form candidates.
        :return: list of images that have, in total, the target number of unlabeled regions (num_regions_unlab)
        """
        unlabeled_regions = 0
        candidates = []
        print("Number of volumes in get_candidates: ", self.num_volumes) #@carina
        volumes_list = list(range(self.num_volumes))
        print("Total number of slices in all volumes: ", self.num_slices_total) #@carina
        slices_list = list(range(self.num_volumes))
        # print("Number of slices per volume: ", self.num_slices_per_volume) #@carina
        # slices_per_volume_list = list(range(self.num_slices_per_volume))

        print("Number of slices per volume in each dim: ", self.num_slices_per_volume_per_dim) #@carina
        slices_per_volume_per_dim_list = list(range(self.num_slices_per_volume_per_dim)) #here 128

        slice_directions = [0, 1, 2] # x,y,z direction

        # compute all possible permutations of volume, slice, direction
        vol_slice_dir = [volumes_list, slices_per_volume_per_dim_list, slice_directions]
        candidates_tuple_list = list(itertools.product(*vol_slice_dir))

        while unlabeled_regions <= num_regions_unlab:
            if len(candidates_tuple_list) == 0:
                raise ValueError('There is no more unlabeled regions to fullfill the amount we want!')
            tuple_idx = np.random.choice(len(candidates_tuple_list)) #get one tuple 
            candidate_tuple = candidates_tuple_list.pop(tuple_idx)
            num_regions_left = self.get_num_unlabeled_regions_volume(int(candidate_tuple))
            if num_regions_left > 0:
                unlabeled_regions += num_regions_left
                candidates.append(candidate_tuple)
        return candidates #list of tuples of (volume_idx, slice_idx, direction)



    def get_candidates(self, num_regions_unlab=1000):    
        """
        Take all slices from a volume into account. 
        Take all regions from an image into account. We will take as many images as we need to get to the number of
        regions we want to have as candidates.
        :param num_regions_unlab: (int) number of unlabeled regions to form candidates.
        :return: list of images that have, in total, the target number of unlabeled regions (num_regions_unlab)
        """
        unlabeled_regions = 0
        candidates = []
        print("Number of volumes in get_candidates: ", self.num_volumes) #@carina
        volumes_list = list(range(self.num_volumes))
        while unlabeled_regions <= num_regions_unlab:
            if len(volumes_list) == 0:
                #import ipdb
                #ipdb.set_trace()
                raise ValueError('There is no more unlabeled regions to fullfill the amount we want!')
            index = np.random.choice(len(volumes_list))
            candidate = volumes_list.pop(index)
            num_regions_left = self.get_num_unlabeled_regions_image(int(candidate))
            if num_regions_left > 0:
                unlabeled_regions += num_regions_left
                candidates.append(candidate)
        return candidates

    def check_class_region(self, img, region, region_size=(128, 120), eps=1E-7):
        img_path, mask_path, im_name = self.imgs[img]
        #mask = Image.open(mask_path)
        mask = np.load(mask_path)
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

    def add_index(self, paths, region=None): #gets list of candidate tuples(vol, slice, dir): here paths
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
