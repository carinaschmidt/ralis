import glob
import os
from copy import deepcopy

import numpy as np
import torch
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


class ACDC_al(data.Dataset):
    def __init__(self, quality, mode, data_path='', code_path='', joint_transform=None,
                 sliding_crop=None, transform=None, target_transform=None, candidates_option=False,
                 region_size=(80, 90),
                 num_each_iter=1, only_last_labeled=True, split='train'):
        self.num_each_iter = num_each_iter
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.root = data_path + path
        print('in ACDC_al class')
        self.imgs = make_dataset(mode, self.root)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

        splits = np.load(
            os.path.join(code_path, 'data/acdc_pat_img_splits.npy'),#'data/acdc_al_splits.npy'), 
            allow_pickle=True
            ).item()

        # img[0] is full path to image
        self.state_subset = [img for i, img in enumerate(self.imgs) if (img[-1] in splits['d_s'])]    
        self.state_subset_regions = {}

        # len(splits['d_s']) is 36
        for i in range(len(splits['d_s'])):
            # region_size is here [80,90]
            x_r1 = np.arange(0, 256 - region_size[0] + 1, region_size[0]) #array([  0,  64, 128, 192]) 
            y_r1 = np.arange(0, 256 - region_size[1] + 1, region_size[1])
            self.state_subset_regions.update({i: np.array(np.meshgrid(x_r1, y_r1)).T.reshape(-1, 2)})

        if split == 'train':
            self.imgs = [img for i, img in enumerate(self.imgs) if (img[-1] in splits['d_t'])]
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
        self.num_imgs = len(self.imgs)
        splitters_x = np.arange(0, 256 - region_size[0] + 1, region_size[0]) #array([  0,  64, 128, 192])
        splitters_y = np.arange(0, 256 - region_size[1] + 1, region_size[1]) ##array([  0,  64, 128, 192])
        splitters_mesh = np.array(np.meshgrid(splitters_y, splitters_x)).T.reshape(-1, 2) #array([[  0,   0],
        prov_splitters = splitters_mesh.copy()
        prov_splitters_x = list(prov_splitters[:, 1])
        prov_splitters_y = list(prov_splitters[:, 0])
        self.unlabeled_regions_x = [deepcopy(prov_splitters_x) for _ in range(self.num_imgs)]
        self.unlabeled_regions_y = [deepcopy(prov_splitters_y) for _ in range(self.num_imgs)]
        #self.num_unlabeled_regions_total = (256 * 256) // (  #3040
        #        region_size[0] * region_size[1]) * self.num_imgs
        self.num_unlabeled_regions_total = (256 // region_size[1]) * (256 // region_size[0]) * self.num_imgs
        print('Number of unlabeled regions total: ', self.num_unlabeled_regions_total) #@carina
        self.region_size = region_size #[64,64]

    def get_subset_state(self, index):
        img_path, mask_path, im_name = self.state_subset[index]
        # @carina added 
        #import ipdb
        #ipdb.set_trace()
        #img, mask = np.stack((np_img,)*3, axis=-1), np.load(mask_path)
        img, mask = np.load(img_path), np.load(mask_path)
        img, mask = torch.from_numpy(img), torch.from_numpy(mask)

        if self.joint_transform is not None:
            #img =np_img #@carina to maybe avoid shape error
            img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            if type(img) != torch.Tensor:
                img = self.transform(img) #self.transforms transforms ToTensor!! but ToTensor returns image (0-255)
            #img = torch.from_numpy(img)           
            #img.shape: torch.Size([3, 256, 256])
            #norm_img = normalise_image(img)
            #img = np.stack((norm_img,)*3, axis=-1)  #@carina added to meet correct shape 
            #img = torch.from_numpy(img)
            #img = img.permute((2, 0, 1)).contiguous() #torch.Size([256, 256, 3])
            #print("type of img: ", type(img))
        if self.target_transform is not None:
            mask = self.target_transform(mask) #after transform: torch.Size([256, 256])
        #img = np.stack((img,)*3, axis=-1) 
        img = torch.stack((img, img, img), dim=0)
        #import ipdb
        #ipdb.set_trace()
        return img, mask, None, (img_path, mask_path, im_name), self.state_subset_regions[index]

    def __getitem__(self, index):
        #import ipdb
        #ipdb.set_trace()
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

        #img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        #@carina
        #np_img = np.load(img_path)
        img, mask = np.load(img_path), np.load(mask_path)
        img = torch.from_numpy(img)
    
        if not self.candidates:
            mask = self.maskout_unselected_regions(mask, selected[0], self.region_size)
            
        #@carina
        #mask = Image.fromarray(mask.astype(np.uint8))
        mask = torch.from_numpy(mask)
        if self.joint_transform is not None:
            if not self.candidates:
                #img, mask = torch.from_numpy(img), torch.from_numpy(mask)
                img, mask = self.joint_transform(img, mask, selected_region)
            else:
                #img, mask = torch.from_numpy(img), torch.from_numpy(mask)
                img, mask = self.joint_transform(img, mask)

        if self.transform is not None:
            #img = np.stack((img,)*3, axis=-1) #@carina added to meet correct shape 
            #img = self.transform(img)
            #norm_img = normalise_image(img)
            #img = np.stack((img,)*3, axis=-1)  #@carina added to meet correct shape 
            if type(img) != torch.Tensor:
                img = self.transform(img) #self.transforms transforms ToTensor!! 
            #img = torch.from_numpy(img)
            #img = img.permute((2, 0, 1)).contiguous() #torch.Size([256, 256, 3])
        if self.target_transform is not None:
            mask = self.target_transform(mask) #mask to tensor

        img = torch.stack((img, img, img), dim=0)
        #print("img shape ACDC_al getitem: ", img.shape)
        #print("mask shape ACDC_al getitem: ", mask.shape)
        #import ipdb
        #ipdb.set_trace()
        return img, mask, (img_path, mask_path, im_name), selected_region[0] if not self.candidates else \
            self.selected_images[index], 0

    def maskout_unselected_regions(self, mask, image, region_size=(128, 120)):
        masked = np.full(mask.shape, ignore_label)
        for region in self.selected_regions[image]:
            # Indexes reverted, because here width is the 2nd index.
            r_x = int(region[1])
            r_y = int(region[0])
            masked[r_x: r_x + region_size[1], r_y: r_y + region_size[0]] = mask[r_x: r_x + region_size[1],
                                                                           r_y: r_y + region_size[0]]
        return masked

    def get_specific_item(self, path):
        img_path, mask_path, im_name = self.imgs[path]
        cost_img = None
        #@carina
        #img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        #np_img = np.load(img_path)
        img, mask = np.load(img_path), np.load(mask_path)
        img, mask = torch.from_numpy(img), torch.from_numpy(mask)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.transform is not None:  #self.transform calls torchvision transforms transforms.py
            #import ipdb
            #ipdb.set_trace()
            #norm_img = normalise_image(img)
            #img = np.stack((img,)*3, axis=-1)  #@carina added to meet correct shape 
            if type(img) != torch.Tensor:
                img = self.transform(img) #self.transforms transforms ToTensor!! 
            #img = self.transform(img) #torch.Size([3, 256, 256])
            #img = torch.from_numpy(img)  #torch.Size([256, 256, 3])
            #img = img.permute((2, 0, 1)).contiguous() #torch.Size([256, 256, 3])
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        #img = np.stack((img,)*3, axis=-1)
        img = torch.stack((img, img, img), dim=0)
        # print("img shape ACDC_al getSpecitem: ", img.shape)
        # print("mask shape ACDC_al getSpecitem: ", mask.shape)
        return img, mask, cost_img, (img_path, mask_path, im_name)

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
        images_list = list(range(self.num_imgs))
        while unlabeled_regions <= num_regions_unlab:
            if len(images_list) == 0:
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
