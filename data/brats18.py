# Author:
# CarinaSchmidt

import glob
import os
import torch
import torch.utils.data
import h5py
import numpy as np
import random
import data.augmentation as aug

num_classes = 3
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

class Brats18(torch.utils.data.Dataset):
    #mode must be train, test or val
    def __init__(self, mode, data_path='', code_path='', joint_transform=None,
                 sliding_crop=None, transform=None, target_transform=None, subset=False):
        super(Brats18, self).__init__()
        #self.filePath = '/mnt/qb/baumgartner/cschmidt77_data/BraTS2018/TrainingData/HGG/data_3D_size_160_192_160_res_1.0_1.0_1.0.hdf5' # @carina only HGG data
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

        #@carina added from ACDC ##################
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        path = 'BraTS2018'
        self.root = data_path + path
        print('in BraTS18 class')
        self.imgs = make_dataset(mode, self.root)
        ####for memory issues, use only the first 2 images
        self.imgs = self.imgs[15:25]

        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        d_t = np.load(
            os.path.join(code_path, 'data/brats18_pat_img_splits.npy'), 
            allow_pickle=True
            ).item()['d_t']
        
        if subset:
            self.imgs = [img for i, img in enumerate(self.imgs) if (img[-1] in d_t)]

    def __getitem__(self, index):
        ############## @carina #####################
        img_path, mask_path, im_name = self.imgs[index]
        img, mask = np.load(img_path), np.load(mask_path)

        #Prepare data depending on soft/hard augmentation scheme
        if not self.nnAugmentation:
            if not self.trainOriginalClasses and (self.mode != "train" or self.softAugmentation):
                if self.hasMask: self._toEvaluationOneHot(mask)
                defaultLabelValues = np.zeros(3, dtype=np.float32)
            else:
                if self.hasMasks: mask = self._toOrignalCategoryOneHot(mask)
                defaultLabelValues = np.asarray([1, 0, 0, 0, 0], dtype=np.float32)
        elif self.hasMasks:
            if mask.ndim < 4:
                mask = np.expand_dims(mask, 3) # add a fourth dimension
            defaultLabelValues = np.asarray([0], dtype=np.float32)

        #augment data
        if self.mode == "train":
            #print("do augmentations")
            mask = np.stack((mask, mask, mask, mask), axis = 3) #stacks mask along third dimension, helper for augmentations
            mask = np.squeeze(mask) #removes dim of size 1 
            #print("image.shape: ", img.shape) #(160, 192, 160, 4)

            # image needs and mask need to be 4-dim, maybe add fourth dim for mask
            img, mask = aug.augment3DImage(img, #image,
                                            mask, #labels,
                                            defaultLabelValues,
                                            self.nnAugmentation,
                                            self.doRotate,
                                            self.rotDegrees,
                                            self.doScale,
                                            self.scaleFactor,
                                            self.doFlip,
                                            self.doElasticAug,
                                            self.sigma,
                                            self.doIntensityShift,
                                            self.maxIntensityShift)
            
            mask = mask[:, :, :, 0]  #removes the fourth dimension, but adds a new one
            mask = np.resize(mask, (160,192,160,1))

        if self.nnAugmentation:
            if self.hasMasks:
                mask = self._toEvaluationOneHot(np.squeeze(mask, 3))
        else:
            if self.mode == "train" and not self.softAugmentation and not self.trainOriginalClasses and self.hasMasks:
                mask = self._toOrdinal(mask)
                mask = self._toEvaluationOneHot(mask)
                # labels = self._toOrdinal(labels)
                # labels = self._toEvaluationOneHot(labels)

        # random crop
        if not self.randomCrop is None:
            shape = img.shape
            x = random.randint(0, shape[0] - self.randomCrop[0])
            y = random.randint(0, shape[1] - self.randomCrop[1])
            z = random.randint(0, shape[2] - self.randomCrop[2])
            img = img[x:x+self.randomCrop[0], y:y+self.randomCrop[1], z:z+self.randomCrop[2], :]
            if self.hasMasks: mask = mask[x:x + self.randomCrop[0], y:y + self.randomCrop[1], z:z + self.randomCrop[2], :]

        img = np.transpose(img, (3, 0, 1, 2))  # bring into NCWH format
        if self.hasMasks: 
            mask = np.resize(mask, (128, 128, 128 ,1))
            mask = np.transpose(mask, (3, 0, 1, 2))  # bring into NCWH format

        # to tensor
        #image = image[:, 0:32, 0:32, 0:32]
        img = torch.from_numpy(img)
        if self.hasMasks:
            #labels = labels[:, 0:32, 0:32, 0:32]
            mask = torch.from_numpy(mask) 

        #get pid
        #pid = self.file["pids_" + self.mode][index]

        if self.returnOffsets:
            print("set offset correctly")
        else:
            if self.hasMasks:
                assert img.shape == torch.Size([4, 128, 128, 128]) #CWHD
                assert mask.shape == torch.Size([1, 128, 128, 128]) #CWHD
                return img, mask, (img_path, mask_path, im_name) # str(pid)) #str(pid), labels
                # return image, labels, pid
            else:
                print("Specify mask path")
                return img, (img_path, "", im_name)

    def __len__(self):
        return len(self.imgs) #return length of images

    def openFileIfNotOpen(self):
        if self.file == None:
            self.file = h5py.File(self.filePath, "r")

    def _toEvaluationOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([shape[0], shape[1], shape[2], 3], dtype=np.float32)
        out[:, :, :, 0] = (labels != 0) #RoI WT (TC and ED)
        out[:, :, :, 1] = (labels != 0) * (labels != 2) #TC (everything besides 0 and 2)
        out[:, :, :, 2] = (labels == 4) # ET
        return out

    def _toOrignalCategoryOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([shape[0], shape[1], shape[2], 5], dtype=np.float32)
        for i in range(5):
            out[:, :, :, i] = (labels == i)
        return out

    def _toOrdinal(self, labels):
        return np.argmax(labels, axis=3)