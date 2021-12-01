# Author: @Carina
import torch.nn as nn
import torch
import numpy as np
import random
from monai.transforms import ResizeWithPadOrCrop, RandRotate, RandAffine, Rand2DElastic, Resize
from skimage.transform import resize, rotate, rescale
import numbers


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, region=None):
        for t in self.transforms:
            if not img is None:
                img, mask = t(img, mask, region=region)
        return img, mask

class DoubleCropOrPad(object):
    def __init__(self, input_size):
        self.input_size = input_size
    def __call__(self, img, mask, region=None):
        crop = ResizeWithPadOrCrop(spatial_size=self.input_size[0])
        img = crop(img)
        mask= crop(mask)
        if type(img) != torch.Tensor:
            img = torch.from_numpy(img.copy())
            mask = torch.from_numpy(mask.copy())
        return img, mask


class DoubleRandomRotate(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, region=None):
        p = random.random()
        if p < self.p:
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            rotate = RandRotate(range_x=(-15*np.pi/180, 15*np.pi/180), prob=1)
            img = rotate(img)
            mask = rotate(mask)
            if type(img) != torch.Tensor:
                img = torch.from_numpy(img.copy())
                mask = torch.from_numpy(mask.copy())
        if len(img.shape) == 3:
                img = torch.squeeze(img)
        if len(mask.shape) == 3:
                mask = torch.squeeze(mask)
        return img, mask


class DoubleRandomScale(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, region=None):
        p = random.random()
        if p < self.p:
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

            scale = RandAffine(scale_range=(-0.1, 0.1), prob=1)
            img = scale(img)
            mask = scale(mask)
            if type(img) != torch.Tensor:
                img = torch.from_numpy(img.copy())
                mask = torch.from_numpy(mask.copy())
        if len(img.shape) == 3:
                img = torch.squeeze(img)
        if len(mask.shape) == 3:
                mask = torch.squeeze(mask)
        return img, mask

class DoubleHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, region=None):
        p = random.random()
        if p < self.p:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
            if type(img) != torch.Tensor:
                img = torch.from_numpy(img.copy())
                mask = torch.from_numpy(mask.copy())
        return img, mask

class DoubleRand2DElastic(object):
    def __init__(self, p=0.5, input_size=[128,128]):
        self.p = p
        self.input_size = input_size

    def __call__(self, img, mask, region=None):
        p = random.random()
        if p < self.p:
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            elas = Rand2DElastic(prob=1.0,spacing=(self.input_size[0]/2, self.input_size[1]/2),magnitude_range=(0, 7),padding_mode="zeros")
            img = elas(img)
            mask = elas(mask)
            if type(img) != torch.Tensor:
                img = torch.from_numpy(img.copy())
                mask = torch.from_numpy(mask.copy())
        if len(img.shape) == 3:
                img = torch.squeeze(img)
        if len(mask.shape) == 3:
                mask = torch.squeeze(mask)
        return img, mask

class ContrastBrightnessAdjustment(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, mask, region=None):
        p = random.random()
        if p < self.p:
            brightness = [-0.1, 0.1]
            contrast = [0.8, 1.2]
            b = np.random.uniform(low=brightness[0], high=brightness[1])
            c = np.random.uniform(low=contrast[0], high=contrast[1])
            mean_img = torch.mean(img)
            img_contrast = (img-mean_img) * c + mean_img
            img = img_contrast + b
            if type(img) != torch.Tensor:
                img = torch.from_numpy(img.copy())
                mask = torch.from_numpy(mask.copy())
        return img, mask

class DoubleScale(object):
    def __init__(self, scale_size=[150, 150]):
        self.scale_size = scale_size

    def __call__(self, img, mask, region=None):
        if len(img.shape) == 2:
                img = img.unsqueeze(0)
        if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
        resize = Resize(self.scale_size)
        img = resize(img)
        mask = resize(mask)
        if type(img) != torch.Tensor:
            img = torch.from_numpy(img.copy())
            mask = torch.from_numpy(mask.copy())
        if len(img.shape) == 3:
                img = torch.squeeze(img)
        if len(mask.shape) == 3:
                mask = torch.squeeze(mask)
        return img, mask


class DoubleCropRandomRegion(object):
    def __init__(self, size, padding=0, region_size=(64, 64)):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.region_size = region_size

    def __call__(self, img, mask, region):
        if self.padding > 0:
            pad_img = nn.ConstantPad2d(self.padding, img.min().item()) #padding with the smallest value in tensor (not 0!)
            pad_mask = nn.ConstantPad2d(self.padding, mask.min().item())
            img = pad_img(img)
            mask = pad_mask(mask)
        if len(img.shape) == 3:
            w, h = img[-1].shape #img.shape of second and third dim
            th, tw = self.size
            if w == tw and h == th:
                return img, mask
            if w < tw or h < th:
                img_resized, mask_resized = resize(img, (tw, th), anti_aliasing=True), resize(mask, (tw, th), anti_aliasing=True)
                img, mask = torch.from_numpy(img_resized), torch.from_numpy(mask_resized)
                return img, mask
        elif len(img.shape)== 2: 
            w, h = img.shape 
            th, tw = self.size
            if w == tw and h == th:
                return img, mask
            if w < tw or h < th:
                img_resized, mask_resized = resize(img, (tw, th), anti_aliasing=True), resize(mask, (tw, th), anti_aliasing=True)
                img, mask = torch.from_numpy(img_resized), torch.from_numpy(mask_resized)
                return img, mask
        else:
            print("image needs to be 2 or 3-dimensional! ")
            
        # Get upper left corner of the crop
        y1 = random.randint(max(0, region[1] + self.region_size[1] // 2 - th),
                            min(region[1] + self.region_size[1] // 2, h - th))
        
        x1 = random.randint(max(0, region[0] + self.region_size[0] // 2 - tw),
                            min(region[0] + self.region_size[0] // 2, w - tw))
        if len(img.shape) == 3:
            img = img[:, x1:x1+tw, y1:y1+th]
        elif len(img.shape) == 2:
            img = img[x1:x1+tw, y1:y1+th]
        
        if len(mask.shape) == 3:
            mask = mask[:, x1:x1+tw, y1:y1+th]
        elif len(mask.shape) == 2:
            mask = mask[x1:x1+tw, y1:y1+th]  
        else: 
            print("change shape to 2 or 3 dim")  
        if torch.is_tensor(img):
            return img, mask
        else:
            return torch.from_numpy(img), torch.from_numpy(mask)
