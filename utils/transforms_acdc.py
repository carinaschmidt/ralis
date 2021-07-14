# Modified from pytorch original transforms
# @carina modified for numpy/tensor inputs
import random

import numpy as np
import torch
#from skimage.transform import resize
from torchvision.transforms import functional as TVF

class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            #return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            #print('img.shape: ', type(img))
            return TVF.vflip(img)
        return img


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        #return torch.from_numpy(img)
        if type(img) != torch.Tensor:
            return torch.from_numpy(np.array(img, dtype=np.int32)).long() #long returns torch.int64 instead of torch.uint8
        else:
            return img.to(torch.long) #turns tensor into torch.int64
        #return torch.from_numpy(np.array(img, dtype=np.int32)).long()
        #return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class ImageToTensor(object):
    def __call__(self, img):
        if type(img) != torch.Tensor:
            return torch.from_numpy(np.array(img, dtype=np.float32)) #long returns torch.int64 instead of torch.uint8
        else:
            return img

# TODO change FreeScale
# class FreeScale(object):
#     # #def __init__(self, size, interpolation=Image.BILINEAR):
#      def __init__(self, size, interpolation=torch.nn.Bilinear):
#         self.size = tuple(reversed(size))  # size: (h, w)
#         self.interpolation = interpolation
#      def __call__(self, img):
#         return resize(img, (self.size, self.interpolation), anti_aliasing=True)


class FlipChannels(object):
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return torch.from_numpy(img)
        #return Image.fromarray(img.astype(np.uint8))


class MaskToTensorOneHot(object):
    def __init__(self, num_classes=19):
        self.num_classes=num_classes
    def __call__(self, img):
        return torch.from_numpy(np.eye(self.num_classes+1)[np.array(img, dtype=np.int32)]).long().transpose(0,2)