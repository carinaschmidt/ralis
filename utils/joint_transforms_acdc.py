# Modified from pytorch original transforms
import math
import numbers
import random
import torch
import torch.nn as nn
#from skimage.util import crop
from skimage.transform import resize, rotate
from torchvision.transforms import functional as TVF

import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.shape == mask.shape #assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class ComposeRegion(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, region=None):
        #print("img.shape: ", img.shape, "mask.shape: ", mask.shape)
        #print("type(img) compose region ", type(img))
        #print("type(mask) compose region ", type(mask))
        assert img.shape == mask.shape #assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask, region=region)
        return img, mask
        
class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            print("RANDOMCROP size before random crop (if): ", size)
            self.size = (int(size), int(size)) ##TODO check how this works!! size is list [196, 196]
        else:
            print("RANDOMCROP size before random crop (else): ", size)
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            pad_img = nn.ConstantPad2d(self.padding, img.min().item()) #padding with the smallest value in tensor (not 0!)
            pad_mask = nn.ConstantPad2d(self.padding, mask.min().item())
            img = pad_img(img)
            mask = pad_mask(mask)
          
        assert img.shape == mask.shape
        w, h = img.shape
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            img_resized, mask_resized = resize(img, (tw, th), anti_aliasing=True), resize(mask, (tw, th), anti_aliasing=True)
            img, mask = torch.from_numpy(img_resized), torch.from_numpy(mask_resized)
            return img, mask

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        #img_cropped = crop(img, ((x1, x1+tw), (y1, y1+th)), copy=False)
        #mask_cropped = crop(mask, ((x1, x1+tw), (y1, y1+th)), copy=False)
        img = img[x1:x1+tw, y1:y1+th]
        mask = mask[x1:x1+tw, y1:y1+th] #slices torch tensor
        #img, mask = torch.from_numpy(img_cropped), torch.from_numpy(mask_cropped)

        return img, mask


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):

        assert img.shape == mask.shape
        w, h = img.shape
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            img_resized, mask_resized = resize(img, (tw, th), anti_aliasing=True), resize(mask, (tw, th), anti_aliasing=True)
            img, mask = torch.from_numpy(img_resized), torch.from_numpy(mask_resized)
            return img, mask
            #return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        #img_cropped = crop(img, ((i, i+tw), (j, j+th)), copy=False)
        #mask_cropped = crop(mask, ((i, i+tw), (j, j+th)), copy=False)
        #img, mask = torch.from_numpy(img_cropped), torch.from_numpy(mask_cropped)
        img = img[i:i+tw, j:j+th]
        mask = mask[i:i+tw, j:j+th]
        return img, mask
        #return img.crop((i, j, i + tw, j + th)), mask.crop((i, j, i + tw, j + th))


class RandomCropRegion(object):
    def __init__(self, size, padding=0, region_size=(128, 128)):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.region_size = region_size

    def __call__(self, img, mask, region):
        if self.padding > 0:
            #img = ImageOps.expand(img, border=self.padding, fill=0)
            #mask = ImageOps.expand(mask, border=self.padding, fill=0)
            pad_img = nn.ConstantPad2d(self.padding, img.min().item()) #padding with the smallest value in tensor (not 0!)
            pad_mask = nn.ConstantPad2d(self.padding, mask.min().item())
            img = pad_img(img)
            mask = pad_mask(mask)
        #img = F.pad(img, pad = (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        assert img.shape == mask.shape
        w, h = img.shape #img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            img_resized, mask_resized = resize(img, (tw, th), anti_aliasing=True), resize(mask, (tw, th), anti_aliasing=True)
            img, mask = torch.from_numpy(img_resized), torch.from_numpy(mask_resized)
            return img, mask#img.resize((tw, th), torch.nn.Upsample('Bilinear')), mask.resize((tw, th), torch.nn.Upsample('nearest'))

        # Get upper left corner of the crop
        y1 = random.randint(max(0, region[1] + self.region_size[1] // 2 - th),
                            min(region[1] + self.region_size[1] // 2, h - th))
        x1 = random.randint(max(0, region[0] + self.region_size[0] // 2 - tw),
                            min(region[0] + self.region_size[0] // 2, w - tw))
        
        #img_cropped = crop(img, ((x1, x1+tw), (y1, y1+th)), copy=False)
        #mask_cropped = crop(mask, ((x1, x1+tw), (y1, y1+th)), copy=False)
        #img, mask = torch.from_numpy(img_cropped), torch.from_numpy(mask_cropped)
        img = img[x1:x1+tw, y1:y1+th]
        mask = mask[x1:x1+tw, y1:y1+th]
        
        return img, mask
        #return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CropRegion(object):
    '''
    Crops the image so that the region is always contained inside
    '''

    def __init__(self, size, padding=0, region_size=128):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.region_size = 128

    def __call__(self, img, mask, region):
        if self.padding > 0:
            #img = ImageOps.expand(img, border=self.padding, fill=0)
            #mask = ImageOps.expand(mask, border=self.padding, fill=0)
            pad_img = nn.ConstantPad2d(self.padding, img.min().item()) #padding with the smallest value in tensor (not 0!)
            pad_mask = nn.ConstantPad2d(self.padding, mask.min().item())
            img = pad_img(img)
            mask = pad_mask(mask)
        #img = F.pad(img, padding = (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        assert img.shape == mask.shape#assert img.size == mask.size
        w, h = img.shape #img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            img_resized, mask_resized = resize(img, (tw, th), anti_aliasing=True), resize(mask, (tw, th), anti_aliasing=True)
            img, mask = torch.from_numpy(img_resized), torch.from_numpy(mask_resized)
            return img, mask
            #return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)
            #return img.resize((tw, th), torch.nn.Upsample('Bilinear')), mask.resize((tw, th), torch.nn.Upsample('nearest'))
        # Get upper left corner of the crop
        y1 = random.randint(max(0, region[1] + self.region_size - th), min(region[1], h - th))
        x1 = random.randint(max(0, region[0] + self.region_size - tw), min(region[0], w - tw))

        #img_cropped = crop(img, ((x1, x1+tw), (y1, y1+th)), copy=False)
        #mask_cropped = crop(mask, ((x1, x1+tw), (y1, y1+th)), copy=False)
        #img, mask = torch.from_numpy(img_cropped), torch.from_numpy(mask_cropped)
        img = img[x1:x1+tw, y1:y1+th]
        mask = mask[x1:x1+tw, y1:y1+th]
        return img, mask #img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, region=None):
        if random.random() < 0.5: #why ?
            #return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            #print('img.shape: ', type(img))
            return TVF.hflip(img), TVF.hflip(mask)
        return img, mask


#@carina: not used
# class FreeScale(object):
#     def __init__(self, size):
#         self.size = tuple(reversed(size))  # size: (h, w)

#     def __call__(self, img, mask):
#         assert img.shape == mask.shape #assert img.size == mask.size
#         #return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)
#         img_resized, mask_resized = resize(img, self.size, anti_aliasing=True), resize(mask, self.size, anti_aliasing=True)
#         img, mask = torch.from_numpy(img_resized), torch.from_numpy(mask_resized)
#         return img, mask #img.resize(self.size, torch.nn.Upsample('Bilinear')), mask.resize(self.size, torch.nn.Upsample('nearest'))

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask, region=None):
        assert img.shape == mask.shape #assert img.size == mask.size
        w, h = img.shape #img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            #return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
            img_resized, mask_resized = resize(img, (ow, oh), anti_aliasing=True), resize(mask, (ow, oh), anti_aliasing=True)
            img, mask = torch.from_numpy(img_resized), torch.from_numpy(mask_resized)
            return img, mask
        else:
            oh = self.size
            ow = int(self.size * w / h)
            #return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
            img_resized, mask_resized = resize(img, (ow, oh), anti_aliasing=True), resize(mask, (ow, oh), anti_aliasing=True)
            img, mask = torch.from_numpy(img_resized), torch.from_numpy(mask_resized)
            return img, mask

#@carina: not used
# class RandomSizedCrop(object):
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, img, mask):
#         assert img.shape == mask.shape #assert img.size == mask.size
#         for attempt in range(10):
#             area = img.shape[0] * img.shape[1]
#             target_area = random.uniform(0.45, 1.0) * area
#             aspect_ratio = random.uniform(0.5, 2)

#             w = int(round(math.sqrt(target_area * aspect_ratio)))
#             h = int(round(math.sqrt(target_area / aspect_ratio)))

#             if random.random() < 0.5:
#                 w, h = h, w

#             if w <= img.shape[0] and h <= img.shape[1]:
#                 x1 = random.randint(0, img.shape[0] - w)
#                 y1 = random.randint(0, img.shape[1] - h)

#                 # img_cropped = crop(img, ((x1, x1+w), (y1, y1+h)), copy=False)
#                 # mask_cropped = crop(mask, ((x1, x1+w), (y1, y1+h)), copy=False)
#                 # img, mask = torch.from_numpy(img_cropped), torch.from_numpy(mask_cropped)
#                 img = img[x1:x1+w, y1:y1+h]
#                 mask = mask[x1:x1+w, y1:y1+h]
#                 assert (img.shape == (w, h))
#                 img_resized, mask_resized = resize(img, (self.size, self.size), anti_aliasing=True), resize(mask, (self.size, self.size), anti_aliasing=True)
#                 img, mask = torch.from_numpy(img_resized), torch.from_numpy(mask_resized)
#                 return img, mask
#                 #return img.resize((self.size, self.size), torch.nn.Upsample('Bilinear')), mask.resize((self.size, self.size), torch.nn.Upsample('nearest'))

#         # Fallback
#         scale = Scale(self.size)
#         crop_ = CenterCrop(self.size)
#         return crop_(*scale(img, mask))


#@carina: not used
# class RandomRotate(object):
#     def __init__(self, degree):
#         self.degree = degree

#     def __call__(self, img, mask):
#         rotate_degree = random.random() * 2 * self.degree - self.degree
#         #return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)
#         return rotate(img, rotate_degree), rotate(mask, rotate_degree) #TODO border of image should be min()-value
#         #return img.rotate(rotate_degree, torch.nn.Upsample('Bilinear')), mask.rotate(rotate_degree, torch.nn.Upsample('nearest'))


#@carina: not used
# class RandomSized(object):
#     def __init__(self, size):
#         self.size = size
#         self.scale = Scale(self.size)
#         self.crop_ = RandomCrop(self.size)

#     def __call__(self, img, mask):
#         #assert img.size == mask.size
#         assert img.shape == mask.shape
#         w = int(random.uniform(0.5, 2) * img.shape[0])
#         h = int(random.uniform(0.5, 2) * img.shape[1])

#         #img, mask = img.resize((w, h), torch.nn.Bilinear), mask.resize((w,h), torch.nn.UpsamplingNearest2d)
#         img_resized, mask_resized = resize(img, (w,h), anti_aliasing=True), resize(mask, (w,h), anti_aliasing=True)
#         img, mask = torch.from_numpy(img_resized), torch.from_numpy(mask_resized)

#         return self.crop_(*self.scale(img, mask))


# class SlidingCropOld(object):
#     def __init__(self, crop_size, stride_rate, ignore_label):
#         self.crop_size = crop_size
#         self.stride_rate = stride_rate
#         self.ignore_label = ignore_label

#     def _pad(self, img, mask):
#         h, w = img.shape[: 2]
#         pad_h = max(self.crop_size - h, 0)
#         pad_w = max(self.crop_size - w, 0)
#         img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
#         mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'constant', constant_values=self.ignore_label)
#         return img, mask

#     def __call__(self, img, mask):
#         #assert img.size == mask.size
#         assert img.shape == mask.shape

#         #w, h = img.size
#         w, h = img.shape
#         long_size = max(h, w)

#         img = np.array(img)
#         mask = np.array(mask)

#         if long_size > self.crop_size:
#             stride = int(math.ceil(self.crop_size * self.stride_rate))
#             h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1
#             w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1
#             img_sublist, mask_sublist = [], []
#             for yy in xrange(h_step_num):
#                 for xx in xrange(w_step_num):
#                     sy, sx = yy * stride, xx * stride
#                     ey, ex = sy + self.crop_size, sx + self.crop_size
#                     img_sub = img[sy: ey, sx: ex, :]
#                     mask_sub = mask[sy: ey, sx: ex]
#                     img_sub, mask_sub = self._pad(img_sub, mask_sub)
#                     #img_sublist.append(Image.fromarray(img_sub.astype(np.uint8)).convert('RGB'))
#                     #mask_sublist.append(Image.fromarray(mask_sub.astype(np.uint8)).convert('P'))
#                     img_sublist.append(img_sub)
#                     mask_sublist.append(mask_sub)
#             return img_sublist, mask_sublist
#         else:
#             img, mask = self._pad(img, mask)
#             #img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
#             #mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
#             return img, mask


# class SlidingCrop(object):
#     def __init__(self, crop_size, stride_rate, ignore_label):
#         self.crop_size = crop_size
#         self.stride_rate = stride_rate
#         self.ignore_label = ignore_label

#     def _pad(self, img, mask):
#         h, w = img.shape[: 2]
#         pad_h = max(self.crop_size - h, 0)
#         pad_w = max(self.crop_size - w, 0)
#         img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
#         mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'constant', constant_values=self.ignore_label)
#         return img, mask, h, w

#     def __call__(self, img, mask):
#         assert img.size == mask.size

#         #w, h = img.size
#         w, h = img.shape
#         long_size = max(h, w)

#         img = np.array(img)
#         mask = np.array(mask)

#         if long_size > self.crop_size:
#             stride = int(math.ceil(self.crop_size * self.stride_rate))
#             h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1
#             w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1
#             img_slices, mask_slices, slices_info = [], [], []
#             for yy in xrange(h_step_num):
#                 for xx in xrange(w_step_num):
#                     sy, sx = yy * stride, xx * stride
#                     ey, ex = sy + self.crop_size, sx + self.crop_size
#                     img_sub = img[sy: ey, sx: ex, :]
#                     mask_sub = mask[sy: ey, sx: ex]
#                     img_sub, mask_sub, sub_h, sub_w = self._pad(img_sub, mask_sub)
#                     #img_slices.append(Image.fromarray(img_sub.astype(np.uint8)).convert('RGB'))
#                     #mask_slices.append(Image.fromarray(mask_sub.astype(np.uint8)).convert('P'))
#                     img_slices.append(img_sub)
#                     mask_slices.append(mask_sub)
#                     slices_info.append([sy, ey, sx, ex, sub_h, sub_w])
#             return img_slices, mask_slices, slices_info
#         else:
#             img, mask, sub_h, sub_w = self._pad(img, mask)
#             #img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
#             #mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
#             return [img], [mask], [[0, sub_h, 0, sub_w, sub_h, sub_w]]
