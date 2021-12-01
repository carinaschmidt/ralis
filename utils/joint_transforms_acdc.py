# Modified from pytorch original transforms
import numbers
import random
import torch
import torch.nn as nn
#from skimage.util import crop
from skimage.transform import resize, rotate, rescale
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import torchio as tio

import numpy as np

def crop_or_pad_slice_to_size(self, img_slice, nx, ny):
    """
    To crop the input 2D slice for the given dimensions
    input params :
        image_slice : 2D slice to be cropped
        nx : dimension in x
        ny : dimension in y
    returns:
        slice_cropped : cropped 2D slice
    """
    slice_cropped=np.zeros((nx,ny))
    x, y = img_slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = img_slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = img_slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = img_slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = img_slice[:, :]

    return slice_cropped


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.shape == mask.shape #assert img.size == mask.size
        for t in self.transforms:
            if not img is None:
                img, mask = t(img, mask)
        return img, mask

class ComposeRegion(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, region=None):
        assert img.shape == mask.shape #assert img.size == mask.size
        for t in self.transforms:
            if not img is None:
                img, mask = t(img, mask, region=region)
        return img, mask



class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
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
        img = img[x1:x1+tw, y1:y1+th]
        mask = mask[x1:x1+tw, y1:y1+th] #slices torch tensor
        return img, mask


class RandomCropRegion(object):
    def __init__(self, size, padding=0, region_size=(128, 128)):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.region_size = region_size


    def __call__(self, img, mask, region):
        #print("in random crop region")
        if self.padding > 0:
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
        
        img = img[x1:x1+tw, y1:y1+th]
        mask = mask[x1:x1+tw, y1:y1+th]
        # print("img.shape after randomcropregion: ", img.shape)
        # print("mask.shape after randomcropregion: ", mask.shape)
        if torch.is_tensor(img):
            return img, mask
        else:
            return torch.from_numpy(img), torch.from_numpy(mask)

class RandomCropRegion3D(object):
    def __init__(self, size, padding=0, region_size=(128, 128, 128)):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.region_size = region_size

    def __call__(self, img, mask, region):
        if self.padding > 0:
            pad_img = nn.ConstantPad3d(self.padding, img.min().item()) #padding with the smallest value in tensor (not 0!)
            pad_mask = nn.ConstantPad3d(self.padding, mask.min().item())
            img = pad_img(img)
            mask = pad_mask(mask)
        #img = F.pad(img, pad = (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        assert img.shape == mask.shape
        w, h, d = img.shape #img.size
        th, tw, td = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            print("If you reach this, you failed.")
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            img_resized, mask_resized = resize(img, (tw, th), anti_aliasing=True), resize(mask, (tw, th), anti_aliasing=True)
            img, mask = torch.from_numpy(img_resized), torch.from_numpy(mask_resized)
            return img, mask#img.resize((tw, th), torch.nn.Upsample('Bilinear')), mask.resize((tw, th), torch.nn.Upsample('nearest'))

        # Get upper left corner of the crop
        z1 = random.randint(max(0, region[2] + self.region_size[2] // 2 - td),
                            min(region[2] + self.region_size[2] // 2, d - td))

        y1 = random.randint(max(0, region[1] + self.region_size[1] // 2 - th),
                            min(region[1] + self.region_size[1] // 2, h - th))

        x1 = random.randint(max(0, region[0] + self.region_size[0] // 2 - tw),
                            min(region[0] + self.region_size[0] // 2, w - tw))
        
        img = img[x1:x1+tw, y1:y1+th, z1:z1+td]
        mask = mask[x1:x1+tw, y1:y1+th, z1:z1+td]
        if torch.is_tensor(img):
            return img, mask
        else:
            return torch.from_numpy(img), torch.from_numpy(mask)


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
        # Get upper left corner of the crop
        y1 = random.randint(max(0, region[1] + self.region_size - th), min(region[1], h - th))
        x1 = random.randint(max(0, region[0] + self.region_size - tw), min(region[0], w - tw))

        img = img[x1:x1+tw, y1:y1+th]
        mask = mask[x1:x1+tw, y1:y1+th]

        if torch.is_tensor(img):
            return img, mask
        else:
            return torch.from_numpy(img), torch.from_numpy(mask)#img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, region=None):
        if random.random() < 0.5: # to apply flipping only to half of the cases
            img, mask = np.fliplr(img), np.fliplr(mask)
            img, mask = torch.from_numpy(img.copy()), torch.from_numpy(mask.copy())
        #print("img.shape after random horizontally flip: ", img.shape)
        return img, mask

class RandomHorizontallyFlip02(object):
    def __call__(self, img, mask, region=None):
        if random.random() < 0.2: # to apply flipping only to half of the cases
            img, mask = np.fliplr(img), np.fliplr(mask)
            img, mask = torch.from_numpy(img.copy()), torch.from_numpy(mask.copy())
        return img, mask

class RandomScale(object):
    def __call__(self, img, mask, region=None):
        if random.random() < 0.5:
            assert img.shape == mask.shape #assert img.size == mask.size
            # print("asserted")
            # print("img: ", img)
            # print("mask: ", mask)
            scales = [0.8, 3]
            random_scale = np.random.uniform(scales[0], scales[1])
            ras = tio.RandomAffine(scales=(random_scale,random_scale))
            # print("RandomAffine")
            if not torch.is_tensor(img):
                img, mask = torch.from_numpy(img), torch.from_numpy(mask)
                print("transformed to torch")
            print("img.shape", img.shape)
            img_u, mask_u = img.unsqueeze(0).unsqueeze(3), mask.unsqueeze(0).unsqueeze(3)
            print("unsqueezed")
            # import ipdb
            # ipdb.set_trace()
            img_sc = ras(img_u)
            print("img_sc")
            mask_sc = ras(mask_u)
            print("mask_u")
            return img_sc[0,:,:,0], mask_sc[0,:,:,0]

# class Scale_Random(object):
#     def __call__(self, img, mask, region=None):
#         if random.random() < 0.5:
#             assert img.shape == mask.shape #assert img.size == mask.size
#             n_x, n_y = img.shape
#             #scale factor between 0.8 and 1.2
#             scale_fact_min=0.8
#             scale_fact_max=1.2  
#             scale_val = round(random.uniform(scale_fact_min,scale_fact_max), 2)
#             slice_rescaled, mask_rescaled = rescale(img, scale_val, order=1, preserve_range=True, mode = 'constant'), rescale(mask, scale_val, order=1, preserve_range=True, mode = 'constant')
#             #print(type(slice_rescaled))
#             img, mask = crop_or_pad_slice_to_size(self, slice_rescaled, n_x, n_y), crop_or_pad_slice_to_size(self, mask_rescaled, n_x, n_y)
#             img, mask = torch.from_numpy(img), torch.from_numpy(mask)
#         return img, mask

class SmallRotation(object):
    def __call__(self, img, mask, region=None):
        if random.random() < 0.5:
            angles = [-15,15]
            random_angle = np.random.uniform(angles[0], angles[1])
            img, mask = rotate(img, random_angle), rotate(mask, random_angle)
            img, mask = torch.from_numpy(img), torch.from_numpy(mask)
        return img, mask


class RandomRotation(object):
    def __call__(self, img, mask, region=None):
        if random.random() < 0.5:
            fixed_angle = 45
            random_angle = np.random.randint(8)*fixed_angle
            img, mask = rotate(img, random_angle), rotate(mask, random_angle)
            img, mask = torch.from_numpy(img), torch.from_numpy(mask)
        return img, mask


class ElasticDeformation(object):
    def __call__(self, img, mask, region=None):
        """Elastic deformation of images as described in Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in Proc. of the International Conference on Document Analysis and
        Recognition, 2003.

        # Arguments
        image: Numpy array with shape (height, width, channels). 
        alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
        Controls intensity of deformation.
        sigma: Float, sigma of gaussian filter that smooths the displacement fields.
        random_state: `numpy.random.RandomState` object for generating displacement fields.
        """
        assert img.shape == mask.shape
        if random.random() < 0.5:
            alpha_range=[90,100]
            sigma=[3,7]
            random_state=None
            #print("applying ElasticDeformation")
            if random_state is None:
                random_state = np.random.RandomState(None)

            if np.isscalar(alpha_range):
                alpha = alpha_range
            else:
                alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

            shape = img.shape
            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')#np.arange(shape[2]), indexing='ij')
            indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))#, np.reshape(z, (-1, 1))

            img = map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)
            mask = map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)
            if torch.is_tensor(img):
                return img, mask
            else:
                return torch.from_numpy(img), torch.from_numpy(mask)
        

class RandomContrastBrightness(object):
    def __call__(self, img, mask, region=None):
        # Create img transform function sequence
        assert img.shape == mask.shape
        if random.random() < 0.5:
            #print("Random Contrast Brightness")
            brightness=[0.01, 0.99]
            contrast=[0.1, 127]
            brightness = np.random.uniform(low=brightness[0], high=brightness[1])
            contrast = np.random.uniform(low=contrast[0], high=contrast[1])
            img = brightness * img + contrast
            if torch.is_tensor(img):
                return img, mask
            else:
                return torch.from_numpy(img), torch.from_numpy(mask)


class ContrastBrightnessAdjustment(object):
    def __call__(self, img, region=None):
        # Create img transform function sequence
        #assert img.shape == mask.shape
        #print("Random Contrast Brightness")
        if torch.is_tensor(img):    
          mean = torch.mean(img)
          brightness=[-0.1, 0.1]
          contrast=[0.8, 1.2]
          brightness = np.random.uniform(low=brightness[0], high=brightness[1])
          contrast = np.random.uniform(low=contrast[0], high=contrast[1])
          img_contrast = (img-mean) * contrast + mean
          img_brightness = img_contrast + brightness     
          return img_brightness#, mask
        else:
          print("--- img is not a tensor! -------------------")
            #return torch.from_numpy(img)#, torch.from_numpy(mask)

# for img and mask
# class ContrastBrightnessAdjustment(object):
#     def __call__(self, img, mask, region=None):
#         # Create img transform function sequence
#         assert img.shape == mask.shape
#         #print("Random Contrast Brightness")
#         brightness=[-0.1, 0.1]
#         contrast=[0.8, 1.2]
#         brightness = np.random.uniform(low=brightness[0], high=brightness[1])
#         contrast = np.random.uniform(low=contrast[0], high=contrast[1])
#         img = brightness * img + contrast
#         if torch.is_tensor(img):

#             return img, mask
#         else:
#             return torch.from_numpy(img), torch.from_numpy(mask)

# works only for 256 colors 
# class RandomContrastBrightness(object):
#     def __call__(self, img, mask, region=None):
#         # Create img transform function sequence
#         assert img.shape == mask.shape
#         if random.random() < 0.5:
#             brightness=[0.1, 2]
#             contrast=[0.1, 0.9]
#             #img = torch.stack((img, img, img), dim=0)
#             #mask = torch.stack((mask, mask, mask), dim=0)
#             img_transforms = []
#             img_mean = img.mean()

#             if np.isscalar(brightness):
#                 brightness = brightness
#             else:
#                 brightness = np.random.uniform(low=brightness[0], high=brightness[1])
            
#             if np.isscalar(contrast):
#                 contrast = contrast
#             else:
#                 contrast = np.random.uniform(low=contrast[0], high=contrast[1])
#             if brightness is not None:
#                 img = (img - img_mean) * brightness  + img_mean
#                 img_transforms.append(img)
#                 #img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
#             if contrast is not None:
#                 #img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
#                 img = (img - img_mean) * contrast  + img_mean
#                 img_transforms.append(img)
#             random.shuffle(img_transforms)
#             for func in img_transforms:
#                 jittered_img = func(img)
#                 jittered_mask = func(mask)
#             return jittered_img, jittered_mask
#             #return only first of stacked img / mask
#             #return jittered_img[0], jittered_mask[0]
#         else:
#             return img, mask


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
#         return rotate(img, rotate_degree), rotate(mask, rotate_degree)
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
