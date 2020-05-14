import random
import math
import numbers
import collections
import numpy as np
import torch
from PIL import Image, ImageOps

try:
    import accimage
except ImportError:
    accimage = None

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Scale(Resize):
    """
    Note: This transform is deprecated in favor of Resize.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the transforms.Scale transform is deprecated, " +
                      "please use transforms.Resize instead.")
        super(Scale, self).__init__(*args, **kwargs)
        
class CenterCrop(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return F.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class MultiScaleCornerCrop(object):
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, scales, size, interpolation=Image.BILINEAR):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, img, inv, flow):
        # print(img.size[0])
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.crop_position == 'tr':
            x1 = image_width - crop_size
            y1 = 1
            x2 = image_width
            y2 = crop_size
        elif self.crop_position == 'bl':
            x1 = 1
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.crop_position = self.crop_positions[random.randint(0, len(self.crop_positions) - 1)]

class FiveCrop(object):
    """Crop the given PIL Image into four corners and the central crop
    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.
    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.
    Example:
         >>> transform = Compose([
         >>>    FiveCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return F.five_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], interpolation=Image.BILINEAR, tenCrops=False):
        self.size = size
        self.interpolation = interpolation
        self.mean = mean
        self.std = std
        self.to_Tensor = ToTensor()
        self.normalize = Normalize(self.mean, self.std)
        self.tenCrops = tenCrops

    def __call__(self, img, inv, flow):
        # print(img.size[0])
        crop_size = self.size

        image_width = img.size[0]
        image_height = img.size[1]
        crop_positions = []
        # center
        center_x = image_width // 2
        center_y = image_height // 2
        box_half = crop_size // 2
        x1 = center_x - box_half
        y1 = center_y - box_half
        x2 = center_x + box_half
        y2 = center_y + box_half
        crop_positions += [[x1, y1, x2, y2]]
    # tl
        x1 = 0
        y1 = 0
        x2 = crop_size
        y2 = crop_size
        crop_positions += [[x1, y1, x2, y2]]
        # tr
        x1 = image_width - crop_size
        y1 = 1
        x2 = image_width
        y2 = crop_size
        crop_positions += [[x1, y1, x2, y2]]
        # bl
        x1 = 1
        y1 = image_height - crop_size
        x2 = crop_size
        y2 = image_height
        crop_positions += [[x1, y1, x2, y2]]
        # br
        x1 = image_width - crop_size
        y1 = image_height - crop_size
        x2 = image_width
        y2 = image_height
        crop_positions += [[x1, y1, x2, y2]]
        cropped_imgs = [img.crop(crop_positions[i]).resize((self.size, self.size), self.interpolation) for i in range(5)]
        # cropped_imgs = [img.resize(self.size, self.size, self.interpolation) for img in cropped_imgs]
        if self.tenCrops is True:
            if inv is True:
                flipped_imgs = [ImageOps.invert(cropped_imgs[i].transpose(Image.FLIP_LEFT_RIGHT)) for i in range(5)]
            else:
                flipped_imgs = [cropped_imgs[i].transpose(Image.FLIP_LEFT_RIGHT) for i in range(5)]
            cropped_imgs += flipped_imgs
                # cropped_imgs.append(img1.transpose(Image.FLIP_LEFT_RIGHT))

        tensor_imgs = [self.to_Tensor(img, inv, flow) for img in cropped_imgs]

        normalized_imgs = [self.normalize(img, inv, flow) for img in tensor_imgs]
        fiveCropImgs = torch.stack(normalized_imgs, 0)
        return fiveCropImgs

    def randomize_parameters(self):
        pass

class TenCrop(object):
    """Crop the given PIL Image into four corners and the central crop plus the flipped version of
    these (horizontal flipping is used by default)
    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        vertical_flip (bool): Use vertical flipping instead of horizontal
    Example:
         >>> transform = Compose([
         >>>    TenCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        return F.ten_crop(img, self.size, self.vertical_flip)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, vertical_flip={1})'.format(self.size, self.vertical_flip)

    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.mean = mean
        self.std = std
        self.fiveCrops = FiveCrops(self.size, self.mean, self.std, self.interpolation, True)

    def __call__(self, img, inv, flow):
        # print(img.size[0])
        return self.fiveCrops(img, inv, flow)

    def randomize_parameters(self):
        pass

    """Image and its horizontally flipped versions
    """

    def __init__(self, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]):
        self.mean = mean
        self.std = std
        self.to_Tensor = ToTensor()
        self.normalize = Normalize(self.mean, self.std)

    def __call__(self, img, inv, flow):
        # print(img.size[0])
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        if inv is True:
            img_flipped = ImageOps.invert(img_flipped)

        # center

        tensor_img = self.to_Tensor(img, inv, flow)
        tensor_img_flipped = self.to_Tensor(img_flipped, inv, flow)

        normalized_img = self.normalize(tensor_img, inv, flow)
        normalized_img_flipped = self.normalize(tensor_img_flipped, inv, flow)
        horFlippedTest_imgs = [normalized_img, normalized_img_flipped]
        horFlippedTest_imgs = torch.stack(horFlippedTest_imgs, 0)
        return horFlippedTest_imgs

    def randomize_parameters(self):
        pass