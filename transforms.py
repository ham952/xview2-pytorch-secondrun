#############################################################################################################                                         
# Data Augmentation                                                                                         #                               
# Applying multiple Augmentations to Image, Bounding boxes and Mask "simultaneously"                        #
# Tested / used for Instance Segmentation Task while using MaskRCNN from pytorch vision                     #
# author : Hamza Rafique https://github.com/ham952                                                          #                                                                                  #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license: #                                                        
# 1. Pytorch/torchvision : https://github.com/pytorch/vision                                                #
# 2. (Some parts)RandomRotation-transform : https://github.com/Paperspace/DataAugmentationForObjectDetection#                                                      #                                                                                                   #
#############################################################################################################

import random
import torch
from torchvision.ops import misc as misc_nn_ops
from torchvision.transforms import functional as F
import numpy as np
import cv2
import numbers


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomOrderCustom(object):
    """Apply a list of transformations in a random order
    But always apply the first transform i-e to_tensor
    """
    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms
        
    def __call__(self, img, target):
        trnsf_count = list(range(len(self.transforms)))
        ordr = trnsf_count[1:]
        random.shuffle(ordr)
        
        order = [None]*len(self.transforms)
        order[0] = trnsf_count[0]
        order[1:] = ordr        

        for i in order:
            img, target = self.transforms[i](img,target)
        return img, target

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(1)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(1)
        return image, target
    
class Resize(object):
    def __init__(self, inp_dim):
        self.inp_dim = inp_dim
        
    def __call__(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]])
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])

        min_size = float(torch.min(im_shape))
        max_size = float(torch.max(im_shape))

        size = float(self.inp_dim)
        scale_factor = size / min_size

        image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor, mode='bilinear',
            align_corners=False)[0]

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = self.resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        if "masks" in target:
            mask = target["masks"]
            mask = misc_nn_ops.interpolate(mask[None].float(), scale_factor=scale_factor)[0].byte()
            target["masks"] = mask

        return image, target

    def resize_boxes(self,boxes, original_size, new_size):
        # type: (Tensor, List[int], List[int])
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device) /
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)
    
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    

class RandomHue(object):
    def __init__(self, prob, hue):
        self.prob = prob
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        hue_factor = random.uniform(hue[0], hue[1])

        return Lambda(lambda img: F.adjust_hue(img, hue_factor))

    def __call__(self, img, target):

        if random.random() < self.prob:

            image = F.to_pil_image(img, mode = 'RGB')
            transform = self.get_params(self.hue)
            image= transform(image)
            img = F.to_tensor(image)

        return img, target

class RandomContrast(object):
    def __init__(self, prob, contrast):
        self.prob = prob
        self.contrast = self._check_input(contrast, 'contrast')

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value    

    @staticmethod
    def get_params(contrast):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        contrast_factor = random.uniform(contrast[0], contrast[1])
        
        return Lambda(lambda img: F.adjust_contrast(img, contrast_factor))

    def __call__(self, img, target):

        if random.random() < self.prob:

            image = F.to_pil_image(img, mode = 'RGB')
            transform = self.get_params(self.contrast)
            image= transform(image)
            img = F.to_tensor(image)

        return img, target

class RandomBrightness(object):
    def __init__(self, prob, brightness):
        self.prob = prob
        self.brightness = self._check_input(brightness, 'brightness')

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value    

    @staticmethod
    def get_params(brightness):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        brightness_factor = random.uniform(brightness[0], brightness[1])
        
        return Lambda(lambda img: F.adjust_brightness(img, brightness_factor))

    def __call__(self, img, target):

        if random.random() < self.prob:

            image = F.to_pil_image(img, mode = 'RGB')
            transform = self.get_params(self.brightness)
            image= transform(image)
            img = F.to_tensor(image)

        return img, target

class RandomSaturation(object):
    def __init__(self, prob, saturation):
        self.prob = prob
        self.saturation = self._check_input(saturation, 'saturation')

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value    

    @staticmethod
    def get_params(saturation):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        saturation_factor = random.uniform(saturation[0], saturation[1])
        
        return Lambda(lambda img: F.adjust_saturation(img, saturation_factor))

    def __call__(self, img, target):

        if random.random() < self.prob:

            image = F.to_pil_image(img, mode = 'RGB')
            transform = self.get_params(self.saturation)
            image= transform(image)
            img = F.to_tensor(image)

        return img, target

class RandomRotateCCW(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.rot90(1,[1,2])
            bbox = target["boxes"]

            h,w = height, width
            cx, cy = w//2, h//2
            bbox = bbox.numpy()
            corners = self.get_corners(bbox)
            corners = np.hstack((corners, bbox[:,4:]))
            corners[:,:8] = self.rotate_box(corners[:,:8], 90, cx, cy, h, w)
            new_bbox = self.get_enclosing_box(corners)
            bbox = new_bbox[:,:4]
            bbox = torch.from_numpy(bbox)
            
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].rot90(1,[1,2])
        return image, target

    def get_corners(self,bboxes):
        
        """Get corners of bounding boxes
        
        Parameters
        ----------
        
        bboxes: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`
        
        returns
        -------
        
        numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
            
        """
        width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
        height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
        
        x1 = bboxes[:,0].reshape(-1,1)
        y1 = bboxes[:,1].reshape(-1,1)
        
        x2 = x1 + width
        y2 = y1 
        
        x3 = x1
        y3 = y1 + height
        
        x4 = bboxes[:,2].reshape(-1,1)
        y4 = bboxes[:,3].reshape(-1,1)
        
        corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
        
        return corners

    def rotate_box(self,corners,angle,  cx, cy, h, w):
        
        """Rotate the bounding box.
        
        
        Parameters
        ----------
        
        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        
        angle : float
            angle by which the image is to be rotated
            
        cx : int
            x coordinate of the center of image (about which the box will be rotated)
            
        cy : int
            y coordinate of the center of image (about which the box will be rotated)
            
        h : int 
            height of the image
            
        w : int 
            width of the image
        
        Returns
        -------
        
        numpy.ndarray
            Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        """

        corners = corners.reshape(-1,2)
        corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
        
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        calculated = np.dot(M,corners.T).T
        
        calculated = calculated.reshape(-1,8)
        
        return calculated


    def get_enclosing_box(self,corners):
        """Get an enclosing box for ratated corners of a bounding box
        
        Parameters
        ----------
        
        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
        
        Returns 
        -------
        
        numpy.ndarray
            Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`
            
        """
        x_ = corners[:,[0,2,4,6]]
        y_ = corners[:,[1,3,5,7]]
        
        xmin = np.min(x_,1).reshape(-1,1)
        ymin = np.min(y_,1).reshape(-1,1)
        xmax = np.max(x_,1).reshape(-1,1)
        ymax = np.max(y_,1).reshape(-1,1)
        
        final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
        
        return final

# ToDo : make a single class of Random rotate , and extract two classes of CCW and CW
class RandomRotateCW(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.rot90(1,[2,1])
            bbox = target["boxes"]

            h,w = height, width
            cx, cy = w//2, h//2
            bbox = bbox.numpy()
            corners = self.get_corners(bbox)
            corners = np.hstack((corners, bbox[:,4:]))
            corners[:,:8] = self.rotate_box(corners[:,:8], -90, cx, cy, h, w)
            new_bbox = self.get_enclosing_box(corners)
            bbox = new_bbox[:,:4]
            bbox = torch.from_numpy(bbox)
            
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].rot90(1,[2,1])
        return image, target

    def get_corners(self,bboxes):
        
        """Get corners of bounding boxes
        
        Parameters
        ----------
        
        bboxes: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`
        
        returns
        -------
        
        numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
            
        """
        width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
        height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
        
        x1 = bboxes[:,0].reshape(-1,1)
        y1 = bboxes[:,1].reshape(-1,1)
        
        x2 = x1 + width
        y2 = y1 
        
        x3 = x1
        y3 = y1 + height
        
        x4 = bboxes[:,2].reshape(-1,1)
        y4 = bboxes[:,3].reshape(-1,1)
        
        corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
        
        return corners

    def rotate_box(self,corners,angle,  cx, cy, h, w):
        
        """Rotate the bounding box.      
        
        Parameters
        ----------        
        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        
        angle : float
            angle by which the image is to be rotated
            
        cx : int
            x coordinate of the center of image (about which the box will be rotated)
            
        cy : int
            y coordinate of the center of image (about which the box will be rotated)
            
        h : int 
            height of the image
            
        w : int 
            width of the image
        
        Returns
        -------
        
        numpy.ndarray
            Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        """

        corners = corners.reshape(-1,2)
        corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
        
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        calculated = np.dot(M,corners.T).T
        
        calculated = calculated.reshape(-1,8)
        
        return calculated


    def get_enclosing_box(self,corners):
        """Get an enclosing box for ratated corners of a bounding box
        
        Parameters
        ----------
        
        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
        
        Returns 
        -------
        
        numpy.ndarray
            Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`
            
        """
        x_ = corners[:,[0,2,4,6]]
        y_ = corners[:,[1,3,5,7]]
        
        xmin = np.min(x_,1).reshape(-1,1)
        ymin = np.min(y_,1).reshape(-1,1)
        xmax = np.max(x_,1).reshape(-1,1)
        ymax = np.max(y_,1).reshape(-1,1)
        
        final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
        
        return final


