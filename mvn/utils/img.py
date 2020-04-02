import numpy as np
import cv2
from PIL import Image

import torch

# IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
IMAGENET_MEAN, IMAGENET_STD = np.array([0.406, 0.456, 0.485]), np.array([0.225, 0.224, 0.229]) # BGR

def crop_image(image, bbox):
    """Crops area from image specified as bbox. Always returns area of size as bbox filling missing parts with zeros
    Args:
        image numpy array of shape (height, width, 3): input image
        bbox tuple of size 4: input bbox (left, upper, right, lower)

    Returns:
        cropped_image numpy array of shape (height, width, 3): resulting cropped image

    """

    image_pil = Image.fromarray(image)
    image_pil = image_pil.crop(bbox)

    return np.asarray(image_pil)


def resize_image(image, shape):
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def check_black_border(image, threshold=10):
    def get_border_length(x):
        subsequent_zeros = 0
        count = 0
        for i in x:
            if i==0:
                count += 1
            else:
                count = 0
            if count > subsequent_zeros:
                subsequent_zeros = count
        return subsequent_zeros  

    l1 = get_border_length(image.sum(0).mean(-1))
    l2 = get_border_length(image.sum(1).mean(-1))    

    return max(l1,l2) >= threshold


def get_square_bbox(bbox):
    """Makes square bbox from any bbox by stretching of minimal length side

    Args:
        bbox tuple of size 4: input bbox (left, upper, right, lower)

    Returns:
        bbox: tuple of size 4:  resulting square bbox (left, upper, right, lower)
    """

    left, upper, right, lower = bbox
    width, height = right - left, lower - upper

    if width > height:
        y_center = (upper + lower) // 2
        upper = y_center - width // 2
        lower = upper + width
    else:
        x_center = (left + right) // 2
        left = x_center - height // 2
        right = left + height

    return left, upper, right, lower


def scale_bbox(bbox, scale):
    left, upper, right, lower = bbox
    width, height = right - left, lower - upper

    x_center, y_center = (right + left) // 2, (lower + upper) // 2
    new_width, new_height = int(scale * width), int(scale * height)

    new_left = x_center - new_width // 2
    new_right = new_left + new_width

    new_upper = y_center - new_height // 2
    new_lower = new_upper + new_height

    return new_left, new_upper, new_right, new_lower


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def image_batch_to_numpy(image_batch):
    image_batch = to_numpy(image_batch)
    if len(image_batch.shape) == 4: 
        image_batch = np.transpose(image_batch, (0, 2, 3, 1)) # BxCxHxW -> BxHxWxC
    elif len(image_batch.shape) == 5:
        image_batch = np.transpose(image_batch, (0, 1, 3, 4, 2)) # BxTxCxHxW -> BxTxHxWxC
    else:
        raise RuntimeError('{} is a bad image ndims'.format(len(image_batch.shape)))        
    return image_batch


def image_batch_to_torch(image_batch):
    image_batch = np.transpose(image_batch, (0, 3, 1, 2)) # BxHxWxC -> BxCxHxW
    image_batch = to_torch(image_batch).float()
    return image_batch


def normalize_image(image):
    """Normalizes image using ImageNet mean and std

    Args:
        image numpy array of shape (h, w, 3): image

    Returns normalized_image numpy array of shape (h, w, 3): normalized image
    """
    return (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def denormalize_image(image):
    """Reverse to normalize_image() function"""
    return np.clip(255.0 * (image * IMAGENET_STD + IMAGENET_MEAN), 0, 255)
