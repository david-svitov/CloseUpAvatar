
import torch
import numpy as np
import cv2 as cv
import math
import warnings

import torch
import torch.nn.functional as F

from torchvision.io import encode_jpeg

def calc_bbox(mask, margin=0):
    # [left right)  [top down)
    height, width = mask.shape
    mask_sum0 = np.sum(mask, axis=0) > 0
    mask_sum1 = np.sum(mask, axis=1) > 0
    left = np.argmax(mask_sum0)
    right = width - np.argmax(mask_sum0[::-1])
    top = np.argmax(mask_sum1)
    down = height - np.argmax(mask_sum1[::-1])

    if margin != 0:
        left = max(0, left - margin)
        right = min(width, right + margin)
        top = max(0, top - margin)
        down = min(height, down + margin)
    bbox = np.array([left, top, right, down], dtype=int)
    return bbox

try:
    from nvjpeg import NvJpeg
    nj = NvJpeg()
except:
    pass 

def encode_bytes(image, image_encode_method=''):
    if image_encode_method == 'cpu':
        image_byte = cv.imencode('.jpg', image)
    elif image_encode_method == 'gpu':
        image_byte = nj.encode(image, 90)
    elif image_encode_method == 'torch':

        if torch.__version__ < "2.4.0":
            warnings.warn(
                "Warning: torch version is less than 2.4.0. "
                "GPU encoding of JPEG images using encode_jpeg is only available in torch version 2.4.0 or higher.",
                UserWarning
            )

        image = torch.flip(image, dims=[2]).permute(2,0,1)
        torch.cuda.synchronize()
        image_byte = encode_jpeg(image.cpu(), 90).numpy().tobytes()
    else:
        image_byte = image
    return image_byte

# code from AnimatableGaussians https://github.com/lizhe00/AnimatableGaussians/blob/master/main_avatar.py
def crop_image(bg_color_cuda, gt_mask, patch_size, randomly, *args):
    """
    :param gt_mask: (H, W)
    :param patch_size: resize the cropped patch to the given patch_size
    :param randomly: whether to randomly sample the patch
    :param args: input images with shape of (C, H, W)
    """
    mask_uv = torch.argwhere(gt_mask > 0.)
    if len(mask_uv) <= 10: # we don't have something to select from
        min_v, min_u = 0, 0
        max_v, max_u = mask_uv.shape[0], mask_uv.shape[1]
    else:
        min_v, min_u = mask_uv.min(0)[0]
        max_v, max_u = mask_uv.max(0)[0]

    len_v = max_v - min_v
    len_u = max_u - min_u
    max_size = max(len_v, len_u)

    cropped_images = []
    if randomly and max_size > patch_size:
        random_v = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
        random_u = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
    for image in args:
        cropped_image = bg_color_cuda[:, None, None] * torch.ones((3, max_size, max_size), dtype = image.dtype, device = image.device)
        if len_v > len_u:
            start_u = (max_size - len_u) // 2
            cropped_image[:, :, start_u: start_u + len_u] = image[:, min_v: max_v, min_u: max_u]
        else:
            start_v = (max_size - len_v) // 2
            cropped_image[:, start_v: start_v + len_v, :] = image[:, min_v: max_v, min_u: max_u]

        if randomly and max_size > patch_size:
            cropped_image = cropped_image[:, random_v: random_v + patch_size, random_u: random_u + patch_size]
        else:
            cropped_image = F.interpolate(cropped_image[None], size = (patch_size, patch_size), mode = 'bilinear')[0]
        cropped_images.append(cropped_image)

    if len(cropped_images) > 1:
        return cropped_images
    else:
        return cropped_images[0]