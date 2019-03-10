import tensorflow as tf
import numpy as np
from scipy.signal import convolve2d
import functions as func
import config

def preprocess(image):
    # TODO use openface or other package to resize
    # align and crop to desired size & convert to [-1, 1] float32
    image = np.array(image)
    shape = image.shape
    desired_shape = (64,64,3)

    if shape != desired_shape:
        return

    image = np.float32((image/255)*2  - 1)

    return image


def single_to_batch(img):
    # fit image or mask to batch size
    img_batch = img.reshape(1, config.image_size, config.image_size, config.channels).repeat(config.BATCH_SIZE,0)
    return img_batch


def make_mask(mask_type, weighted_mask = config.weighted_mask, ratio = config.mask_ratio, nsize = config.window_size):
    # mask_type can be center, random, or half
    # config.image_size is int, defaults to 64
    # ratio is number < 0.5, either for amount masked in center or the percentage masked randomly
    mask = np.ones(shape=(config.image_size, config.image_size))

    assert(ratio < 0.5), "ratio must be less than 0.5"

    if mask_type == 'Center':
        start = int(config.image_size*ratio)
        stop = int(config.image_size*(1-ratio))
        mask[start:stop, start:stop] = 0
    elif mask_type == 'Random':
        rand_mask = np.random.randn(config.image_size, config.image_size)
        mask[rand_mask < ratio] = 0
    elif mask_type == 'Half':
        half = config.image_size // 2
        mask[:, half:] = 0
    else:
        assert(False), "mask_type must be Center, Random, or Half."

    if weighted_mask:
        ker = np.ones((nsize,nsize), dtype=np.float32)
        ker = ker/np.sum(ker)
        mask = mask * convolve2d(mask, ker, mode='same', boundary='symm')

    return mask.reshape(config.image_size, config.image_size, 1).repeat(3,2)


def get_masked_image(mask, image):
    msked_img = mask * image
    return msked_img


def bin_inv_mask(mask, single_channel=False):
    # binarize & invert mask for poisson blending
    mask[mask > 0] = 1
    mask = np.invert(mask.astype(int))
    if single_channel:
        mask = mask[:,:,0]

    return mask
