# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch



def tstack(a):  # cv2.merge()
    a = np.asarray(a)
    return np.concatenate([x[..., np.newaxis] for x in a], axis=-1)


def tsplit(a):  # cv2.split()
    a = np.asarray(a)
    return np.array([a[..., x] for x in range(a.shape[-1])])


def masks_CFA_Bayer(shape):
    pattern = 'RGGB'
    channels = dict((channel, np.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1
    return tuple(channels[c].astype(bool) for c in 'RGB')
def masks_CFA_Bayer_cuda(shape):
    pattern = 'RGGB'
    channels = dict((channel, torch.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels[c].bool() for c in 'RGB')

def mosaic_CFA_Bayer(RGB):
    R_m, G_m, B_m = masks_CFA_Bayer(RGB.shape[0:2])
    mask = np.concatenate((R_m[..., np.newaxis], G_m[..., np.newaxis], B_m[..., np.newaxis]), axis=-1)
    # mask = tstack((R_m, G_m, B_m))
    mosaic = np.multiply(mask, RGB)  # mask*RGB
    CFA = mosaic.sum(2).astype(np.uint8)

    CFA4 = np.zeros((RGB.shape[0] // 2, RGB.shape[1] // 2, 4), dtype=np.uint8)
    CFA4[:, :, 0] = CFA[0::2, 0::2]
    CFA4[:, :, 1] = CFA[0::2, 1::2]
    CFA4[:, :, 2] = CFA[1::2, 0::2]
    CFA4[:, :, 3] = CFA[1::2, 1::2]

    return CFA, CFA4, mosaic, mask

def mosaic_CFA_Bayer_cuda(RGB):
    R_m, G_m, B_m = masks_CFA_Bayer_cuda(RGB.shape[0:2])
    mask = torch.cat((torch.unsqueeze(R_m,dim=-1), torch.unsqueeze(G_m,dim=-1), torch.unsqueeze(B_m,dim=-1)), dim=-1)
    # mask = tstack((R_m, G_m, B_m))
    mosaic = torch.multiply(mask.cuda(), RGB)  # mask*RGB
    CFA = mosaic.sum(2).int()

    CFA4 = torch.zeros((RGB.shape[0] // 2, RGB.shape[1] // 2, 4)).int()
    CFA4[:, :, 0] = CFA[0::2, 0::2]
    CFA4[:, :, 1] = CFA[0::2, 1::2]
    CFA4[:, :, 2] = CFA[1::2, 0::2]
    CFA4[:, :, 3] = CFA[1::2, 1::2]

    return CFA, CFA4, mosaic, mask

def gasuss_noise(image, mean=0, var=0.001):

    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    # cv.imshow("gasuss", out)
    return out



def compute_mask(pattern, im_shape):
    """
    Function compute_mask create a mask accordying to patter. The purpose
    of mask is to transform 2D image to 3D RGB.
    """
    # code from https://github.com/VLOGroup/joint-demosaicing-denoising-sem
    if pattern == 'bayer_rggb':
        r_mask = np.zeros(im_shape)
        r_mask[0::2, 0::2] = 1

        g_mask = np.zeros(im_shape)
        g_mask[::2, 1::2] = 1
        g_mask[1::2, ::2] = 1

        b_mask = np.zeros(im_shape)
        b_mask[1::2, 1::2] = 1
        mask = np.zeros(im_shape + (3,))
        mask[:, :, 0] = r_mask
        mask[:, :, 1] = g_mask
        mask[:, :, 2] = b_mask

    return mask

def compute_mask_cuda(pattern, im_shape):
    """
    Function compute_mask create a mask accordying to patter. The purpose
    of mask is to transform 2D image to 3D RGB.
    """
    # code from https://github.com/VLOGroup/joint-demosaicing-denoising-sem
    device = torch.device("cuda:1")
    if pattern == 'bayer_rggb':
        r_mask = torch.zeros(size=im_shape,device=device)
        r_mask[0::2, 0::2] = 1

        g_mask = torch.zeros(size=im_shape,device=device)
        g_mask[::2, 1::2] = 1
        g_mask[1::2, ::2] = 1

        b_mask = torch.zeros(size=im_shape,device=device)
        b_mask[1::2, 1::2] = 1
        mask = torch.zeros(im_shape + (3,),device=device)
        mask[:, :, 0] = r_mask
        mask[:, :, 1] = g_mask
        mask[:, :, 2] = b_mask

    return mask

def bayer2rgb_cuda(Im):
    device = torch.device("cuda:1")
    # Im = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)

    mask = compute_mask_cuda('bayer_rggb', Im.shape)
    mask = mask.int()
    image_mosaic = torch.zeros((Im.shape[0], Im.shape[1], 3),device=device).int()

    image_mosaic[:, :, 0] = mask[..., 0] * Im
    image_mosaic[:, :, 1] = mask[..., 1] * Im
    image_mosaic[:, :, 2] = mask[..., 2] * Im

    # image_input = torch.sum(image_mosaic, axis=2, dtype='uint16')
    # perform bilinear interpolation for bayer_rggb images

    image_input = image_mosaic.float() / 255  # /65535*255
    # 512 512 3
    # io.imwrite('1.png', image_input)
    return image_input.permute(2,0,1)
    # return 3 512 512

def bayer2rgb(Im):

    # Im = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)

    mask = compute_mask('bayer_rggb', Im.shape)
    mask = mask.astype(np.int32)
    image_mosaic = np.zeros((Im.shape[0], Im.shape[1], 3)).astype(np.int32)

    image_mosaic[:, :, 0] = mask[..., 0] * Im
    image_mosaic[:, :, 1] = mask[..., 1] * Im
    image_mosaic[:, :, 2] = mask[..., 2] * Im

    # image_input = np.sum(image_mosaic, axis=2, dtype='uint16')
    # perform bilinear interpolation for bayer_rggb images

    image_input = image_mosaic.astype(np.float32) / 255  # /65535*255
    # 512 512 3
    # io.imwrite('1.png', image_input)
    return image_input.transpose(2,0,1)
    # return 3 512 512





