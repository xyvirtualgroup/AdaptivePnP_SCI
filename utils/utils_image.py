import os
import math
import random
import numpy as np
import torch
import cv2
from torchvision.utils import make_grid
from datetime import datetime

import matplotlib.pyplot as plt
import os


bayer = [[0,0], [0,1], [1,0], [1,1]]

def np2tch_cuda(a):
    return torch.from_numpy(a).cuda()
def cuda2np(a):
    return a.cpu().detach().numpy()
def masks_CFA_Bayer_tensor(shape, pattern='RGGB'):
    pattern = pattern.upper()
    channels = dict((channel, torch.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels[c].bool().cuda() for c in 'RGB')

def gen_bayer_mask(R_m, G_m, B_m):
    # torch.unsqueeze
    mask = torch.cat([R_m.unsqueeze(2), G_m.unsqueeze(2), B_m.unsqueeze(2)],dim=2)
    return mask

def gen_bayer_img(RGB,output_ch=1):# M N 3 B
    mask = gen_bayer_mask(masks_CFA_Bayer_tensor(RGB.shape[0],RGB.shape[1]))
    torch.repeat_interleave(mask.unsqueeze(3),RGB.shape[2],dim=3)
    bayer_img = RGB*mask
    bayer_img = torch.sum(bayer_img,dim=3).squeeze(3)
    if output_ch==1:
        return bayer_img # M N B
    elif output_ch==4:
        return oneCh2FourCh(bayer_img) # M/2 N/2 B 4

def fourCh2OneCh(RGGB):
    
    M ,N,B = RGGB.shape[0],RGGB.shape[1],RGGB.shape[2]
    oneCh = torch.zeros(M*2,N*2,B).cuda()
    for ib in range(len(bayer)): 
        b = bayer[ib]
        oneCh[b[0]::2, b[1]::2]=RGGB[...,ib] 
    return oneCh

def oneCh2FourCh(oneCh):
    M ,N,B = oneCh.shape[0],oneCh.shape[1],oneCh.shape[2]
    RGGB = torch.zeros(M//2,N//2,B,4).cuda()
    for ib in range(len(bayer)): 
        b = bayer[ib]
        RGGB[...,ib] = oneCh[b[0]::2,b[1]::2]
    return RGGB

def oneCh2ThreeCh(oneCh):
    M ,N,B = oneCh.shape[0],oneCh.shape[1],oneCh.shape[2]
    RGB = torch.zeros(M,N,3,B).cuda()

    RGB[0::2,0::2,0,:] = oneCh[0::2,0::2,:]
    RGB[0::2,1::2,1,:] = oneCh[0::2,1::2,:]
    RGB[1::2,0::2,1,:] = oneCh[1::2,0::2,:]
    RGB[1::2,1::2,2,:] = oneCh[1::2,1::2,:]
    return RGB
def fourCh2ThreeCh(RGGB):
    M ,N,B = RGGB.shape[0]*2,RGGB.shape[1]*2,RGGB.shape[2]
    RGB = torch.zeros(M,N,3,B).cuda()

    RGB[0::2,0::2,0,:] = RGGB[:,:,:,0]
    RGB[0::2,1::2,1,:] = RGGB[:,:,:,1]
    RGB[1::2,0::2,1,:] = RGGB[:,:,:,2]
    RGB[1::2,1::2,2,:] = RGGB[:,:,:,3]

    return RGB

def add_gasuss_noise_meas(meas, std=0.01):
    mean=0
    noise = np.random.normal(mean, std, meas.shape)
   
    out = meas + noise

    # out = np.clip(out, 0.0, 1.0)

    return out