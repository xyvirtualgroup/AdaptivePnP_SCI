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





def gen_masked_data_less(x,type='s',ratio=0.1):
	bs,ch,W,H = x.shape
	zero = 1e-6

	if type=='s': # spatial
		# mask1 = torch.ones(W,H)
		for i in range(int(W*H*ratio)):
			indx = np.random.randint(0,W)
			indy = np.random.randint(0,H)
			x[:,9:12,indx,indy] = zero
		# masked_x = torch.mul(mask1,x)
		return x
	elif type=='t': # temperol
		x[:,9:12,:,:] = zero
		return x
		# n_frame = ch//3*ratio
		# if n_frame<1:
		# 	n_frame=1
		# for i in range(int(n_frame)):
		# 	indc = np.random.randint(0,ch//3)
		# 	x[:,3*indc:3*indc+3,:,:] = zero
		
	elif type=='b': # block
		# mask2 = torch.ones(ch,W,H)
		for i in range(W*H//10):
			indx = np.random.randint(0,W)
			indy = np.random.randint(0,H)
			# indz = np.random.randint(9,12)
			indz = np.random.randint(0,8)
			x[indz,:,indx,indy] = zero
		# masked_x = torch.mul(mask1,x)
		return x
def gen_masked_data(x,type='s',ratio=0.1):
	bs,ch,W,H = x.shape
	zero = 1e-6

	if type=='s': # spatial
		# mask1 = torch.ones(W,H)
		for i in range(int(W*H*ratio)):
			indx = np.random.randint(0,W)
			indy = np.random.randint(0,H)
			x[:,9:12,indx,indy] = zero
		# masked_x = torch.mul(mask1,x)
		return x
	elif type=='t': # temperol
		x[:,9:12,:,:] = zero
		return x
		# n_frame = ch//3*ratio
		# if n_frame<1:
		# 	n_frame=1
		# for i in range(int(n_frame)):
		# 	indc = np.random.randint(0,ch//3)
		# 	x[:,3*indc:3*indc+3,:,:] = zero
		
	elif type=='b': # block
		# mask2 = torch.ones(ch,W,H)
		for i in range(W*H//10):
			indx = np.random.randint(0,W)
			indy = np.random.randint(0,H)
			indz = np.random.randint(9,12)
			x[:,indz,indx,indy] = zero
		# masked_x = torch.mul(mask1,x)
		return x


def mask_sequence(x,type_list=['s'],ratio=0.1):
    masked_seq = ()
    for i in type_list:
        masked = gen_masked_data(x,i)
        masked_seq = masked_seq+(masked,) 
    return masked_seq


    
bayer = [[0,0], [0,1], [1,0], [1,1]]

def np2tch_cuda(a):
    return torch.from_numpy(a).cuda()
def cuda2np(a):
    return a.cpu().detach().numpy()

def masks_CFA_Bayer_cuda(shape):
    pattern = 'RGGB'
    channels = dict((channel, torch.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels[c].bool() for c in 'RGB')

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
    if len(RGGB.shape)==3:
        M ,N = RGGB.shape[0],RGGB.shape[1]
        oneCh = torch.zeros(M*2,N*2).cuda()
        for ib in range(len(bayer)): 
            b = bayer[ib]
            oneCh[b[0]::2, b[1]::2]=RGGB[...,ib] 
    elif len(RGGB.shape)==4:
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

def add_gaussian_noise_meas_cuda(meas, std=0.01):
    meas = meas.detach().cpu().numpy()
    mean=0
    noise = np.random.normal(mean, std, meas.shape)
   
    out = meas + noise

    # out = np.clip(out, 0.0, 1.0)

    return torch.from_numpy(out).cuda()

def mosaic_CFA_Bayer_cuda(RGB):
    R_m, G_m, B_m = masks_CFA_Bayer_cuda(RGB.shape[0:2])
    mask = torch.cat((torch.unsqueeze(R_m,dim=-1), torch.unsqueeze(G_m,dim=-1), torch.unsqueeze(B_m,dim=-1)), dim=-1)
    # mask = tstack((R_m, G_m, B_m))
    mosaic = torch.multiply(mask.cuda(), RGB)  # mask*RGB
    CFA = mosaic.sum(2)

    CFA4 = torch.zeros((RGB.shape[0] // 2, RGB.shape[1] // 2, 4))
    CFA4[:, :, 0] = CFA[0::2, 0::2]
    CFA4[:, :, 1] = CFA[0::2, 1::2]
    CFA4[:, :, 2] = CFA[1::2, 0::2]
    CFA4[:, :, 3] = CFA[1::2, 1::2]

    return CFA, CFA4, mosaic, mask
