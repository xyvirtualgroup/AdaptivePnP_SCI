from packages.colour_demosaicing.bayer.demosaicing.malvar2004 import demosaicing_CFA_Bayer_Malvar2004_tensor
import time
import math
import skimage
import numpy as np
import torch
import cv2
import imageio as io
from statistics import mean
import matplotlib.pyplot as plt
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
# from packages.vnlnet.test import vnlnet
from packages.ffdnet.test_ffdnet_ipol import ffdnet_rgb_denoise_full_tensor, ffdnet_vdenoiser
from packages.ffdnet.test_ffdnet_ipol import ffdnet_rgb_denoise
# from packages.fastdvdnet.test_fastdvdnet import fastdvdnet_denoiser
from packages.colour_demosaicing.bayer import demosaicing_CFA_Bayer_bilinear
from packages.colour_demosaicing.bayer import demosaicing_CFA_Bayer_Malvar2004
from packages.colour_demosaicing.bayer import demosaicing_CFA_Bayer_Menon2007
from utilspy_tensor import (A_, At_, psnr)
import tqdm
if skimage.__version__ < '0.18':
    from skimage.measure import (compare_psnr, compare_ssim)
else: # skimage.measure deprecated in version 0.18 ( -> skimage.metrics )
    from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
# from utils_full import utils_logger
# from utils_full import utils_image as util2
from packages.colour_demosaicing.bayer import masks_CFA_Bayer
import tqdm

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

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
def fourCh2OneCh(RGGB):
    bayer = [[0,0], [0,1], [1,0], [1,1]]
    M ,N,B = RGGB.shape[0],RGGB.shape[1],RGGB.shape[2]
    oneCh = torch.zeros(M*2,N*2,B).cuda()
    for ib in range(len(bayer)): 
        b = bayer[ib]
        oneCh[b[0]::2, b[1]::2]=RGGB[...,ib] 
    return oneCh


def admm_denoise_bayer(y_bayer, Phi_bayer, _lambda=1, gamma=0.01,
                denoiser='tv', iter_max=50, noise_estimate=True, sigma=None, 
                tv_weight=0.1, tv_iter_max=5, multichannel=True, x0_bayer=None, 
                X_orig=None, model=None, show_iqa=True,demosaic_method = 'malvar2004',lr_=0.000001, inital_iter=1, if_continue=False,inter_time=5,logf = None,device=0):


    bayer = [[0,0], [0,1], [1,0], [1,1]] # `BGGR` Bayer pattern

    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)

    # stack the bayer channels at the last dimension [consistent to image color channels]
    (nrow, ncol, nmask) = Phi_bayer.shape
    yall = torch.zeros([nrow//2, ncol//2, 4], dtype=torch.float32)
    Phiall = torch.zeros([nrow//2, ncol//2, nmask, 4], dtype=torch.float32)
    Phi_sumall = torch.zeros([nrow//2, ncol//2, 4], dtype=torch.float32)
    x0all = torch.zeros([nrow//2, ncol//2, nmask, 4], dtype=torch.float32)
    
    # iterative solve for each Bayer channel
    for ib in range(len(bayer)): 
        b = bayer[ib]
        yall[...,ib] = y_bayer[b[0]::2, b[1]::2]
        Phiall[...,ib] =  Phi_bayer[b[0]::2, b[1]::2]
        # y = y_bayer[b[0]::2][b[1]::2]
        # Phi = Phi_bayer[b[0]::2][b[1]::2]

        # A  = lambda x :  A_(x, Phi) # forward model function handle
        # At = lambda y : At_(y, Phi) # transpose of forward model

        Phib = Phiall[...,ib]
        Phib_sum = np.sum(Phib, axis=2)
        Phib_sum[Phib_sum==0] = 1

        Phi_sumall[...,ib] = Phib_sum

        # [0] initialization
        if x0_bayer is None:
            # x0 = At(y, Phi) # default start point (initialized value)
            x0all[...,ib] = At_(yall[...,ib], Phiall[...,ib]) # default start point (initialized value)
        else:
            x0all[...,ib] = x0_bayer[b[0]::2,b[1]::2]

    # y1 = torch.zeros(y.shape)
    y1all = torch.zeros_like(yall) 
    # [1] start iteration for reconstruction
    xall = x0all # initialization
    thetaall = x0all
    x_bayer = torch.zeros_like(Phi_bayer)
    ball = torch.zeros_like(x0all)
    b = torch.zeros_like(x0all) #

    psnr_all = []
    k = 0
    for idx, nsig in enumerate(sigma): # iterate all noise levels
        for it in range(iter_max[idx]): 
            start_time = time.time()

            for ib in range(len(bayer)): # iterate all bayer channels
                yb = A_(thetaall[...,ib]+ball[...,ib], Phiall[...,ib])
                xall[...,ib] = thetaall[...,ib]+ball[...,ib] + _lambda*(At_((yall[...,ib]-yb)/(Phi_sumall[...,ib]+gamma), Phiall[...,ib])) # GAP

            end_time = time.time()
            # print('    Euclidean projection eclipsed in {:.3f}s.'.format(end_time-start_time))
            # joint Bayer multi-channel denoising
            # switch denoiser 
            if denoiser.lower() == 'tv': # total variation (TV) denoising
                thetaall_vch = (xall-ball).reshape([nrow//2, ncol//2, nmask*4])
                thetaall_vch = denoise_tv_chambolle(thetaall_vch, tv_weight, n_iter_max=tv_iter_max, 
                                        multichannel=multichannel)
                thetaall = thetaall_vch.reshape([nrow//2, ncol//2, nmask, 4])
                # xall = xall.clip(0., 1.) # [0,1]
            elif denoiser.lower() == 'ffdnet': # FFDNet frame-wise video denoising
                xall_vch = xall.reshape([nrow//2, ncol//2, nmask*4])
                xall_vch = ffdnet_vdenoiser(xall_vch, nsig, model)
                xall = xall_vch.reshape([nrow//2, ncol//2, nmask, 4])
            elif denoiser.lower() == 'fastdvdnet': # FastDVDnet video denoising
                # # option 1 - run denoising twice
                # xrgb1 = xall[..., [0,1,3]] # R-G1-B (H x W x F x C)
                # xrgb2 = xall[..., [0,2,3]] # R-G2-B (H x W x F x C)
                # xrgb1 = fastdvdnet_denoiser(xrgb1, nsig, model)
                # xrgb2 = fastdvdnet_denoiser(xrgb2, nsig, model)
                # xall[...,0] = (xrgb1[...,0] + xrgb2[...,0])/2 # R  channel (average over two)
                # xall[...,1] = xrgb1[...,1]                    # G1 channel (average over two)
                # xall[...,2] = xrgb2[...,1]                    # G2 channel (average over two)
                # xall[...,3] = (xrgb1[...,2] + xrgb2[...,2])/2 # B  channel (average over two)
                # option 2 - run deniosing once
                thetargb1 = (xall-ball)[..., [3,1,0]] # R-G1-B (H x W x F x C)
                thetargb1 = fastdvdnet_denoiser(thetargb1, nsig, model)
                thetaall[...,3] = thetargb1[...,0] # R  channel (average over two)
                thetaall[...,2] = thetargb1[...,1] # G1=G2 channel (average over two)
                thetaall[...,1] = thetargb1[...,1] # G2=G1 channel (average over two)
                thetaall[...,0] = thetargb1[...,2] # B  channel (average over two)
            else:
                raise ValueError('Unsupported denoiser {}!'.format(denoiser))
            ball = ball - (xall-thetaall) # update residual

            # [optional] calculate image quality assessment, i.e., PSNR for 
            # every five iterations
            if show_iqa and X_orig is not None:
                for ib in range(len(bayer)): 
                    b = bayer[ib]
                    x_bayer[b[0]::2, b[1]::2] = xall[...,ib]
                psnr_all.append(psnr(X_orig, x_bayer))
                if (k+1)%5 == 0:
                    if not noise_estimate and nsig is not None:
                        if nsig < 1:
                            print('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                            'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                            k+1, nsig*255, psnr_all[k]))
                        else:
                            print('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                                k+1, nsig, psnr_all[k]))
                    else:
                        print('  ADMM-{0} iteration {1: 3d}, ' 
                            'PSNR {2:2.2f} dB.'.format(denoiser.upper(), 
                            k+1, psnr_all[k]))
            k = k+1

    for ib in range(len(bayer)): 
        b = bayer[ib]
        x_bayer[b[0]::2, b[1]::2] = xall[...,ib]
    

    return x_bayer, psnr_all

#%%
#gray
def admm_denoise(y, Phi_sum, A, At, _lambda=1, gamma=0.01, 
                denoiser='tv', iter_max=50, noise_estimate=False, sigma=None, 
                tv_weight=0.1, tv_iter_max=5, multichannel=True, x0=None, 
                X_orig=None, show_iqa=True,model=None):
    # [0] initialization
    if x0 is None:
        x0 = At(y) # default start point (initialized value)
    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)
    # [1] start iteration for reconstruction
    x = x0 # initialization
    theta = x0
    b = torch.zeros_like(x0)
    psnr_all = []
    k = 0
    for idx, nsig in enumerate(sigma): # iterate all noise levels
        for it in range(iter_max[idx]):
            # Euclidean projection
            yb = A(theta+b)
            x = (theta+b) + _lambda*(At((y-yb)/(Phi_sum+gamma))) # ADMM
            # switch denoiser 
            if denoiser.lower() == 'tv': # total variation (TV) denoising
                theta = denoise_tv_chambolle(x-b, tv_weight, n_iter_max=tv_iter_max, 
                                         multichannel=multichannel)
            elif denoiser.lower() == 'wavelet': # wavelet denoising
                if noise_estimate or nsig is None: # noise estimation enabled
                    theta = denoise_wavelet(x-b, multichannel=multichannel)
                else:
                    theta = denoise_wavelet(x-b, sigma=nsig, multichannel=multichannel)
            # elif denoiser.lower() == 'vnlnet': # Video Non-local net denoising
            #     theta = vnlnet(np.expand_dims((x-b).transpose(2,0,1),3), nsig)
            #     theta = np.transpose(theta.squeeze(3),(1,2,0))
            elif denoiser.lower() == 'ffdnet': # FFDNet frame-wise video denoising
                theta = ffdnet_vdenoiser(x-b, nsig, model)
            elif denoiser.lower() == 'fastdvdnet': # FastDVDnet video denoising
                theta = fastdvdnet_denoiser(x-b, nsig, model, gray=True)
            else:
                raise ValueError('Unsupported denoiser {}!'.format(denoiser))
            
            theta = np.clip(theta,0,1)
            b = b - (x-theta) # update residual
            # [optional] calculate image quality assessment, i.e., PSNR for 
            # every five iterations
            if show_iqa and X_orig is not None:
                psnr_all.append(psnr(X_orig, x))
                if (k+1)%5 == 0:
                    if not noise_estimate and nsig is not None:
                        if nsig < 1:
                            print('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                              'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                               k+1, nsig*255, psnr_all[k]))
                        else:
                            print('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                                k+1, nsig, psnr_all[k]))
                    else:
                        print('  ADMM-{0} iteration {1: 3d}, ' 
                              'PSNR {2: 2.2f} dB.'.format(denoiser.upper(), 
                               k+1, psnr_all[k]))
            k = k+1
    
    psnr_ = []
    ssim_ = []
    nmask = x.shape[2]
    if X_orig is not None:
        for imask in range(nmask):
            psnr_.append(compare_psnr(X_orig[:,:,imask], x[:,:,imask], data_range=1.))
            ssim_.append(compare_ssim(X_orig[:,:,imask], x[:,:,imask], data_range=1.))
    return x, psnr_, ssim_, psnr_all


def admm_denoise_bayer_demosaic(y_bayer, Phi_bayer, _lambda=1, gamma=0.01,
                denoiser='tv', iter_max=50, noise_estimate=True, sigma=None, 
                 x0_bayer=None, 
                X_orig=None, model=None, show_iqa=True,demosaic_method = 'malvar2004',lr_=0.000001,
                 inital_iter=1, if_continue=False,inter_time=5,logf = None,useGPU=True,device=0,update_=False,update_per_iter=1):




    y_bayer = np2tch_cuda(y_bayer)
    Phi_bayer = np2tch_cuda(Phi_bayer)
    # y_bayer = np2tch_cuda(y_bayer)
    # y_bayer = np2tch_cuda(y_bayer)
    bayer = [[0,0], [0,1], [1,0], [1,1]] # `RGGB` Bayer pattern

    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)

    # stack the bayer channels at the last dimension [consistent to image color channels]
    (nrow, ncol, nmask) = Phi_bayer.shape
    yall = torch.zeros([nrow//2, ncol//2, 4], dtype=torch.float32).cuda()
    Phiall = torch.zeros([nrow//2, ncol//2, nmask, 4], dtype=torch.float32).cuda()
    Phi_sumall = torch.zeros([nrow//2, ncol//2, 4], dtype=torch.float32).cuda()
    x0all = torch.zeros([nrow//2, ncol//2, nmask, 4], dtype=torch.float32).cuda()
    
    # iterative solve for each Bayer channel
    for ib in range(len(bayer)): 
        b = bayer[ib]
        yall[...,ib] = y_bayer[b[0]::2, b[1]::2]
        Phiall[...,ib] =  Phi_bayer[b[0]::2, b[1]::2]

        Phib = Phiall[...,ib]
        Phib_sum = torch.sum(Phib, dim=2)
        Phib_sum[Phib_sum==0] = 1

        Phi_sumall[...,ib] = Phib_sum

        # [0] initialization
        if x0_bayer is None:
            # x0 = At(y, Phi) # default start point (initialized value)
            x0all[...,ib] = At_(yall[...,ib], Phiall[...,ib]) # default start point (initialized value)
        else:
            x0all[...,ib] = x0_bayer[b[0]::2,b[1]::2]

    # y1 = torch.zeros(y.shape)
    y1all = torch.zeros_like(yall).cuda() 
    # [1] start iteration for reconstruction
    xall = x0all # initialization
    ball = torch.zeros_like(x0all).cuda()
    theta_all = x0all
    x_bayer = torch.zeros_like(Phi_bayer).cuda()
    R_m, G_m, B_m = masks_CFA_Bayer(x_bayer[:,:,0].shape)
    R_m, G_m, B_m = np2tch_cuda(R_m), np2tch_cuda(G_m), np2tch_cuda(B_m)
    b = torch.zeros_like(x0all).cuda()

    psnr_all = []
    k = 0
    for idx, nsig in tqdm.tqdm(enumerate(sigma)): # iterate all noise levels
        for it in range(iter_max[idx]): 
            start_time = time.time()

            for ib in range(len(bayer)): # iterate all bayer channels
                yb = A_(theta_all[...,ib]+ball[...,ib], Phiall[...,ib])
                xall[...,ib] = theta_all[...,ib]+ball[...,ib] + _lambda*(At_((yall[...,ib]-yb)/(Phi_sumall[...,ib]+gamma), Phiall[...,ib])) # GAP

            
            end_time = time.time()
            # print('    Euclidean projection eclipsed in {:.3f}s.'.format(end_time-start_time))
            # joint Bayer multi-channel denoising
            # switch denoiser 
            if k<5: # total variation (TV) denoising
                tv_weight=0.1
                tv_iter_max=5
                multichannel=True
                
                xb_all = xall-ball
                xall_vch = xb_all.reshape([nrow//2, ncol//2, nmask*4])
                xall_vch =np2tch_cuda( denoise_tv_chambolle(cuda2np(xall_vch), tv_weight, n_iter_max=tv_iter_max, 
                                        multichannel=multichannel))
                theta_all = xall_vch.reshape([nrow//2, ncol//2, nmask, 4])
               
            elif denoiser.lower() == 'ffdnet_color_demosaic':
                x_rgb = torch.zeros([nrow, ncol, 3,nmask], dtype=torch.float32)
                xb_all = xall-ball
                for ib in range(len(bayer)): 
                    b = bayer[ib]
                    x_bayer[b[0]::2, b[1]::2] = xb_all[...,ib]
                for imask in range(nmask):
                    if demosaic_method.lower == 'bilinear':
                        x_rgb[:,:,:,imask] = demosaicing_CFA_Bayer_bilinear(x_bayer[:,:,imask])
                    # elif demosaic_method.lower == 'malvar2004':
                    #     x_rgb[:,:,:,imask] = demosaicing_CFA_Bayer_Malvar2004_tensor(x_bayer[:,:,imask],R_m, G_m, B_m)
                    else:
                        x_rgb[:,:,:,imask] = demosaicing_CFA_Bayer_Menon2007(x_bayer[:,:,imask]) #cv2.cvtColor(np.uint8(np.clip(x_bayer[:,:,imask],0,1)*255), cv2.COLOR_BAYER_RG2BGR)
                xbgr3 = ffdnet_rgb_denoise(x_rgb,yall, Phiall, nsig,model,useGPU,lr_,update_)
                #xbgr4 = np.transpose(xbgr3,(0,1,3,2))
                theta_all[...,0] = xbgr3[0::2,0::2,0,:] # R  channel (average over two)
                theta_all[...,1] = xbgr3[0::2,1::2,1,:] # G1=G2 channel (average over two)
                theta_all[...,2] = xbgr3[1::2,0::2,1,:] # G2=G1 channel (average over two)
                theta_all[...,3] = xbgr3[1::2,1::2,2,:] # B  channel (average over two)    
    
            elif denoiser.lower() == 'ffdnet_online_color': # FastDVDnet video denoising
                x_rgb = torch.zeros([nrow, ncol, 3,nmask], dtype=torch.float32).cuda()
                xb_all = xall-ball
                
                
                for ib in range(len(bayer)): 
                    b = bayer[ib]
                    x_bayer[b[0]::2, b[1]::2] = xb_all[...,ib]
                for imask in range(nmask):
                    if demosaic_method.lower == 'bilinear':
                        x_rgb[:,:,:,imask] = demosaicing_CFA_Bayer_bilinear(x_bayer[:,:,imask])
                    elif demosaic_method == 'malvar2004':
                        x_rgb[:,:,:,imask] = demosaicing_CFA_Bayer_Malvar2004_tensor(x_bayer[:,:,imask],  R_m, G_m, B_m)
                    else:
                        x_rgb[:,:,:,imask] = np2tch_cuda(demosaicing_CFA_Bayer_Menon2007(cuda2np(x_bayer[:,:,imask]))) 
                    #     x_rgb[:,:,:,imask] = demosaicing_CFA_Bayer_Menon2007(x_bayer[:,:,imask], R_m, G_m, B_m) #cv2.cvtColor(np.uint8(np.clip(x_bayer[:,:,imask],0,1)*255), cv2.COLOR_BAYER_RG2BGR)
                if k>inital_iter and k%inter_time==0:
                    update_=True
                    xbgr3, model = ffdnet_rgb_denoise_full_tensor(x_rgb,yall, Phiall, nsig,model,useGPU,lr_,update_,update_per_iter)
                #     #Load saved weights
                #     #state_temp_dict = torch.load('./packages/fastdvdnet/model_gray.pth')
                    
                #     xbgr3 = ffdnet_rgb_denoise(x_rgb,yall, Phiall, nsig,model,useGPU,lr_,update_,if_continue=if_continue,device=device)
                #     # model.load_state_dict(torch.load('model_zoo/update/ffdnet_color.pth'), strict=True)
                #     # model.load_state_dict(torch.load('model_zoo/ffdnet_color.pth'), strict=True)
                #     # device_ids = [0]
                #     #model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
                #     #model = model.cuda()
                   
                else:
                    update_=False
                    xbgr3 = ffdnet_rgb_denoise_full_tensor(x_rgb,yall, Phiall, nsig,model,useGPU,lr_,update_)
                #xbgr4 = np.transpose(xbgr3,(0,1,3,2))
                theta_all[...,0] = xbgr3[0::2,0::2,0,:] # R  channel (average over two)
                theta_all[...,1] = xbgr3[0::2,1::2,1,:] # G1=G2 channel (average over two)
                theta_all[...,2] = xbgr3[1::2,0::2,1,:] # G2=G1 channel (average over two)
                theta_all[...,3] = xbgr3[1::2,1::2,2,:] # B  channel (average over two)    
                # io.imwrite('./tem_old/'+str(k)+'_6.png',cuda2np(xbgr3[:,:,:,6]))
                # if k>inital_iter and k%inter_time==0:
                #     update_=True
                #     #Load saved weights
                #     #state_temp_dict = torch.load('./packages/fastdvdnet/model_gray.pth')
                    
                #     xbgr3 = ffdnet_rgb_denoise(x_rgb,yall, Phiall, nsig,model,useGPU,lr_,update_,if_continue=if_continue,device=device)
                #     # model.load_state_dict(torch.load('model_zoo/update/ffdnet_color.pth'), strict=True)
                #     # model.load_state_dict(torch.load('model_zoo/ffdnet_color.pth'), strict=True)
                #     # device_ids = [0]
                #     #model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
                #     #model = model.cuda()
                   
                # else:
                #     update_=False
                #     #xbgr3 = ffdnet_rgb_denoise(x_rgb, nsig,model)
                #     #xbgr3 = ffdnet_rgb_denoise(x_rgb, nsig,model)
                #     xbgr3 = ffdnet_rgb_denoise(x_rgb,yall, Phiall, nsig,model,useGPU,lr_,update_,if_continue=if_continue,device=device)
                #     #xrgb3 = fastdvdnet_denoiser(x_rgb, nsig, model, useGPU,lr_,update_,if_continue=if_continue)  
               
              
            else:
                raise ValueError('Unsupported denoiser {}!'.format(denoiser))
            
            # theta = np.clip(theta_all,0,1)
            theta_all = torch.clip(theta_all,0,1)
          
            ball = ball - (xall-theta_all) # update residual
            # [optional] calculate image quality assessment, i.e., PSNR for 
            # every five iterations
            if show_iqa and X_orig is not None:
                for ib in range(len(bayer)): 
                    b = bayer[ib]
                    x_bayer[b[0]::2, b[1]::2] = xall[...,ib]
                x_bayer_np = cuda2np(x_bayer)
                
                psnr_all.append(compare_psnr(X_orig, x_bayer_np,data_range=1.))
                if (k+1)%2 == 0:
                    if not noise_estimate and nsig is not None:
                        if nsig < 1:
                            print('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                            'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                            k+1, nsig*255, psnr_all[k]))
                            logf.write('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                            'PSNR {3:2.2f} dB. \n'.format(denoiser.upper(), 
                            k+1, nsig*255, psnr_all[k]))
                        else:
                            print('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                                k+1, nsig, psnr_all[k]))
                            logf.write('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.\n'.format(denoiser.upper(), 
                                k+1, nsig, psnr_all[k]))
                    else:
                        print('  ADMM-{0} iteration {1: 3d}, ' 
                            'PSNR {2:2.2f} dB.'.format(denoiser.upper(), 
                            k+1, psnr_all[k]))
                        logf.write('  ADMM-{0} iteration {1: 3d}, ' 
                            'PSNR {2:2.2f} dB.\n'.format(denoiser.upper(), 
                            k+1, psnr_all[k]))
            k = k+1

    for ib in range(len(bayer)): 
        b = bayer[ib]
        x_bayer[b[0]::2, b[1]::2] = xall[...,ib]
    x_bayer_np = cuda2np(x_bayer)
    psnr_ = []
    ssim_ = []
    if X_orig is not None:
        for imask in range(nmask):
            psnr_.append(compare_psnr(X_orig[:,:,imask], x_bayer_np[:,:,imask], data_range=1.))
            ssim_.append(compare_ssim(X_orig[:,:,imask], x_bayer_np[:,:,imask], data_range=1.))
    return cuda2np(xbgr3), x_bayer_np,psnr_, ssim_, psnr_all,model # numpy

def twoStageAdmm_denoise_bayer(y_bayer, Phi_bayer, _lambda=1, gamma=0.01,
                denoiser='tv', iter_max=50, noise_estimate=True, sigma=None, 
                 x0_bayer=None, 
                X_orig=None, model=None, show_iqa=True,demosaic_method = 'malvar2004',lr_=0.000001,
                 inital_iter=1, if_continue=False,inter_time=5,logf = None,useGPU=True,device=0,update_=False,update_per_iter=1):




    y_bayer = np2tch_cuda(y_bayer)
    Phi_bayer = np2tch_cuda(Phi_bayer)
    
    bayer = [[0,0], [0,1], [1,0], [1,1]] # `RGGB` Bayer pattern

    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)

    # stack the bayer channels at the last dimension [consistent to image color channels]
    (nrow, ncol, nmask) = Phi_bayer.shape
    yall = torch.zeros([nrow//2, ncol//2, 4], dtype=torch.float32).cuda()
    Phiall = torch.zeros([nrow//2, ncol//2, nmask, 4], dtype=torch.float32).cuda()
    Phi_sumall = torch.zeros([nrow//2, ncol//2, 4], dtype=torch.float32).cuda()
    x0all = torch.zeros([nrow//2, ncol//2, nmask, 4], dtype=torch.float32).cuda()
    
    # iterative solve for each Bayer channel
    for ib in range(len(bayer)): 
        b = bayer[ib]
        yall[...,ib] = y_bayer[b[0]::2, b[1]::2]
        Phiall[...,ib] =  Phi_bayer[b[0]::2, b[1]::2]

        Phib = Phiall[...,ib]
        Phib_sum = torch.sum(Phib, dim=2)
        Phib_sum[Phib_sum==0] = 1

        Phi_sumall[...,ib] = Phib_sum

        # [0] initialization
        if x0_bayer is None:
            # x0 = At(y, Phi) # default start point (initialized value)
            x0all[...,ib] = At_(yall[...,ib], Phiall[...,ib]) # default start point (initialized value)
        else:
            x0all[...,ib] = x0_bayer[b[0]::2,b[1]::2]

    # y1 = torch.zeros(y.shape)
    y1all = torch.zeros_like(yall).cuda() 
    # [1] start iteration for reconstruction
    xall = x0all # initialization
    ball = torch.zeros_like(x0all).cuda()
    theta_all = x0all
    x_bayer = torch.zeros_like(Phi_bayer).cuda()
    R_m, G_m, B_m = masks_CFA_Bayer_tensor(x_bayer[:,:,0].shape)
    # R_m, G_m, B_m = np2tch_cuda(R_m), np2tch_cuda(G_m), np2tch_cuda(B_m)
    b = torch.zeros_like(x0all).cuda()
    w = torch.zeros([nrow, ncol, 3,nmask], dtype=torch.float32).cuda()
    psnr_all = []
    k = 0
    rou = 1
    tau = 20
    for idx, nsig in tqdm.tqdm(enumerate(sigma)): # iterate all noise levels
        for it in range(iter_max[idx]): 
            start_time = time.time()
          
            # for ib in range(len(bayer)): # iterate all bayer channels
                
            #     yb = A_(theta_all[...,ib]-ball[...,ib], Phiall[...,ib])
            #     xall[...,ib] = theta_all[...,ib]-ball[...,ib] + _lambda*(At_((yall[...,ib]-yb)/(Phi_sumall[...,ib]+rou), Phiall[...,ib])) 

            
            for ib in range(len(bayer)): # iterate all bayer channels
         
                # matrix
                p = theta_all[...,ib]-(1/rou)*ball[...,ib]

                yb = A_(p,Phiall[...,ib])      # M/2 N/2 B i
        
                # trans_cat_b = At_((yall[...,ib] - yb) / (rou + Phi_sumall[...,ib]),Phiall[...,ib]) # M/2 N/2 B i

                trans_cat_b = (yall[...,ib]-yb)/(rou+Phi_sumall[...,ib]) # M/2 N/2 i
                trans_cat_b = Phiall[...,ib]*torch.repeat_interleave(trans_cat_b.unsqueeze(2),nmask,dim=2) # M/2 N/2 B i
            
                xall[...,ib] = p+trans_cat_b# M/2 N/2 B i

            end_time = time.time()
            # print('    Euclidean projection eclipsed in {:.3f}s.'.format(end_time-start_time))
            # joint Bayer multi-channel denoising
            # switch denoiser 
            if k<5: # total variation (TV) denoising
                TV = True
                tv_weight=0.1
                tv_iter_max=5
                multichannel=True
                
                xb_all = xall + (1/rou)*ball
                xall_vch = xb_all.reshape([nrow//2, ncol//2, nmask*4])
                xall_vch =np2tch_cuda( denoise_tv_chambolle(cuda2np(xall_vch), tv_weight, n_iter_max=tv_iter_max, 
                                        multichannel=multichannel))
                theta_all = xall_vch.reshape([nrow//2, ncol//2, nmask, 4])
               
           
    
            elif denoiser.lower() == 'ffdnet_online_color': # FastDVDnet video denoising
                TV = False
                x_rgb = torch.zeros([nrow, ncol, 3,nmask], dtype=torch.float32).cuda()
             
                
                xb_all = xall + (1/rou)*ball
                for ib in range(len(bayer)): 
                    b = bayer[ib]
                    x_bayer[b[0]::2, b[1]::2] = xb_all[...,ib]
            
                for imask in range(nmask):
                    if demosaic_method.lower == 'bilinear':
                        x_rgb[:,:,:,imask] = demosaicing_CFA_Bayer_bilinear(x_bayer[:,:,imask])
                    elif demosaic_method == 'malvar2004':
                        x_rgb[:,:,:,imask] = demosaicing_CFA_Bayer_Malvar2004_tensor(x_bayer[:,:,imask],  R_m, G_m, B_m)
                    # else:
                    #     x_rgb[:,:,:,imask] = np2tch_cuda(demosaicing_CFA_Bayer_Menon2007(cuda2np(x_bayer[:,:,imask]))) 
                    #     x_rgb[:,:,:,imask] = demosaicing_CFA_Bayer_Menon2007(x_bayer[:,:,imask], R_m, G_m, B_m) #cv2.cvtColor(np.uint8(np.clip(x_bayer[:,:,imask],0,1)*255), cv2.COLOR_BAYER_RG2BGR)
                
                update_=False
                x_rgb_w = x_rgb-(1/tau)*w
                xbgr3 = ffdnet_rgb_denoise_full_tensor(x_rgb_w,yall, Phiall, nsig,model,useGPU,lr_,update_)
                #xbgr4 = np.transpose(xbgr3,(0,1,3,2))
                theta_all[...,0] = xbgr3[0::2,0::2,0,:] # R  channel (average over two)
                theta_all[...,1] = xbgr3[0::2,1::2,1,:] # G1=G2 channel (average over two)
                theta_all[...,2] = xbgr3[1::2,0::2,1,:] # G2=G1 channel (average over two)
                theta_all[...,3] = xbgr3[1::2,1::2,2,:] # B  channel (average over two)    
                # io.imwrite('./tem_old/'+str(k)+'_6.png',cuda2np(xbgr3[:,:,:,6]))
               
              
            else:
                raise ValueError('Unsupported denoiser {}!'.format(denoiser))
            
            # theta = np.clip(theta_all,0,1)
            theta_all = torch.clip(theta_all,0,1)
          
            ball = ball + (xall - theta_all) # update residual
            # [optional] calculate image quality assessment, i.e., PSNR for 
            # every five iterations
            if not TV:
                w = w + (x_rgb - xbgr3)
          
             # update residual
            # [optional] calculate image quality assessment, i.e., PSNR for 
            # every five iterations
         
            if show_iqa and X_orig is not None:
                for ib in range(len(bayer)): 
                    b = bayer[ib]
                    x_bayer[b[0]::2, b[1]::2] = xall[...,ib]
                x_bayer_np = cuda2np(x_bayer)
                
                psnr_all.append(compare_psnr(X_orig, x_bayer_np,data_range=1.))
                if (k+1)%2 == 0:
                    if not noise_estimate and nsig is not None:
                        if nsig < 1:
                            print('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                            'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                            k+1, nsig*255, psnr_all[k]))
                            logf.write('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                            'PSNR {3:2.2f} dB. \n'.format(denoiser.upper(), 
                            k+1, nsig*255, psnr_all[k]))
                        else:
                            print('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                                k+1, nsig, psnr_all[k]))
                            logf.write('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.\n'.format(denoiser.upper(), 
                                k+1, nsig, psnr_all[k]))
                    else:
                        print('  ADMM-{0} iteration {1: 3d}, ' 
                            'PSNR {2:2.2f} dB.'.format(denoiser.upper(), 
                            k+1, psnr_all[k]))
                        logf.write('  ADMM-{0} iteration {1: 3d}, ' 
                            'PSNR {2:2.2f} dB.\n'.format(denoiser.upper(), 
                            k+1, psnr_all[k]))
            k = k+1

    for ib in range(len(bayer)): 
        b = bayer[ib]
        x_bayer[b[0]::2, b[1]::2] = xall[...,ib]
    x_bayer_np = cuda2np(x_bayer)
    psnr_ = []
    ssim_ = []
    if X_orig is not None:
        for imask in range(nmask):
            psnr_.append(compare_psnr(X_orig[:,:,imask], x_bayer_np[:,:,imask], data_range=1.))
            ssim_.append(compare_ssim(X_orig[:,:,imask], x_bayer_np[:,:,imask], data_range=1.))
    return cuda2np(xbgr3), x_bayer_np,psnr_, ssim_, psnr_all,model # numpy

