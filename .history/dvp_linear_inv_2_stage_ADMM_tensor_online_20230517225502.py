from re import T
from six import u
from packages.colour_demosaicing.bayer.demosaicing.malvar2004 import demosaicing_CFA_Bayer_Malvar2004_tensor
import time
import math
import skimage
import numpy as np
import torch
# import cv2
import imageio as io
from statistics import mean
import matplotlib.pyplot as plt
from skimage.restoration import (denoise_tv_chambolle)
# from packages.fastdvdnet.utils import close_logger

from packages.ffdnet.test_ffdnet_ipol import ffdnet_rgb_denoise_full_tensor #, ffdnet_vdenoiser,ffdnet_rgb_denoise_full_tensor_v2
from packages.ffdnet.test_ffdnet_ipol import ffdnet_rgb_denoise
from packages.fastdvdnet.test_fastdvdnet import fastdvdnet_denoiser, fastdvdnet_denoiser_full_tensor, fastdvdnet_denoiser_full_tensor_v2
from packages.colour_demosaicing.bayer import demosaicing_CFA_Bayer_bilinear

from utilspy import (A_, At_, psnr)
import tqdm
if skimage.__version__ < '0.18':
    from skimage.measure import (compare_psnr, compare_ssim)
else: # skimage.measure deprecated in version 0.18 ( -> skimage.metrics )
    from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics._structural_similarity import structural_similarity as compare_ssim

from packages.colour_demosaicing.bayer import masks_CFA_Bayer
from packages.DDnet.DDnet_test import test_ddnet

import tqdm
from utils.utils_image import *
# from tensorboardX import SummaryWriter


def twoStageAdmm_denoise_bayer(y_bayer, Phi_bayer, _lambda=1, gamma=0.01,
                denoiser='tv', iter_max=50, noise_estimate=True, sigma=None, 
                 x0_bayer=None, 
                X_orig=None, model_denoise=None,model_demosaic=None, show_iqa=True,demosaic_method = 'malvar2004',lr_=0.000001,
                 inital_iter=1,interval_iter=5,logf = None,useGPU=True,update_=False,update_per_iter=1,close_form_demosaic=False,
                 large=False,update_times=-1,args=None):


    y_bayer = np2tch_cuda(y_bayer)
    Phi_bayer = np2tch_cuda(Phi_bayer)
    # writer = SummaryWriter('runs/'+denoiser)
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
    update_i = 0
    update_dmi = 0
    if denoiser=='tv':
        alpha = 0.01 # previous ADMM set for ADMM-TV
    else: 
        alpha = 1

    if denoiser=='fastdvd_color':
        rou = 0.55
    else: 
        rou = 1
    tau = 100
    
    if close_form_demosaic:
        tau = 10
        rou = 0.55
      
        bayer_mask = gen_bayer_mask(R_m, G_m, B_m) # M N 3
        bayer_mask_sum_plus = rou*bayer_mask+tau
        inv_3ch = torch.repeat_interleave(bayer_mask_sum_plus.unsqueeze(3),nmask,dim=3)
       

    for idx, nsig in tqdm.tqdm(enumerate(sigma)): # iterate all noise levels
        for it in range(iter_max[idx]): 
            start_time = time.time()
 
    

            
            for ib in range(len(bayer)): # iterate all bayer channels
         
                # matrix
                p = theta_all[...,ib]-(1/rou)*ball[...,ib]

                yb = A_(p,Phiall[...,ib])      # M/2 N/2 B i
        
                # trans_cat_b = At_((yall[...,ib] - yb) / (rou + Phi_sumall[...,ib]),Phiall[...,ib]) # M/2 N/2 B i

                trans_cat_b = (yall[...,ib]-yb)/(alpha*rou+Phi_sumall[...,ib]) # M/2 N/2 i
                trans_cat_b = Phiall[...,ib]*torch.repeat_interleave(trans_cat_b.unsqueeze(2),nmask,dim=2) # M/2 N/2 B i

                xall[...,ib] = p+trans_cat_b# M/2 N/2 B i


            end_time = time.time()

            # joint Bayer multi-channel denoising
            # switch denoiser 
            if k<0 or denoiser=='tv': # total variation (TV) denoising
                TV = True
                tv_weight=0.1
                tv_iter_max=5
                multichannel=True
               
                xb_all = xall + (1/rou)*ball
    
                

                xall_vch = xb_all.reshape([nrow//2, ncol//2, nmask*4])
                xall_vch =np2tch_cuda( denoise_tv_chambolle(cuda2np(xall_vch), tv_weight, n_iter_max=tv_iter_max, 
                                        multichannel=multichannel))
                theta_all = xall_vch.reshape([nrow//2, ncol//2, nmask, 4])
                # writer.add_image('7_dn_tv',theta_all[:,:,7,0],k,dataformats='HW')
            

            elif denoiser.lower() == 'ffdnet_color': 
                TV = False
                x_rgb = torch.zeros([nrow, ncol, 3,nmask], dtype=torch.float32).cuda()
             

                xb_all = xall + (1/rou)*ball
                for ib in range(len(bayer)): 
                    b = bayer[ib]
                    x_bayer[b[0]::2, b[1]::2] = xb_all[...,ib]
               

                if close_form_demosaic and k>0:
                    xall_3ch = fourCh2ThreeCh(xall)
                    ball_3ch = fourCh2ThreeCh(ball)

                    rou_u_v_w = rou*xall_3ch+ball_3ch+tau*xbgr3+w
        
                    x_rgb = rou_u_v_w/inv_3ch
                    x_rgb = x_rgb.clip(0,1)


                elif model_demosaic==None:
                    for imask in range(nmask):
                        if demosaic_method.lower == 'bilinear':
                            x_rgb[:,:,:,imask] = demosaicing_CFA_Bayer_bilinear(x_bayer[:,:,imask])
                        elif demosaic_method == 'malvar2004':
                            # x_rgb[:,:,:,imask] = np2tch_cuda(demosaicing_CFA_Bayer_Malvar2004(cuda2np(x_bayer[:,:,imask]))) 
                            x_rgb[:,:,:,imask] = demosaicing_CFA_Bayer_Malvar2004_tensor(x_bayer[:,:,imask],  R_m, G_m, B_m)
                else: # deep demosaicking
                        x_bayer_3ch = oneCh2ThreeCh(x_bayer)
                        x_rgb = test_ddnet(x_bayer_3ch,yall, Phiall,model_demosaic)    
                       


                x_rgb_w = x_rgb -(1/tau)*w

                if update_ and k>inital_iter and k%interval_iter==0:
                    xbgr3, model_denoise = ffdnet_rgb_denoise_full_tensor(x_rgb_w,yall, Phiall, nsig,model_denoise,useGPU,lr_,update_,update_per_iter)
                else:   
                    xbgr3 = ffdnet_rgb_denoise_full_tensor(x_rgb_w,yall, Phiall, nsig,model_denoise,useGPU,lr_)


                theta_all[...,0] = xbgr3[0::2,0::2,0,:] # R  channel (average over two)
                theta_all[...,1] = xbgr3[0::2,1::2,1,:] # G1=G2 channel (average over two)
                theta_all[...,2] = xbgr3[1::2,0::2,1,:] # G2=G1 channel (average over two)
                theta_all[...,3] = xbgr3[1::2,1::2,2,:] # B  channel (average over two)    
                # writer.add_image('7_dn',xbgr3[:,:,:,7],k,dataformats='HWC')

               
               
            elif denoiser.lower() == 'fastdvd_color':  
                TV = False
                x_rgb = torch.zeros([nrow, ncol, 3,nmask], dtype=torch.float32).cuda()
             
                
                xb_all = xall + (1/rou)*ball
                for ib in range(len(bayer)): 
                    b = bayer[ib]
                    x_bayer[b[0]::2, b[1]::2] = xb_all[...,ib]
            
                if close_form_demosaic and k>0:
                    xall_3ch = fourCh2ThreeCh(xall)
                    ball_3ch = fourCh2ThreeCh(ball)

                    rou_u_v_w = rou*xall_3ch+ball_3ch+tau*xbgr3+w
        
                    x_rgb = rou_u_v_w/inv_3ch
                elif model_demosaic==None:
                    for imask in range(nmask):
                        if demosaic_method.lower == 'bilinear':
                            x_rgb[:,:,:,imask] = demosaicing_CFA_Bayer_bilinear(x_bayer[:,:,imask])
                        elif demosaic_method == 'malvar2004':
                            # x_rgb[:,:,:,imask] = np2tch_cuda(demosaicing_CFA_Bayer_Malvar2004(cuda2np(x_bayer[:,:,imask]))) 
                            x_rgb[:,:,:,imask] = demosaicing_CFA_Bayer_Malvar2004_tensor(x_bayer[:,:,imask],  R_m, G_m, B_m)

                
                
                
                else: 
                    x_bayer_3ch = oneCh2ThreeCh(x_bayer)
                    x_rgb = test_ddnet(x_bayer_3ch,yall, Phiall,model_demosaic)     
               
                x_rgb_w = x_rgb-(1/tau)*w
                if update_ and k>inital_iter and k%interval_iter==0 and (update_i<update_times or update_times<0):

                    torch.cuda.empty_cache()
                    xbgr3,model_denoise = fastdvdnet_denoiser_full_tensor_v2(x_rgb_w,nsig,yall, Phiall,model_denoise,useGPU,lr_,update_,update_per_iter,update_times=update_times)
                    update_i+=1
                else:
                    xbgr3 = fastdvdnet_denoiser_full_tensor_v2(x_rgb_w,nsig,yall, Phiall,model_denoise,useGPU,lr_)

                # xbgr3 = xbgr3.permute()
                theta_all[...,0] = xbgr3[0::2,0::2,0,:] # R  channel (average over two)
                theta_all[...,1] = xbgr3[0::2,1::2,1,:] # G1=G2 channel (average over two)
                theta_all[...,2] = xbgr3[1::2,0::2,1,:] # G2=G1 channel (average over two)
                theta_all[...,3] = xbgr3[1::2,1::2,2,:] # B  channel (average over two)    
                

            else:
                raise ValueError('Unsupported denoiser {}!'.format(denoiser))
          
            theta_all = torch.clip(theta_all,0,1)
          
            ball = ball + (xall - theta_all) # update residual

            
            if not TV:
                w = w + (x_rgb - xbgr3) # update residual
        
         
            if show_iqa and X_orig is not None:
                for ib in range(len(bayer)): 
                    b = bayer[ib]
                    x_bayer[b[0]::2, b[1]::2] = theta_all[...,ib] # theta_all xall
                x_bayer_np = cuda2np(x_bayer)
                psnr = compare_psnr(X_orig, x_bayer_np,data_range=1.)
                psnr_all.append(psnr)
                # writer.add_scalar('psnr',psnr,k)
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

            if (X_orig is None) and ((k+1)%2==0):
                logf.write('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}/255 \n'.format(denoiser.upper(), 
                            k+1, nsig*255))


    for ib in range(len(bayer)): 
        b = bayer[ib]
        x_bayer[b[0]::2, b[1]::2] = theta_all[...,ib]
    x_bayer_np = cuda2np(x_bayer)
    psnr_ = []
    ssim_ = []
    if X_orig is not None:
        for imask in range(nmask):
            psnr_.append(compare_psnr(X_orig[:,:,imask], x_bayer_np[:,:,imask], data_range=1.))
            ssim_.append(compare_ssim(X_orig[:,:,imask], x_bayer_np[:,:,imask], data_range=1.))
    if denoiser=='tv':
        return x_bayer_np,psnr_, ssim_, psnr_all
    return cuda2np(xbgr3), x_bayer_np,psnr_, ssim_, psnr_all,model_denoise,model_demosaic # numpy

