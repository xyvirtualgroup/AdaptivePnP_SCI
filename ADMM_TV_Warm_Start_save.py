


import os
import time
import math
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from statistics import mean
import torch


from dvp_linear_inv_2_stage_ADMM_tensor_online import admm_denoise_bayer_demosaic_pre as reconstruct # one stage
from utilspy import *


worker_init_fn(0)
datasetdir = './dataset/cacti/mid_scale' # color middle scale dataset 512*512 px

t = time.time()
str1='New1/'+str(int(t))
resultsdir1 = "results/"+str1+'/'  #
mkdir(resultsdir1)
resultsdir = "results/"+str1
f = open(resultsdir+'/log.txt','a')

f.write('cacti midscale bayer: \n')


average_psnr = []
average_ssim = []
average_time = []

sigma = [0 / 255]  # pre-set noise standard deviation
iter_max = [40]  # maximum number of iterations

for ncount in range(0,6):

    average_psnr_i = []
    average_ssim_i = []
    average_time_i = []
    if ncount == 0:
        datname = 'Beauty_bayer'
    elif ncount == 1:
        datname = 'Bosphorus_bayer'
    elif ncount == 2:
        datname = 'Jockey_bayer'
    elif ncount == 3:
        datname = 'Runner_bayer'
    elif ncount == 4:
        datname = 'ShakeNDry_bayer'
    elif ncount == 5:
        datname = 'Traffic_bayer'

    print('next')
    print('===========BEGIN============'+ datname +'=============BEGIN==============')

    f.write(datname + ':\n')

    matfile = datasetdir + '/' + datname + '.mat' # path of the .mat data file


    # In[3]:


    # [1] load data
    with h5py.File(matfile, 'r') as file: # for '-v7.3' .mat file (MATLAB)
        # print(list(file.keys()))
        meas_bayer = np.array(file['meas_bayer'])
        mask_bayer = np.array(file['mask_bayer'])
        orig_bayer = np.array(file['orig_bayer'])
        orig_real = np.array(file['orig'])
    #==============================================================================
    # file = scipy.io.loadmat(matfile) # for '-v7.2' and below .mat file (MATLAB)
    # X = list(file[varname])
    #file = sio.loadmat(matfile)
    #meas_bayer = np.array(file['meas'])
    #mask_bayer = np.array(file['mask'])
    #orig_bayer = np.array(file['orig_bayer'])

    #==============================================================================

    mask_bayer = np.float32(mask_bayer).transpose((2,1,0))
    if len(meas_bayer.shape) < 3:
        meas_bayer = np.float32(meas_bayer).transpose((1,0))
    else:
        meas_bayer = np.float32(meas_bayer).transpose((2,1,0))
    orig_bayer = np.float32(orig_bayer).transpose((2,1,0))
    # print(mask_bayer.shape, meas_bayer.shape, orig_bayer.shape)
    (nrows, ncols,nmea) = meas_bayer.shape
    (nrows, ncols,nmask) = mask_bayer.shape


    v_Admm_tv_gray_bayer = np.zeros([nrows, ncols, nmask*nmea], dtype=np.float32)
    psnr_tv_gray = np.zeros([nmask*nmea,1], dtype=np.float32)
    ssim_tv_gray = np.zeros([nmask*nmea,1], dtype=np.float32)


    tv_denoiser = np.zeros([nmea,ncols,nrows,3,nmask], dtype=np.float32)

    _lambda = 1 # regularization factor

    denoiser = 'tv'
    noise_estimate = False 

    useGPU = True # use GPU
    if_continue=False
    f.write('tv_denoiser start...\n')      

    MAXB = 255.

    for iframe in range(nmea):
        
        f.write('Measurement Frame {}.\n'.format(iframe))
        if len(meas_bayer.shape) >= 3:
            meas_bayer_t = np.squeeze(meas_bayer[:,:,iframe])/MAXB
        else:
            meas_bayer_t = meas_bayer/MAXB
        orig_bayer_t = orig_bayer[:,:,iframe*nmask:(iframe+1)*nmask]/MAXB
            
        f.write('tv_denoiser start.\n')

        ## [2.2] ADMM-tv_denoiser
        
        begin_time = time.time()

        
        v_Admm_tv_denoiser_gray_bayer_t,\
        psnr_Admm_tv_denoiser_gray_t,ssim_Admm_tv_denoiser_gray_t,psnrall_tv_denoiser_gray_t=   \
            reconstruct(meas_bayer_t, mask_bayer, _lambda,
                                            0.01, 'tv', iter_max, noise_estimate, sigma,x0_bayer=None,
                                            X_orig=orig_bayer_t, model=None,show_iqa=True,
                                            logf = f)                       
                                            
        end_time = time.time()
        t_Admm_tv_denoiser_gray = end_time - begin_time
        print('ADMM-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
            denoiser.upper(), mean(psnr_Admm_tv_denoiser_gray_t), mean(ssim_Admm_tv_denoiser_gray_t), t_Admm_tv_denoiser_gray))
        f.write('ADMM-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds. \n'.format(
            denoiser.upper(), mean(psnr_Admm_tv_denoiser_gray_t), mean(ssim_Admm_tv_denoiser_gray_t), t_Admm_tv_denoiser_gray))
        v_Admm_tv_gray_bayer[:,:,iframe*nmask:(iframe+1)*nmask] =  v_Admm_tv_denoiser_gray_bayer_t
        psnr_tv_gray[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(psnr_Admm_tv_denoiser_gray_t,(nmask,1))
        ssim_tv_gray[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(ssim_Admm_tv_denoiser_gray_t,(nmask,1))
        # In[12]:

        

        average_psnr_i.append(mean(psnr_Admm_tv_denoiser_gray_t))  # mean of 8 frames
        average_ssim_i.append(mean(ssim_Admm_tv_denoiser_gray_t))
        # average_time_i.append(t_Admm_tv_denoiser_color)



        
    print('===========END============'+ datname +'=============END==============')
    sigma_out = [x*255 for x in sigma]
    print(mean(average_psnr_i),end='    ')
    print(mean(average_ssim_i))

    
    print(iter_max,end=' ')

    average_psnr.append(mean(average_psnr_i))
    average_ssim.append(mean(average_ssim_i))
    # average_time.append(mean(average_time_i))

    # In[12]:
    savedmatdir = './results' + '/savedmat/'
    if not os.path.exists(savedmatdir):
        os.makedirs(savedmatdir)

    sio.savemat('{}_Admm_{}_{}{:d}.mat'.format(savedmatdir,denoiser.lower(),datname,nmask),
                {'v_Admm_tv_denoise':v_Admm_tv_gray_bayer,
                'psnr_Admm_tv_denoise':psnr_tv_gray,
                'ssim_Admm_tv_denoise':ssim_tv_gray,
                })
    print('{}_Admm_{}_{}{:d}.mat -- saved '.format(savedmatdir,denoiser.lower(),datname,nmask))


print('all= ')
print(mean(average_psnr))
print(mean(average_ssim))
# print(mean(average_time))
f.close()
            