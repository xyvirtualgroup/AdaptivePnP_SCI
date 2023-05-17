

import os

root_path = os.path.dirname(os.path.abspath(__file__))
print(root_path)
os.chdir( root_path )
import time
import math
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from statistics import mean
import torch

from dvp_linear_inv_2_stage_ADMM_tensor_online import twoStageAdmm_denoise_bayer as reconstruct,np2tch_cuda
from utilspy import *
from packages.fastdvdnet.models import FastDVDnet

os.environ["CUDA_VISIBLE_DEVICES"] = "0,9"

worker_init_fn(0)


datasetdir = './dataset/cacti/mid_scale' # color middle scale dataset 512*512 px
update = True # use 'online' denoiser
reuse_model = True # use refined model for later frames
denoiser = 'fastdvd_color' # video denosing network

deep_demosaicking = True
# demosaic_update = False

# pre-load the model for fastdvdnet image denoising
NUM_IN_FR_EXT = 5 # temporal size of patch
    #sf = 1                    # unused for denoising

model_denoise_path ='./packages/fastdvdnet/model.pth'


from models.network_demosaicking_dual_path0913 import DDnet
model_demosaic_path = '/home/wuzongliang/py/Two-stage-online-PnP/DDN_plus0905/DDn_plus_baseline_0913_32/joint_logs/ckpt_old.pth'


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


fix = False
for ncount in range(1,2):

    average_psnr_i = []
    average_ssim_i = []
    average_time_i = []
    if ncount == 0:
        datname = 'Beauty_bayer'
        sigma = [8/255]  # pre-set noise standard deviation
        iter_max = [18]  # maximum number of iterations
        

        lr = 2e-6 # 2e-7
        update_per_iter = 2 # best 2 for ddn         # 1  
        interval_iter = 9 # best 9 for ddn       
        update_times = 1 # -1: default=iter_max/interval_iter, 1: only once, ...

        if deep_demosaicking:
            sigma = [12/255,6/255]
            iter_max = [21,2]  
            # sigma = [12/255]
            # iter_max = [20]  

            interval_iter = 22
            
 
            
    elif ncount == 1:
        datname = 'Bosphorus_bayer'
        sigma = [12/255,6/255]    #    pre-set noise standard deviation
        iter_max = [24,12]  # maximum number of iterations\


        lr = 2e-7
        update_per_iter = 2                  
        interval_iter = 12 
        update_times = -1
        if deep_demosaicking:
            sigma = [8/255,6/255]  
            iter_max = [24,12]    
            interval_iter = 25
                

    elif ncount == 2:
        datname = 'Jockey_bayer'
        sigma = [12/255]  #    pre-set noise standard deviation
        iter_max = [24]  # maximum number of iterations

        lr = 2e-7
        update_per_iter = 2                 
        interval_iter = 12
        update_times = -1
        if deep_demosaicking:
            sigma = [12/255,6/255]  
            iter_max = [24,6]  
            interval_iter = 25
                         


    elif ncount == 3:
        datname = 'Runner_bayer'
        sigma = [14/255]  #    pre-set noise standard deviation
        iter_max = [24]  # maximum number of iterations

        lr = 2e-7
        update_per_iter = 2                  
        interval_iter = 12
        update_times = -1 
        if deep_demosaicking:
            sigma = [12/255,6/255]  
            iter_max = [40,15]  

            interval_iter = 41
            
        


    elif ncount == 4:
        datname = 'ShakeNDry_bayer'
        sigma = [10/255]  #    pre-set noise standard deviation
        iter_max = [15]  # maximum number of iterations

        lr = 2e-7
        update_per_iter = 1                 
        interval_iter = 7
        update_times = -1        
        if deep_demosaicking:
            sigma = [12/255,6/255]  
            iter_max = [14,4]  
            interval_iter = 15
            

    elif ncount == 5:
        datname = 'Traffic_bayer'
        sigma = [30/255]  #    pre-set noise standard deviation
        iter_max = [22]  # maximum number of iterations

        lr = 2e-7
        update_per_iter = 2                  
        interval_iter = 11 
        update_times = -1  

        if deep_demosaicking:
            sigma = [25/255,12/255,6/255]  
            iter_max = [36,6,2]  
            interval_iter = 43
            

                
    # iter_max[1] = iter_max[1]+8
        
    print('next')
    print('===========BEGIN============'+ datname +'=============BEGIN==============')
    
    f.write(datname + ':\n')

    matfile = datasetdir + '/' + datname + '.mat' # path of the .mat data file



    # [1] load data
    with h5py.File(matfile, 'r') as file: # for '-v7.3' .mat file (MATLAB)
        # print(list(file.keys()))
        meas_bayer = np.array(file['meas_bayer'])
        mask_bayer = np.array(file['mask_bayer'])
        orig_bayer = np.array(file['orig_bayer'])
        orig_real = np.array(file['orig'])
    
    warm_start_file = './results/savedmat/_Admm_tv_'+datname+'8.mat'


    file = sio.loadmat(warm_start_file) # for '-v7.3' .mat file (MATLAB)
        # print(list(file.keys()))
    recon_tv = np.array(file['v_Admm_tv_denoise'])

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


    v_twoStageAdmm_fastdvd_gray_bayer = np.zeros([nrows, ncols, nmask*nmea], dtype=np.float32)
    psnr_fastdvd_gray = np.zeros([nmask*nmea,1], dtype=np.float32)
    ssim_fastdvd_gray = np.zeros([nmask*nmea,1], dtype=np.float32)


    onlinefastdvdnet = np.zeros([nmea,ncols,nrows,3,nmask], dtype=np.float32)

    _lambda = 1 # regularization factor
    accelerate = True # enable accelerated version of _twoStageAdmm_
    
    noise_estimate = False # disable noise estimation for _twoStageAdmm_


    useGPU = True # use GPU


    
        
    model_denoise = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)

    # Load saved weights
    state_temp_dict = torch.load(model_denoise_path)

    if useGPU:
        device_ids = [0,1]
        model_denoise = torch.nn.DataParallel(model_denoise, device_ids=device_ids).cuda()
        # model = model.cuda()
    # else:
        # CPU mode: remove the DataParallel wrapper
        # state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)

        model_denoise.load_state_dict(state_temp_dict)


    
    

    model_denoise.eval()

    if deep_demosaicking:
        # from torch import nn
        model_demosaic = DDnet()
        model_demosaic = torch.nn.DataParallel(model_demosaic).cuda()
        state_dict = torch.load(model_demosaic_path)['state_dict']
        model_demosaic.load_state_dict(state_dict, strict=True)
        model_demosaic.eval()
    else: 
        model_demosaic=None

    MAXB = 255.

    for iframe in range(nmea):
        
        f.write('Measurement Frame {}.\n'.format(iframe))
        if len(meas_bayer.shape) >= 3:
            meas_bayer_t = np.squeeze(meas_bayer[:,:,iframe])/MAXB
        else:
            meas_bayer_t = meas_bayer/MAXB
        orig_bayer_t = orig_bayer[:,:,iframe*nmask:(iframe+1)*nmask]/MAXB
        
        

        f.write('fastdvdnet-rgb-demosaic start.\n')

        ## [2.2] ADMM-fastdvdnet
        
        begin_time = time.time()


        v_tv = recon_tv[:,:,iframe*nmask:(iframe+1)*nmask]
        # v_tv,_,_,_ =   \
        #     reconstruct(meas_bayer_t, mask_bayer, _lambda,
        #                                   0.01, 'tv', 40, noise_estimate, 0,x0_bayer=None,
        #                                   X_orig=orig_bayer_t, 
        #                                   model_denoise=None,model_demosaic=None,
        #                                   show_iqa=True,demosaic_method ='malvar2004',
        #                                   lr_=lr,
        #                                   interval_iter=interval_iter,logf = f,update_=update,update_per_iter=update_per_iter)   

        onlinefastdvdnet[iframe,:,:,:,:],v_twoStageAdmm_fastdvdnet_gray_bayer_t,\
            psnr_twoStageAdmm_fastdvdnet_gray_t,ssim_twoStageAdmm_fastdvdnet_gray_t,psnrall_fastdvdnet_gray_t,refined_model,dm_refined_model =   \
            reconstruct(meas_bayer_t, mask_bayer, _lambda,
                                            0.01, denoiser, iter_max, noise_estimate, sigma,x0_bayer= np2tch_cuda( v_tv),
                                            X_orig=orig_bayer_t, 
                                            model_denoise=model_denoise,
                                            model_demosaic=model_demosaic,
                                            show_iqa=True,demosaic_method ='malvar2004',
                                            lr_=lr,
                                            interval_iter=interval_iter,logf = f,update_=update,update_per_iter=update_per_iter,update_times=update_times)   
        if reuse_model and update:
            model_denoise = refined_model
            model_demosaic = dm_refined_model
        else: 
            model_denoise.load_state_dict(torch.load(model_denoise_path), strict=True)
            model_denoise.eval()
            model_denoise = model_denoise.cuda()    

            model_demosaic.load_state_dict(torch.load(model_demosaic_path)['state_dict'], strict=True)
            model_demosaic.eval()
            model_demosaic = model_demosaic.cuda()            
                                            
        end_time = time.time()
        t_twoStageAdmm_fastdvdnet_gray = end_time - begin_time
        print('ADMM-{}--{}-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
            denoiser.upper(),datname,iframe ,mean(psnr_twoStageAdmm_fastdvdnet_gray_t), mean(ssim_twoStageAdmm_fastdvdnet_gray_t), t_twoStageAdmm_fastdvdnet_gray))
        f.write('ADMM-{}--{}-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds. \n'.format(
            denoiser.upper(),datname,iframe , mean(psnr_twoStageAdmm_fastdvdnet_gray_t), mean(ssim_twoStageAdmm_fastdvdnet_gray_t), t_twoStageAdmm_fastdvdnet_gray))
        v_twoStageAdmm_fastdvd_gray_bayer[:,:,iframe*nmask:(iframe+1)*nmask] =  v_twoStageAdmm_fastdvdnet_gray_bayer_t
        psnr_fastdvd_gray[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(psnr_twoStageAdmm_fastdvdnet_gray_t,(nmask,1))
        ssim_fastdvd_gray[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(ssim_twoStageAdmm_fastdvdnet_gray_t,(nmask,1))

        average_psnr_i.append(mean(psnr_twoStageAdmm_fastdvdnet_gray_t))  # mean of 8 frames
        average_ssim_i.append(mean(ssim_twoStageAdmm_fastdvdnet_gray_t))
        # average_time_i.append(t_twoStageAdmm_fastdvdnet_color)


    print('===========END============'+ datname +'=============END==============')
    sigma_out = [x*255 for x in sigma]
    print(round(mean(average_psnr_i),2),end=', ')
    print(round(mean(average_ssim_i),4))
    print('Params:')
    print(lr,end=' ')

    print(iter_max,end=' ')
    print(sigma_out,end=' ')

    print('interval_iter='+str(interval_iter),end=' ')
    # print('inital_iter='+str(inital_iter),end=' ')
    print('update_per_iter='+str(update_per_iter))

    

    average_psnr.append(mean(average_psnr_i))
    average_ssim.append(mean(average_ssim_i))
    # average_time.append(mean(average_time_i))

    savedmatdir = resultsdir + '/savedmat/'
    if not os.path.exists(savedmatdir):
        os.makedirs(savedmatdir)

    sio.savemat('{}twoStageAdmm_{}_{}{:d}_sigma{:d}_all7_log.mat'.format(savedmatdir,denoiser.lower(),datname,nmask,int(sigma[-1]*MAXB)),
                {
                    'v_twoStageAdmm_fastdvd_gray_bayer':v_twoStageAdmm_fastdvd_gray_bayer,
                    'psnr_fastdvd_gray':psnr_fastdvd_gray,
                    'ssim_fastdvd_gray':ssim_fastdvd_gray,
                    # 'v_twoStageAdmm_fastdvd_color':onlinefastdvdnet,

                    'orig_real':orig_real,
                    'meas_bayer':meas_bayer
                    })
print('all= ')
print(round(mean(average_psnr),2),end=', ')
print(round(mean(average_ssim),4))
# print(mean(average_time))
f.close()
