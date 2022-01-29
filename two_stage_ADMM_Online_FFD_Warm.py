


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
from models.network_ffdnet import FFDNet as net
from models.network_demosaicking import DDnet


    
    
#==============================================================================
worker_init_fn(0)
datasetdir = './dataset/cacti/mid_scale' # color middle scale dataset 512*512 px
update = True # use 'online' denoiser
reuse_model = True # use refined model for later frames
denoiser = 'ffdnet_color' # video denosing network
deep_demosaicking = True
# pre-load the model for FFDNet image denoising
model_name = 'ffdnet_color'          

    #sf = 1                    # unused for denoising
if 'color' in model_name:
        n_channels = 3        # setting for color image
        nc = 96               # setting for color image
        nb = 12               # setting for color image
else:
        n_channels = 1        # setting for grayscale image
        nc = 64               # setting for grayscale image
        nb = 15               # setting for grayscale image

model_pool = 'model_zoo'  # fixed
model_denoise_path = os.path.join(model_pool, model_name+'.pth')
model_demosaic_path = os.path.join(model_pool,'ddnet.pth')


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

# sigma = [50 / 255, 25 / 255, 12 / 255,6/255]  # pre-set noise standard deviation
# iter_max = [20, 20, 20,20]  # maximum number of iterations


# lr = 1e-5
# update_per_iter = 1

# interval_iter = 12

for ncount in range(0,6):

    average_psnr_i = []
    average_ssim_i = []
    average_time_i = []
    psnrall_ffdnet_gray = []
    if ncount == 0:
        datname = 'Beauty_bayer'
        sigma = [25/255,12/255,6/255]  # pre-set noise standard deviation
        iter_max = [15,6,3]  # maximum number of iterations

        lr = 2e-6
        update_per_iter = 2                  
        interval_iter = 15 
        
        ## pre params
        # sigma = [ 25/255]  # pre-set noise standard deviation
        # iter_max = [8]  # maximum number of iterations 
        
        # sigma = [50/255,25/255,12/255,6/255]  # time test
        # iter_max = [20,20,20,20]  # time test
        # interval_iter = 60 # time test

        

    elif ncount == 1:
        datname = 'Bosphorus_bayer'
        sigma = [50/255,25/255,12/255,6/255]  #    pre-set noise standard deviation
        iter_max = [8,4,4,4]  # maximum number of iterations\

        lr = 2e-6
        update_per_iter = 2                  
        interval_iter = 8 
        
        # sigma = [ 25/255]  # pre-set noise standard deviation
        # iter_max = [8]  # maximum number of iterations
        # sigma = [50 / 255, 25 / 255, 12 / 255,6/255]  # pre-set noise standard deviation
        # iter_max = [30, 20, 20,2]  # maximum number of iterations


    elif ncount == 2:
        datname = 'Jockey_bayer'
        sigma = [25/255,12/255,6/255]  #    pre-set noise standard deviation
        iter_max = [16,8,4]  # maximum number of iterations
        if deep_demosaicking:
            sigma = [25/255,12/255,6/255]  #    pre-set noise standard deviation
            iter_max = [14,8,8]  # maximum number of iterations
        ## pre params
        # sigma = [50 / 255,  25 / 255,12 / 255]  # pre-set noise standard deviation
        # iter_max = [10, 15,5]  # maximum number of iterations
        #0.1+
        # sigma = [50 / 255, 40 / 255, 10 / 255]  # pre-set noise standard deviation
        # iter_max = [28, 20, 25]  # maximum number of iterations
        lr = 2e-6
        update_per_iter = 2                 
        interval_iter = 16


    elif ncount == 3:
        datname = 'Runner_bayer'
        sigma = [50/255,25/255,12/255,6/255]  #    pre-set noise standard deviation
        iter_max = [8,4,4,4]  # maximum number of iterations
        ## pre params
        # sigma = [ 25 / 255,12/255]  # pre-set noise standard deviation
        # iter_max = [20,20]  # maximum number of iterations
        # sigma = [50 / 255, 25 / 255, 12 / 255,6/255]  # pre-set noise standard deviation
        # iter_max = [20,20,20,20]  # maximum number of iterations
        lr = 2e-6
        update_per_iter = 2                  
        interval_iter = 8 



    elif ncount == 4:
        datname = 'ShakeNDry_bayer'
        sigma = [50/255,25/255,12/255,6/255]  #    pre-set noise standard deviation
        iter_max = [8,4,4,4]  # maximum number of iterations
        ## pre params
        # sigma = [ 25 / 255]  # pre-set noise standard deviation
        # iter_max = [40]  # maximum number of iterations
        # lr = 1e-5
        # update_per_iter = 1
        # inital_iter = 5  # warm start
        # interval_iter = 11  # 13
        lr = 2e-6
        update_per_iter = 2                 
        interval_iter = 10


    elif ncount == 5:
        datname = 'Traffic_bayer'
        sigma = [50/255,25/255]  #    pre-set noise standard deviation
        iter_max = [16,8]  # maximum number of iterations
        ## pre params
        # sigma = [ 50 / 255]  # pre-set noise standard deviation
        # iter_max = [8]  # maximum number of iterations
        # sigma = [50 / 255, 40 / 255, 10 / 255]  # pre-set noise standard deviation
        # iter_max = [30, 20, 20]  # maximum number of iterations
    
        lr = 2e-6
        update_per_iter = 2                  
        interval_iter = 16 
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


    v_twoStageAdmm_ffd_gray_bayer = np.zeros([nrows, ncols, nmask*nmea], dtype=np.float32)
    psnr_ffd_gray = np.zeros([nmask*nmea,1], dtype=np.float32)
    ssim_ffd_gray = np.zeros([nmask*nmea,1], dtype=np.float32)


    onlineffdnet = np.zeros([nmea,ncols,nrows,3,nmask], dtype=np.float32)

    _lambda = 1 # regularization factor
    accelerate = True # enable accelerated version of _twoStageAdmm_
    
    noise_estimate = False 

    useGPU = True # use GPU
    if_continue=False

    
        
    
    model_denoise = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    # model_denoise.load_state_dict(torch.load('packages\\ffdnet\models/ffdnet_color.pth'), strict=True)
    # torch.save(model.state_dict(), './model_zoo/update/ffdnet_color.pth')
    model_denoise.load_state_dict(torch.load(model_denoise_path), strict=True)
    # for k, v in model.named_parameters():
    #     v.requires_grad = False

    model_denoise.eval()

    model_denoise = model_denoise.cuda()

    if deep_demosaicking:
        # from torch import nn
        model_demosaic = DDnet()
        model_demosaic = torch.nn.DataParallel(model_demosaic).cuda()
        state_dict = torch.load(model_demosaic_path)
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
        
        

        f.write('FFDnet-rgb-demosaic start.\n')

        
        
        begin_time = time.time()


        v_tv = recon_tv[:,:,iframe*nmask:(iframe+1)*nmask]
        onlineffdnet[iframe,:,:,:,:],v_twoStageAdmm_ffdnet_gray_bayer_t,\
            psnr_twoStageAdmm_ffdnet_gray_t,ssim_twoStageAdmm_ffdnet_gray_t,psnrall_ffdnet_gray_t,refined_model =   \
            reconstruct(meas_bayer_t, mask_bayer, _lambda,
                                            0.01, denoiser, iter_max, noise_estimate, sigma,x0_bayer= np2tch_cuda( v_tv),
                                            X_orig=orig_bayer_t, 
                                            model_denoise=model_denoise,
                                            model_demosaic=model_demosaic,
                                            show_iqa=True,demosaic_method ='malvar2004',
                                            lr_=lr,
                                            interval_iter=interval_iter,logf = f,update_=update,update_per_iter=update_per_iter)  
        if reuse_model and update:
            model_denoise = refined_model
        else: 
            model_denoise.load_state_dict(torch.load(model_denoise_path), strict=True)
            model_denoise.eval()
            model_denoise = model_denoise.cuda()                    
                                            
        end_time = time.time()
        t_twoStageAdmm_ffdnet_gray = end_time - begin_time
        print('ADMM-{}--{}-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
            denoiser.upper(),datname,iframe ,mean(psnr_twoStageAdmm_ffdnet_gray_t), mean(ssim_twoStageAdmm_ffdnet_gray_t), t_twoStageAdmm_ffdnet_gray))
        f.write('ADMM-{}--{}-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds. \n'.format(
            denoiser.upper(),datname,iframe , mean(psnr_twoStageAdmm_ffdnet_gray_t), mean(ssim_twoStageAdmm_ffdnet_gray_t), t_twoStageAdmm_ffdnet_gray))
        v_twoStageAdmm_ffd_gray_bayer[:,:,iframe*nmask:(iframe+1)*nmask] =  v_twoStageAdmm_ffdnet_gray_bayer_t
        psnr_ffd_gray[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(psnr_twoStageAdmm_ffdnet_gray_t,(nmask,1))
        ssim_ffd_gray[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(ssim_twoStageAdmm_ffdnet_gray_t,(nmask,1))
        # In[12]:

        

        average_psnr_i.append(mean(psnr_twoStageAdmm_ffdnet_gray_t))  # mean of 8 frames
        average_ssim_i.append(mean(ssim_twoStageAdmm_ffdnet_gray_t))
        psnrall_ffdnet_gray.append(psnrall_ffdnet_gray_t)
        # average_time_i.append(t_twoStageAdmm_ffdnet_color)



        
    print('===========END============'+ datname +'=============END==============')
    sigma_out = [x*255 for x in sigma]
    print(round(mean(average_psnr_i),2),end=', ')
    print(round(mean(average_ssim_i),4))
    print('Params:')
    print(lr,end=' ')
# print(sigma*255)
    print(iter_max,end=' ')
    print(sigma_out,end=' ')

    print('interval_iter='+str(interval_iter),end=' ')
    # print('inital_iter='+str(inital_iter),end=' ')
    print('update_per_iter='+str(update_per_iter))
    # print(mean(average_time_i))

    average_psnr.append(mean(average_psnr_i))
    average_ssim.append(mean(average_ssim_i))
    # average_time.append(mean(average_time_i))

    # In[12]:
    savedmatdir = resultsdir + '/savedmat/'
    if not os.path.exists(savedmatdir):
        os.makedirs(savedmatdir)

    sio.savemat('{}twoStageAdmm_{}_{}{:d}_sigma{:d}_all7_log.mat'.format(savedmatdir,denoiser.lower(),datname,nmask,int(sigma[-1]*MAXB)),
                {
              
                    'v_twoStageAdmm_ffd_gray_bayer':v_twoStageAdmm_ffd_gray_bayer,
                    'psnr_ffd_gray':psnr_ffd_gray,
                    'ssim_ffd_gray':ssim_ffd_gray,
                    'onlineffdnet':onlineffdnet,
                    'psnr_all_iter':psnrall_ffdnet_gray,
              
                    'orig_real':orig_real,
                    'meas_bayer':meas_bayer})
print('all= ')
print(mean(average_psnr))
print(mean(average_ssim))
# print(mean(average_time))
f.close()
            