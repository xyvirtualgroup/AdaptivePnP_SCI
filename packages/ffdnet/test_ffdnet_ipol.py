"""
Denoise an image with the FFDNet denoising method

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable, grad
# from models import FFDNet
# from utils import batch_psnr, normalize, init_logger_ipol, \
# 				variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb
from .models import FFDNet
from .utils import batch_psnr, normalize, init_logger_ipol,gen_bayer_img,A_,At_
torch.autograd.set_detect_anomaly(True)

# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def ffdnet_imdenoiser(imnoisy, sigma, model=None, useGPU=True):
    r"""Denoises an input image (M x N) with FFDNet
	"""
    # from HxWxC to  CxHxW grayscale image (C=1)
    imnoisy = np.expand_dims(imnoisy, 0)

    # # Handle odd sizes
    # expanded_h = False
    # expanded_w = False
    # sh_im = imorig.shape
    # if sh_im[2]%2 == 1:
    # 	expanded_h = True
    # 	imorig = np.concatenate((imorig, \
    # 			imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

    # if sh_im[3]%2 == 1:
    # 	expanded_w = True
    # 	imorig = np.concatenate((imorig, \
    # 			imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

    imnoisy = normalize(imnoisy)
    imnoisy = torch.Tensor(imnoisy)

    if model is None:
        in_ch = 1
        model_fn = 'models/net_gray.pth'
        # Absolute path to model file
        model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                                model_fn)

        # Create model
        print('Loading model ...\n')
        net = FFDNet(num_input_channels=in_ch)
        # Load saved weights
        if useGPU:
            state_dict = torch.load(model_fn)
            device_ids = [0]
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
        else:
            state_dict = torch.load(model_fn, map_location='cpu')
            # CPU mode: remove the DataParallel wrapper
            state_dict = remove_dataparallel_wrapper(state_dict)
            model = net
        model.load_state_dict(state_dict)

    
    model.eval()

    # # Sets data type according to CPU or GPU modes
    # if useGPU:
    # 	dtype = torch.cuda.FloatTensor
    # else:
    # 	dtype = torch.FloatTensor

    #     # Test mode
    # with torch.no_grad(): # PyTorch v0.4.0
    #     imorig, imnoisy = Variable(imorig.type(dtype)), \
    #     				Variable(imnoisy.type(dtype))
    #     nsigma = Variable(
    #     		torch.FloatTensor([args['noise_sigma']]).type(dtype))

    # Estimate noise and subtract it to the input image
    im_noise_estim = model(imnoisy, sigma)
    outim = torch.clamp(imnoisy - im_noise_estim, 0., 1.)

    return outim


def ffdnet_vdenoiser(vnoisy, sigma, model=None, useGPU=True):
    r"""Denoises an input video (M x N x F) with FFDNet in a frame-wise manner
	"""
    if model is None:
        in_ch = 1
        model_fn = 'models/net_gray.pth'
        # Absolute path to model file
        model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                                model_fn)

        # Create model
        # print('Loading model ...\n')
        net = FFDNet(num_input_channels=in_ch)
        # Load saved weights
        if useGPU:
            state_dict = torch.load(model_fn)
            device_ids = [0]
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
        else:
            state_dict = torch.load(model_fn, map_location='cpu')
            # CPU mode: remove the DataParallel wrapper
            state_dict = remove_dataparallel_wrapper(state_dict)
            model = net
        model.load_state_dict(state_dict)

    
    model.eval()

    # apply FFDNet denoising for each frame
    vshape = vnoisy.shape
    vnoisy = vnoisy.reshape(*vshape[0:2], -1)
    nmask = vnoisy.shape[-1]
    outv = np.zeros(vnoisy.shape)
    for imask in range(nmask):
        # imnoisy = vnoisy[:,:,imask]*255 # to match the scale of the input [0,255]
        imnoisy = vnoisy[:, :, imask]  # to match the scale of the input [0,255]

        # from HxWxC to  CxHxW grayscale image (C=1)
        imnoisy = np.expand_dims(imnoisy, 0)
        imnoisy = np.expand_dims(imnoisy, 0)

        # # Handle odd sizes
        # expanded_h = False
        # expanded_w = False
        # sh_im = imorig.shape
        # if sh_im[2]%2 == 1:
        # 	expanded_h = True
        # 	imorig = np.concatenate((imorig, \
        # 			imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

        # if sh_im[3]%2 == 1:
        # 	expanded_w = True
        # 	imorig = np.concatenate((imorig, \
        # 			imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

        # imnoisy = normalize(imnoisy) # omit normalization
        imnoisy = torch.Tensor(imnoisy)
        # Sets data type according to CPU or GPU modes
        if useGPU:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # Test mode
        with torch.no_grad():  # PyTorch v0.4.0
            imnoisy = Variable(imnoisy.type(dtype))
            sigma = Variable(torch.FloatTensor([sigma]).type(dtype))

        # Estimate noise and subtract it to the input image
        im_noise_estim = model(imnoisy, sigma)

        # # with clip/clamp
        # outim = torch.clamp(imnoisy-im_noise_estim, 0., 1.)
        # without clip/clamp
        outim = imnoisy - im_noise_estim
        outv[:, :, imask] = (outim.data.cpu().numpy()[0, 0, :])

    outv = outv.reshape(vshape)
    return outv


# convert single (HxWxC) to 4-dimensional torch tensor
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)


# convert torch tensor to single
def tensor2single(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return img

def tensor4to3dim(img):
    img = img.data.squeeze().float()
    if img.ndim == 3:
        img = img.permute(1, 2, 0)

    return img
def tensor3to4dim(img):   
    return img.permute(2, 0, 1).float().unsqueeze(0)


def ffdnet_rgb_denoise(x, sigma,model):
    
    #model_path = os.path.join(model_pool, model_name+'.pth')

    # ----------------------------------------
  
    #need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 

    # ----------------------------------------
    # load model
    # ----------------------------------------

    #from packages.ffdnet.models.network_ffdnet import FFDNet as net
    
    
    (nrow, ncol, ncolor,nmask) = x.shape
    outv = np.zeros(x.shape)
    
    for imask in range(nmask):
    #img_L = util.uint2single(x)
        x_L = x[:,:,:,imask]
        img_L = single2tensor4(x_L)
        img_L = img_L.to(device)
    
        sigma1 = torch.full((1,1,1,1), sigma).type_as(img_L)
        img_E = model(img_L, sigma1)
        x_d = tensor2single(img_E)
    #        x_d = x_d/255
        outv[:,:,:,imask] = x_d #(x_d.data.cpu().numpy()[0, 0, 0,:])
    return outv

def ffdnet_rgb_denoise_full_tensor(x, yall, Phiall, sigma, model, useGPU=True, lr_=0.000001, updata_=False,update_per_iter=4,device=0):
    
    # device = torch.device(device if torch.cuda.is_available() else 'cpu')

   

    (nrow, ncol, ncolor, nmask) = x.shape

    if updata_:

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_)  # using ADAM opt       
        
        
        xbgr3 = torch.zeros(x.shape).cuda()

        mse = torch.nn.MSELoss().cuda()  # using MSE loss
        # yall = yall.type('torch.cuda.FloatTensor').cuda()
        # Phiall = Phiall.type('torch.cuda.FloatTensor').cuda()
        # yall = yall.requires_grad_()
        m0, m1, m2, m3 = Phiall.shape[0], Phiall.shape[1], Phiall.shape[3], Phiall.shape[2]
        xall = torch.zeros(m0, m1, m3, m2).cuda()
        # up_meas = torch.zeros(m0, m1, m2).cuda()
        for _ in range(update_per_iter):
            
            
            for imask in range(nmask):
                # x_g = Variable(x).requires_grad_()
                x_L = x[:, :, :, imask]
                img_L = x_L.permute(2, 0, 1).float().unsqueeze(0)

                sigma1 = torch.full((1, 1, 1, 1), sigma).type_as(img_L)
                xbgr3[:, :, :, imask] = model(img_L, sigma1).permute(2, 3, 1, 0).squeeze(3)
                #  = out
            
            xall[..., 0] = xbgr3[0::2, 0::2, 0, :]  # R  channel (average over two)
            xall[..., 1] = xbgr3[0::2, 1::2, 1, :]  # G1=G2 channel (average over two)
            xall[..., 2] = xbgr3[1::2, 0::2, 1, :]  # G2=G1 channel (average over two)
            xall[..., 3] = xbgr3[1::2, 1::2, 2, :]  # B  channel (average over two)
            # xall = gen_bayer_img(xbgr3,4)
            # x_re = torch.zeros(m0, m1, m3).cuda()
           
            # for ib in range(4):
            #     # b = bayer[ib]
            #     x_re = xall[..., ib]

            #     # x_re=x_bayer[b[0]::2, b[1]::2]
            #     matrix1 = Phiall[..., ib]
            #     up_meas[:, :, ib] = torch.sum(x_re * matrix1, dim=2)
            up_meas = torch.sum(xall * Phiall, dim=2)
            # up_meas = A_(xall,Phiall)
            total_loss = mse(up_meas,yall)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            xbgr3 = xbgr3.detach()
            xall = xall.detach()
        
            print('loss:',end=' ')
            print(total_loss)

            # optimizer.step()

        outv = torch.zeros(x.shape).cuda()
        model.eval()
        for imask in range(nmask):
            # img_L = util.uint2single(x)
            x_L = x[:, :, :, imask]
            img_L = x_L.permute(2, 0, 1).float().unsqueeze(0)
            img_L = img_L.cuda()

            sigma1 = torch.full((1, 1, 1, 1), sigma).type_as(img_L)
            img_E = model(img_L, sigma1)
            x_d = tensor4to3dim(img_E)
            #        x_d = x_d/255
            outv[:, :, :, imask] = x_d  # (x_d.data.cpu().numpy()[0, 0, 0,:])

        xall[..., 0] = outv[0::2, 0::2, 0, :]  # R  channel (average over two)
        xall[..., 1] = outv[0::2, 1::2, 1, :]  # G1=G2 channel (average over two)
        xall[..., 2] = outv[1::2, 0::2, 1, :]  # G2=G1 channel (average over two)
        xall[..., 3] = outv[1::2, 1::2, 2, :]  # B  channel (average over two)
        # xall = gen_bayer_img(outv,4)
            # x_re = torch.zeros(m0, m1, m3).cuda()
           
            # for ib in range(4):
            #     # b = bayer[ib]
            #     x_re = xall[..., ib]

            #     # x_re=x_bayer[b[0]::2, b[1]::2]
            #     matrix1 = Phiall[..., ib]
            #     up_meas[:, :, ib] = torch.sum(x_re * matrix1, dim=2)
        up_meas = torch.sum(xall * Phiall, dim=2)
        total_loss = mse(up_meas,yall)
        print('loss:',end=' ')
        print(total_loss)

        # if if_continue:
        #     torch.save(model.state_dict(), './model_zoo/update/ffdnet_color.pth')
		# scheduler.step()
        
    else:

        outv = torch.zeros(x.shape).cuda()

        for imask in range(nmask):
            # img_L = util.uint2single(x)
            x_L = x[:, :, :, imask]
            img_L = tensor3to4dim(x_L)
            img_L = img_L.cuda()

            sigma1 = torch.full((1, 1, 1, 1), sigma).type_as(img_L)
            img_E = model(img_L, sigma1)
            x_d = tensor4to3dim(img_E)
            #        x_d = x_d/255
            outv[:, :, :, imask] = x_d  # (x_d.data.cpu().numpy()[0, 0, 0,:])
    torch.cuda.empty_cache()
    if updata_:
        return outv, model
    else:
        return outv


def ffdnet_rgb_denoise_full_tensor_large(x, yall, Phiall, sigma, model, useGPU=True, lr_=0.000001, updata_=False,update_per_iter=4,device=0):
    
    # device = torch.device(device if torch.cuda.is_available() else 'cpu')

   

    (nrow, ncol, ncolor, nmask) = x.shape

    if updata_:

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_)  # using ADAM opt       
        
        
        xbgr3 = torch.zeros(x.shape).cuda()

        mse = torch.nn.MSELoss().cuda()  # using MSE loss
        # yall = yall.type('torch.cuda.FloatTensor').cuda()
        # Phiall = Phiall.type('torch.cuda.FloatTensor').cuda()
        # yall = yall.requires_grad_()
        m0, m1, m2, m3 = Phiall.shape[0], Phiall.shape[1], Phiall.shape[3], Phiall.shape[2]
        xall = torch.zeros(m0, m1, m3, m2).cuda()
        xall_i = torch.zeros(m0, m1,m2).cuda()
        # up_meas = torch.zeros(m0, m1, m2).cuda()
        for _ in range(update_per_iter):
            
            
            for imask in range(nmask):
                # x_g = Variable(x).requires_grad_()
                x_L = x[:, :, :, imask]
                img_L = x_L.permute(2, 0, 1).float().unsqueeze(0)

                sigma1 = torch.full((1, 1, 1, 1), sigma).type_as(img_L)
                xbgr3_i = model(img_L, sigma1).permute(2, 3, 1, 0).squeeze(3)
                #  = out
            
                xall_i[..., 0] = xbgr3[0::2, 0::2, 0, :]  # R  channel (average over two)
                xall_i[..., 1] = xbgr3[0::2, 1::2, 1, :]  # G1=G2 channel (average over two)
                xall_i[..., 2] = xbgr3[1::2, 0::2, 1, :]  # G2=G1 channel (average over two)
                xall_i[..., 3] = xbgr3[1::2, 1::2, 2, :]  # B  channel (average over two)

                up_meas = torch.sum(xall * Phiall, dim=2)
                # up_meas = A_(xall,Phiall)
                total_loss = mse(up_meas,yall)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                xbgr3 = xbgr3.detach()
                xall = xall.detach()
        
            print('loss:',end=' ')
            print(total_loss)

            # optimizer.step()

        outv = torch.zeros(x.shape).cuda()
        model.eval()
        for imask in range(nmask):
            # img_L = util.uint2single(x)
            x_L = x[:, :, :, imask]
            img_L = x_L.permute(2, 0, 1).float().unsqueeze(0)
            img_L = img_L.cuda()

            sigma1 = torch.full((1, 1, 1, 1), sigma).type_as(img_L)
            img_E = model(img_L, sigma1)
            x_d = tensor4to3dim(img_E)
            #        x_d = x_d/255
            outv[:, :, :, imask] = x_d  # (x_d.data.cpu().numpy()[0, 0, 0,:])

        xall[..., 0] = outv[0::2, 0::2, 0, :]  # R  channel (average over two)
        xall[..., 1] = outv[0::2, 1::2, 1, :]  # G1=G2 channel (average over two)
        xall[..., 2] = outv[1::2, 0::2, 1, :]  # G2=G1 channel (average over two)
        xall[..., 3] = outv[1::2, 1::2, 2, :]  # B  channel (average over two)
        # xall = gen_bayer_img(outv,4)
            # x_re = torch.zeros(m0, m1, m3).cuda()
           
            # for ib in range(4):
            #     # b = bayer[ib]
            #     x_re = xall[..., ib]

            #     # x_re=x_bayer[b[0]::2, b[1]::2]
            #     matrix1 = Phiall[..., ib]
            #     up_meas[:, :, ib] = torch.sum(x_re * matrix1, dim=2)
        up_meas = torch.sum(xall * Phiall, dim=2)
        total_loss = mse(up_meas,yall)
        print('loss:',end=' ')
        print(total_loss)

        # if if_continue:
        #     torch.save(model.state_dict(), './model_zoo/update/ffdnet_color.pth')
		# scheduler.step()
        
    else:

        outv = torch.zeros(x.shape).cuda()

        for imask in range(nmask):
            # img_L = util.uint2single(x)
            x_L = x[:, :, :, imask]
            img_L = tensor3to4dim(x_L)
            img_L = img_L.cuda()

            sigma1 = torch.full((1, 1, 1, 1), sigma).type_as(img_L)
            img_E = model(img_L, sigma1)
            x_d = tensor4to3dim(img_E)
            #        x_d = x_d/255
            outv[:, :, :, imask] = x_d  # (x_d.data.cpu().numpy()[0, 0, 0,:])
    torch.cuda.empty_cache()
    if updata_:
        return outv, model
    else:
        return outv



def ffdnet_rgb_denoise_full_tensor_v2(x, yall=None, Phiall=None, sigma=[25/255], model=None, useGPU=True, lr_=0.000001, updata_=False,device=0,update_per_iter=4):
    
    # device = torch.device(device if torch.cuda.is_available() else 'cpu')

   
    assert not model == None
    (nrow, ncol, ncolor, nmask) = x.shape

    if updata_:

    
        outv1 = torch.zeros(x.shape).cuda()

        mse = torch.nn.MSELoss().cuda()  # using MSE loss
      

        for imask in range(nmask):
            
            x_L = x[:, :, :, imask]
            img_L = x_L.permute(2, 0, 1).float().unsqueeze(0)
            img_L = img_L.cuda()

            sigma1 = torch.full((1, 1, 1, 1), sigma).type_as(img_L)
            img_E = model(img_L, sigma1)
            mid_ = img_E.permute(2, 3, 1, 0)
            outv1[:, :, :, imask] = mid_.squeeze(3)

        xbgr3 = outv1
        yall = yall.type('torch.cuda.FloatTensor').cuda()
        Phiall = Phiall.type('torch.cuda.FloatTensor').cuda()
        m0, m1, m2 = Phiall.shape # M N B
        xall = gen_bayer_img(xbgr3)
        up_meas = A_(xall,Phiall)
        
        
        total_loss = mse(yall, up_meas)
        print('loss:',end=' ')
        print(total_loss)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr_)  # using ADAM opt       
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        outv = torch.zeros(x.shape).cuda()

        for imask in range(nmask):
            # img_L = util.uint2single(x)
            x_L = x[:, :, :, imask]
            img_L = x_L.permute(2, 0, 1).float().unsqueeze(0)
            img_L = img_L.cuda()

            sigma1 = torch.full((1, 1, 1, 1), sigma).type_as(img_L)
            img_E = model(img_L, sigma1)
            x_d = tensor4to3dim(img_E)
            #        x_d = x_d/255
            outv[:, :, :, imask] = x_d  # (x_d.data.cpu().numpy()[0, 0, 0,:])

        # if if_continue:
        #     torch.save(model.state_dict(), './model_zoo/update/ffdnet_color.pth')
		# scheduler.step()
    else:

        outv = torch.zeros(x.shape).cuda()

        for imask in range(nmask):
            # img_L = util.uint2single(x)
            x_L = x[:, :, :, imask]
            img_L = tensor3to4dim(x_L)
            img_L = img_L.cuda()

            sigma1 = torch.full((1, 1, 1, 1), sigma).type_as(img_L)
            img_E = model(img_L, sigma1)
            x_d = tensor4to3dim(img_E)
            #        x_d = x_d/255
            outv[:, :, :, imask] = x_d  # (x_d.data.cpu().numpy()[0, 0, 0,:])
    torch.cuda.empty_cache()
    if updata_:
        return outv, model
    else:
        return outv

def test_ffdnet(**args):
    r"""Denoises an input image with FFDNet
	"""
    # Init logger
    logger = init_logger_ipol()

    # Check if input exists and if it is RGB
    try:
        rgb_den = is_rgb(args['input'])
    except:
        raise Exception('Could not open the input image')

    # Open image as a CxHxW torch.Tensor
    if rgb_den:
        in_ch = 3
        model_fn = 'models/net_rgb.pth'
        imorig = cv2.imread(args['input'])
        # from HxWxC to CxHxW, RGB image
        imorig = (cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
    else:
        # from HxWxC to  CxHxW grayscale image (C=1)
        in_ch = 1
        model_fn = 'models/net_gray.pth'
        imorig = cv2.imread(args['input'], cv2.IMREAD_GRAYSCALE)
        imorig = np.expand_dims(imorig, 0)
    imorig = np.expand_dims(imorig, 0)

    # Handle odd sizes
    expanded_h = False
    expanded_w = False
    sh_im = imorig.shape
    if sh_im[2] % 2 == 1:
        expanded_h = True
        imorig = np.concatenate((imorig, \
                                 imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

    if sh_im[3] % 2 == 1:
        expanded_w = True
        imorig = np.concatenate((imorig, \
                                 imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

    imorig = normalize(imorig)
    imorig = torch.Tensor(imorig)

    # Absolute path to model file
    model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                            model_fn)

    # Create model
    print('Loading model ...\n')
    net = FFDNet(num_input_channels=in_ch)

    # Load saved weights
    if args['cuda']:
        state_dict = torch.load(model_fn)
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
    else:
        state_dict = torch.load(model_fn, map_location='cpu')
        # CPU mode: remove the DataParallel wrapper
        state_dict = remove_dataparallel_wrapper(state_dict)
        model = net
    model.load_state_dict(state_dict)

    
    model.eval()

    # Sets data type according to CPU or GPU modes
    if args['cuda']:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Add noise
    if args['add_noise']:
        noise = torch.FloatTensor(imorig.size()). \
            normal_(mean=0, std=args['noise_sigma'])
        imnoisy = imorig + noise
    else:
        imnoisy = imorig.clone()

    # Test mode
    with torch.no_grad():  # PyTorch v0.4.0
        imorig, imnoisy = Variable(imorig.type(dtype)), \
                          Variable(imnoisy.type(dtype))
        nsigma = Variable(
            torch.FloatTensor([args['noise_sigma']]).type(dtype))

    # Measure runtime
    start_t = time.time()

    # Estimate noise and subtract it to the input image
    im_noise_estim = model(imnoisy, nsigma)
    outim = torch.clamp(imnoisy - im_noise_estim, 0., 1.)
    stop_t = time.time()

    if expanded_h:
        imorig = imorig[:, :, :-1, :]
        outim = outim[:, :, :-1, :]
        imnoisy = imnoisy[:, :, :-1, :]

    if expanded_w:
        imorig = imorig[:, :, :, :-1]
        outim = outim[:, :, :, :-1]
        imnoisy = imnoisy[:, :, :, :-1]

    # Compute PSNR and log it
    if rgb_den:
        logger.info("### RGB denoising ###")
    else:
        logger.info("### Grayscale denoising ###")
    if args['add_noise']:
        psnr = batch_psnr(outim, imorig, 1.)
        psnr_noisy = batch_psnr(imnoisy, imorig, 1.)

        logger.info("\tPSNR noisy {0:0.2f}dB".format(psnr_noisy))
        logger.info("\tPSNR denoised {0:0.2f}dB".format(psnr))
    else:
        logger.info("\tNo noise was added, cannot compute PSNR")
    logger.info("\tRuntime {0:0.4f}s".format(stop_t - start_t))

    # Compute difference
    diffout = 2 * (outim - imorig) + .5
    diffnoise = 2 * (imnoisy - imorig) + .5

    # Save images
    if not args['dont_save_results']:
        noisyimg = variable_to_cv2_image(imnoisy)
        outimg = variable_to_cv2_image(outim)
        cv2.imwrite("noisy.png", noisyimg)
        cv2.imwrite("ffdnet.png", outimg)
        if args['add_noise']:
            cv2.imwrite("noisy_diff.png", variable_to_cv2_image(diffnoise))
            cv2.imwrite("ffdnet_diff.png", variable_to_cv2_image(diffout))


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="FFDNet_Test")
    parser.add_argument('--add_noise', type=str, default="True")
    parser.add_argument("--input", type=str, default="", \
                        help='path to input image')
    parser.add_argument("--suffix", type=str, default="", \
                        help='suffix to add to output name')
    parser.add_argument("--noise_sigma", type=float, default=25, \
                        help='noise level used on test set')
    parser.add_argument("--dont_save_results", action='store_true', \
                        help="don't save output images")
    parser.add_argument("--no_gpu", action='store_true', \
                        help="run model on CPU")
    argspar = parser.parse_args()
    # Normalize noises ot [0, 1]
    argspar.noise_sigma /= 255.

    # String to bool
    argspar.add_noise = (argspar.add_noise.lower() == 'true')

    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FFDNet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    test_ffdnet(**vars(argspar))
