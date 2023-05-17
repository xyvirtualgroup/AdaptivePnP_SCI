"""
ddnet denoising algorithm

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import torch
import torch.nn.functional as F
# import os
# import argparse
# import time
# import cv2



import imageio as io
NUM_IN_FR_EXT = 5 # temporal size of patch



def temp_denoise(model, noisyframe, sigma_noise):
	'''Encapsulates call to denoising model and handles padding.
		Expects noisyframe to be normalized in [0., 1.]
	'''
	# make size a multiple of four (we have two scales in the denoiser)
	sh_im = noisyframe.size()
	expanded_h = sh_im[-2]%4
	if expanded_h:
		expanded_h = 4-expanded_h
	expanded_w = sh_im[-1]%4
	if expanded_w:
		expanded_w = 4-expanded_w
	padexp = (0, expanded_w, 0, expanded_h)
	noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
	sigma_noise = F.pad(input=sigma_noise, pad=padexp, mode='reflect')

	# denoise
	out = torch.clamp(model(noisyframe), 0., 1.)

	if expanded_h:
		out = out[:, :, :-expanded_h, :]
	if expanded_w:
		out = out[:, :, :, :-expanded_w]

	return out



def denoise_seq_ddnet(seq, noise_std, temp_psz, model_temporal):
	r"""Denoises a sequence of frames with ddnet.

	Args:
		seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
		noise_std: Tensor. Standard deviation of the added noise
		temp_psz: size of the temporal patch
		model_temp: instance of the PyTorch model of the temporal denoiser
	Returns:
		denframes: Tensor, [numframes, C, H, W]
	"""
	# init arrays to handle contiguous frames and related patches
	numframes, C, H, W = seq.shape
	ctrlfr_idx = int((temp_psz-1)//2)
	inframes = list()
	denframes = torch.empty((numframes, C, H, W)).to(seq.device)

	# build noise map from noise std---assuming Gaussian noise
	noise_map = noise_std.expand((1, 1, H, W))

	for fridx in range(numframes):
		# load input frames
		if not inframes:
		# if list not yet created, fill it with temp_patchsz frames
			for idx in range(temp_psz):
				relidx = abs(idx-ctrlfr_idx) # handle border conditions, reflect
				inframes.append(seq[relidx])
		else:
			del inframes[0]
			relidx = min(fridx + ctrlfr_idx, -fridx + 2*(numframes-1)-ctrlfr_idx) # handle border conditions
			inframes.append(seq[relidx])

		inframes_t = torch.stack(inframes, dim=0).contiguous().view((1, temp_psz*C, H, W)).to(seq.device)

		# append result to output list
		denframes[fridx] = temp_denoise(model_temporal, inframes_t, noise_map)

	# free memory up
	del inframes
	del inframes_t
	torch.cuda.empty_cache()

	# convert to appropiate type and return
	return denframes

def temp_denoise_for_train(model, noisyframe, sigma_noise):
	'''Encapsulates call to denoising model and handles padding.
		Expects noisyframe to be normalized in [0., 1.]
	'''
	# make size a multiple of four (we have two scales in the denoiser)
	sh_im = noisyframe.size()
	expanded_h = sh_im[-2]%4
	if expanded_h:
		expanded_h = 4-expanded_h
	expanded_w = sh_im[-1]%4
	if expanded_w:
		expanded_w = 4-expanded_w
	padexp = (0, expanded_w, 0, expanded_h)
	noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
	sigma_noise = F.pad(input=sigma_noise, pad=padexp, mode='reflect')

	# denoise
	out = model(noisyframe, sigma_noise)

	if expanded_h:
		out = out[:, :, :-expanded_h, :]
	if expanded_w:
		out = out[:, :, :, :-expanded_w]

	return out

def denoise_seq_ddnet_for_train(seq, noise_map, temp_psz, model_temporal):
	r"""Denoises a sequence of frames with ddnet.

	Args:
		seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
		noise_std: Tensor. Standard deviation of the added noise
		temp_psz: size of the temporal patch
		model_temp: instance of the PyTorch model of the temporal denoiser
	Returns:
		denframes: Tensor, [numframes, C, H, W]
	"""
	# init arrays to handle contiguous frames and related patches
	numframes, C, H, W = seq.shape
	ctrlfr_idx = int((temp_psz-1)//2)
	inframes = list()
	denframes = torch.empty((numframes, C, H, W)).to(seq.device)

	# build noise map from noise std---assuming Gaussian noise
	# noise_map = noise_std.expand((1, 1, H, W))

	for fridx in range(numframes):
		# load input frames
		if not inframes:
		# if list not yet created, fill it with temp_patchsz frames
			for idx in range(temp_psz):
				relidx = abs(idx-ctrlfr_idx) # handle border conditions, reflect
				inframes.append(seq[relidx])
		else:
			del inframes[0]
			relidx = min(fridx + ctrlfr_idx, -fridx + 2*(numframes-1)-ctrlfr_idx) # handle border conditions
			inframes.append(seq[relidx])
		# inframes patch  15 w h
		inframes_t = torch.stack(inframes, dim=0).contiguous().view((1, temp_psz*C, H, W)).to(seq.device)

		# append result to output list
		#iframe_t 1 90 w h
		denframes[fridx] = temp_denoise_for_train(model_temporal, inframes_t, noise_map[fridx].unsqueeze(dim=0))
		# denframes 4 15  h w

	# free memory up
	del inframes
	del inframes_t
	torch.cuda.empty_cache()

	# convert to appropiate type and return
	return denframes

def ddnet_seqdenoise(seq, windsize, model):
	
	# print(seq.shape)
	N, C, H, W = seq.shape
	hw = int((windsize-1)//2) # half window size
	seq_denoised = torch.empty((N, C, H, W)).to(seq.device)

	for frameidx in range(N):
		# cicular padding for edge frames in the video sequence
		idx = (torch.tensor(range(frameidx, frameidx+windsize)) - hw) % N # circular padding
		noisy_seq = seq[idx].reshape((1, -1, H, W)) # reshape from [W, C, H, W] to [1, W*C, H, W]
		
		# make sure the width W and height H multiples of 4
		#   pad the width W and height H to multiples of 4
		M = 4 # multipier
		wpad, hpad = W%M, H%M
		if wpad:
			wpad = M-wpad
		if hpad:
			hpad = M-hpad
		pad = (0, wpad, 0, hpad) 
		noisy_seq = F.pad(noisy_seq, pad, mode='reflect')
	
		
		# apply the denoising model to the input datat
		frame_denoised = model(noisy_seq)

		# unpad the results
		if wpad:
			frame_denoised = frame_denoised[:, :, :, :-wpad]
		if hpad:
			frame_denoised = frame_denoised[:, :, :-hpad, :]
		
		seq_denoised[frameidx] = frame_denoised

		# # apply the denoising model to the input datat
		# seq_denoised[frameidx] = model(noisy_seq, noise_map)

	return seq_denoised



def gen_bayer_img(rgb_video):
	# W,H,C,frame = rgb_video.shape
	bayer = torch.zeros_like(rgb_video).cuda()
	bayer[0::2,0::2,0,:] = rgb_video[0::2,0::2,0,:] # R  channel (average over two)
	bayer[0::2,1::2,1,:] = rgb_video[0::2,1::2,1,:] # G1=G2 channel (average over two)
	bayer[1::2,0::2,1,:] = rgb_video[1::2,0::2,1,:] # G2=G1 channel (average over two)
	bayer[1::2,1::2,2,:] = rgb_video[1::2,1::2,2,:] # B  channel (average over two)  	
	bayer = bayer.permute(3,2,0,1)
	return bayer

def test_ddnet(vnoisy,yall, Phiall,model=None, useGPU=True,args = None,gray=False):
	updata_ = False
	if args!=None:
		lr_ = args.dm_lr
		update_per_iter = args.dm_update_per_iter
		updata_ = args.dm_update

	

	# start_time = time.time()
	nColor = 1 if gray else 3 # number of color channels (3 - RGB color, 1 - grayscale)
	# Sets data type according to CPU or GPU modes
	#useGPU=True
	if useGPU:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')


    
	# print(vnoisy.finfo, noisestd.finfo)                                                                                                                                                                                                                                        
	if updata_:
		mse = torch.nn.MSELoss().to(device)  # using MSE loss


		# M,N,B= Phi.shape[0],Phi.shape[1],Phi.shape[2] # M N B
		# print(torch.max(vnoisy),torch.min(vnoisy))
		# vnoisy = torch.clamp(vnoisy,0.,1.)

		# noisestd = torch.FloatTensor([sigma]).to(device)

		#Phi=torch.from_numpy(Phi).type('torch.cuda.FloatTensor').to(device)
		if gray:
			vnoisy = vnoisy.unsqueeze(3) # unsqueeze the color dimension - [H,W,F] to [H,W,F,C=1]
		vnoisy = vnoisy.permute(3,2,0, 1) # from H x W x F x C to F x C x H x W
		model.train()
		# model.eval()
		# model.module.weight_tensor_in.requires_grad =  True
		# model.module.weight_tensor_in2.requires_grad =  True
		# model.module.weight_tensor_out.requires_grad =  True
		# for (name, param) in model.module.temp1.named_parameters():
		# 	# param.requires_grad = False
		# 	if isinstance(param,torch.nn.parameter.Parameter):
		# 		m.requires_grad = True

		for iter in range(update_per_iter):	
			outv = ddnet_seqdenoise( seq=vnoisy,\
											windsize=NUM_IN_FR_EXT,\
											model=model ) #8 3 512 512

			outv = outv.permute(2, 3, 1,0) # back from F x C x H x W to H x W  x C x F

			if gray:
				outv = outv.squeeze(3) 
			
			xall = gen_bayer_img(outv)

			total_loss=mse(vnoisy,xall)

			optimizer = torch.optim.Adam(model.parameters(), lr=lr_)  # using ADAM opt
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()
			print('ddn loss:',end=' ')
			print(total_loss)

		# model.eval()
		with torch.no_grad():
			outv = ddnet_seqdenoise( seq=vnoisy,\
										windsize=NUM_IN_FR_EXT,\
										model=model)
		# print(outv.shape)
		# print(torch.max(outv),torch.min(outv))
		outv = outv.permute(2, 3, 1,0) # back from F x C x H x W to H x W x F x C
		if gray:
			outv = outv.squeeze(3) # squeeze the color dimension - [H,W,F,C=1] to [H,W,F]
	
	else:
		model.eval()

		
		if gray:
			vnoisy = vnoisy.unsqueeze(3) # unsqueeze the color dimension - [H,W,F] to [H,W,F,C=1]
		vnoisy = vnoisy.permute(3,2,0, 1) # from H x W x F x C to F x C x H x W

		with torch.no_grad():

			outv = ddnet_seqdenoise( seq=vnoisy,\
										windsize=NUM_IN_FR_EXT,\
										model=model )
		# print(outv.shape)
		# print(torch.max(outv),torch.min(outv))
		outv = outv.permute(2,3, 1,0) # back from F x C x H x W to H x W x F x C
		if gray:
			outv = outv.squeeze(3) # squeeze the color dimension - [H,W,F,C=1] to [H,W,F]
		

	torch.cuda.empty_cache()

	
	if updata_:
		return outv, model
	else:
		return outv
