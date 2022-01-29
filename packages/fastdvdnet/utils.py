"""
Different utilities such as orthogonalization of weights, initialization of
loggers, etc

Copyright (C) 2019, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import subprocess
import glob
import logging
from random import choices # requires Python >= 3.6
import numpy as np
import cv2
import torch
import skimage
if skimage.__version__ < '0.18':
    from skimage.measure import (compare_psnr, compare_ssim)
else: # skimage.measure deprecated in version 0.18 ( -> skimage.metrics )
    from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from tensorboardX import SummaryWriter

IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif') # Supported image types

def A_(x, Phi):
    '''
    Forward model of snapshot compressive imaging (SCI), where multiple coded
    frames are collapsed into a snapshot measurement.
    '''
    return torch.sum(x*Phi, dim=2)  # element-wise product

def At_(y, Phi):
    '''
    Transpose of the forward model. 
    '''
    # (nrow, ncol, nmask) = Phi.shape
    # x = np.zeros((nrow, ncol, nmask))
    # for nt in range(nmask):
    #     x[:,:,nt] = np.multiply(y, Phi[:,:,nt])
    # return x
    return torch.multiply(torch.repeat_interleave(torch.unsqueeze(y,dim=2),Phi.shape[2],axis=2), Phi)

	
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
		R_m, G_m, B_m = masks_CFA_Bayer_tensor((RGB.shape[0],RGB.shape[1]))
		mask = gen_bayer_mask(R_m, G_m, B_m)
		mask = torch.repeat_interleave(mask.unsqueeze(3),RGB.shape[3],dim=3)
		bayer_img = RGB*mask
		bayer_img = torch.sum(bayer_img,dim=2)
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

	
def normalize_augment(datain, ctrl_fr_idx, gray_mode=False):
	'''Normalizes and augments an input patch of dim [N, num_frames, C. H, W] in [0., 255.] to \
		[N, num_frames*C. H, W] in  [0., 1.]. It also returns the central frame of the temporal \
		patch as a ground truth.
	'''
	def transform(sample):
		# define transformations
		do_nothing = lambda x: x
		do_nothing.__name__ = 'do_nothing'
		flipud = lambda x: torch.flip(x, dims=[2])
		flipud.__name__ = 'flipup'
		rot90 = lambda x: torch.rot90(x, k=1, dims=[2, 3])
		rot90.__name__ = 'rot90'
		rot90_flipud = lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
		rot90_flipud.__name__ = 'rot90_flipud'
		rot180 = lambda x: torch.rot90(x, k=2, dims=[2, 3])
		rot180.__name__ = 'rot180'
		rot180_flipud = lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2])
		rot180_flipud.__name__ = 'rot180_flipud'
		rot270 = lambda x: torch.rot90(x, k=3, dims=[2, 3])
		rot270.__name__ = 'rot270'
		rot270_flipud = lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
		rot270_flipud.__name__ = 'rot270_flipud'
		add_csnt = lambda x: x + torch.normal(mean=torch.zeros(x.size()[0], 1, 1, 1), \
								 std=(5/255.)).expand_as(x).to(x.device)
		add_csnt.__name__ = 'add_csnt'

		# define transformations and their frequency, then pick one.
		aug_list = [do_nothing, flipud, rot90, rot90_flipud, \
					rot180, rot180_flipud, rot270, rot270_flipud, add_csnt]
		w_aug = [32, 12, 12, 12, 12, 12, 12, 12, 12] # one fourth chances to do_nothing
		transf = choices(aug_list, w_aug)

		# transform all images in array
		return transf[0](sample)

	img_train = datain
	# GRAY = 0.2989 * R + 0.5870 * G + 0.1140 * B 
	if gray_mode:
		img_train = 0.2989*img_train[:,:,0,:,:] + 0.5870*img_train[:,:,1,:,:] + 0.1140*img_train[:,:,2,:,:]
	# convert to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
	img_train = img_train.view(img_train.size()[0], -1, \
							   img_train.size()[-2], img_train.size()[-1]) / 255.

	#augment
	img_train = transform(img_train)

	C = 1 if gray_mode else 3
	# extract ground truth (central frame)
	gt_train = img_train[:, ctrl_fr_idx*C:ctrl_fr_idx*C+C, :, :]
	return img_train, gt_train

def init_logging(argdict):
	"""Initilizes the logging and the SummaryWriter modules
	"""
	if not os.path.exists(argdict['log_dir']):
		os.makedirs(argdict['log_dir'])
	writer = SummaryWriter(argdict['log_dir'])
	logger = init_logger(argdict['log_dir'], argdict)
	return writer, logger

def get_imagenames(seq_dir, pattern=None):
	""" Get ordered list of filenames
	"""
	files = []
	for typ in IMAGETYPES:
		files.extend(glob.glob(os.path.join(seq_dir, typ)))

	# filter filenames
	if not pattern is None:
		ffiltered = []
		ffiltered = [f for f in files if pattern in os.path.split(f)[-1]]
		files = ffiltered
		del ffiltered

	# sort filenames alphabetically
	files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	return files

def open_sequence(seq_dir, gray_mode, expand_if_needed=False, max_num_fr=100):
	r""" Opens a sequence of images and expands it to even sizes if necesary
	Args:
		fpath: string, path to image sequence
		gray_mode: boolean, True indicating if images is to be open are in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
		max_num_fr: maximum number of frames to load
	Returns:
		seq: array of dims [num_frames, C, H, W], C=1 grayscale or C=3 RGB, H and W are even.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
	"""
	# Get ordered list of filenames
	files = get_imagenames(seq_dir)

	seq_list = []
	print("\tOpen sequence in folder****: ", seq_dir)
	for fpath in files[0:max_num_fr]:

		img, expanded_h, expanded_w = open_image(fpath,\
												   gray_mode=gray_mode,\
												   expand_if_needed=expand_if_needed,\
												   expand_axis0=False,\
												   tile_axis0=True)
		seq_list.append(img)
		seq = np.stack(seq_list, axis=0)
	return seq, expanded_h, expanded_w

def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, tile_axis0=True, normalize_data=True):
	r""" Opens an image and expands it if necesary
	Args:
		fpath: string, path of image file
		gray_mode: boolean, True indicating if image is to be open
			in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
	Returns:
		img: image of dims NxCxHxW, N=1, C=1 grayscale or C=3 RGB, H and W are even.
			if expand_axis0=False, the output will have a shape CxHxW.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
	"""
	if not gray_mode:
		# Open image as a [H,W,C] torch.Tensor
		img = cv2.imread(fpath)
		# from [H,W,C] to [C,H,W], RGB image
		img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
	else:
		# from [H,W] to [H,W,C] grayscale image (C=1)
		img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE) # [H,W] from cv2
		# expand dimensions from [H,W] to [C,H,W] (C=1)
		img = np.expand_dims(img, 0) 

	if expand_axis0:
		img = np.expand_dims(img, 0)

	# Handle odd sizes
	expanded_h = False
	expanded_w = False
	sh_im = img.shape
	if expand_if_needed:
		if sh_im[-2]%2 == 1:
			expanded_h = True
			if expand_axis0:
				img = np.concatenate((img, \
					img[:, :, -1, :][:, :, np.newaxis, :]), axis=2)
			else:
				img = np.concatenate((img, \
					img[:, -1, :][:, np.newaxis, :]), axis=1)


		if sh_im[-1]%2 == 1:
			expanded_w = True
			if expand_axis0:
				img = np.concatenate((img, \
					img[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
			else:
				img = np.concatenate((img, \
					img[:, :, -1][:, :, np.newaxis]), axis=2)

	if normalize_data:
		img = normalize(img)
	return img, expanded_h, expanded_w

def batch_psnr(img, imclean, data_range):
	r"""
	Computes the PSNR along the batch dimension (not pixel-wise)

	Args:
		img: a `torch.Tensor` containing the restored image
		imclean: a `torch.Tensor` containing the reference image
		data_range: The data range of the input image (distance between
			minimum and maximum possible values). By default, this is estimated
			from the image data-type.
	"""
	img_cpu = img.data.cpu().numpy().astype(np.float32)
	imgclean = imclean.data.cpu().numpy().astype(np.float32)
	psnr = 0
	for i in range(img_cpu.shape[0]):
		# psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
		# 			   data_range=data_range)
		psnr += compare_psnr(imgclean[i], img_cpu[i], \
					   data_range=data_range)

	return psnr/img_cpu.shape[0]

def variable_to_cv2_image(invar, conv_rgb_to_bgr=True):
	r"""Converts a torch.autograd.Variable to an OpenCV image

	Args:
		invar: a torch.autograd.Variable
		conv_rgb_to_bgr: boolean. If True, convert output image from RGB to BGR color space
	Returns:
		a HxWxC uint8 image
	"""
	assert torch.max(invar) <= 1.0

	size4 = len(invar.size()) == 4
	if size4:
		nchannels = invar.size()[1]
	else:
		nchannels = invar.size()[0]

	if nchannels == 1:
		if size4:
			res = invar.data.cpu().numpy()[0, 0, :]
		else:
			res = invar.data.cpu().numpy()[0, :]
		res = (res*255.).clip(0, 255).astype(np.uint8)
	elif nchannels == 3:
		if size4:
			res = invar.data.cpu().numpy()[0]
		else:
			res = invar.data.cpu().numpy()
		res = res.transpose(1, 2, 0)
		res = (res*255.).clip(0, 255).astype(np.uint8)
		if conv_rgb_to_bgr:
			res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
	else:
		raise Exception('Number of color channels not supported')
	return res

def get_git_revision_short_hash():
	r"""Returns the current Git commit.
	"""
	return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def init_logger(log_dir, argdict):
	r"""Initializes a logging.Logger to save all the running parameters to a
	log file

	Args:
		log_dir: path in which to save log.txt
		argdict: dictionary of parameters to be logged
	"""
	from os.path import join

	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(log_dir, 'log.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	try:
		logger.info("Commit: {}".format(get_git_revision_short_hash()))
	except Exception as e:
		logger.error("Couldn't get commit number: {}".format(e))
	logger.info("Arguments: ")
	for k in argdict.keys():
		logger.info("\t{}: {}".format(k, argdict[k]))

	return logger

def init_logger_test(result_dir):
	r"""Initializes a logging.Logger in order to log the results after testing
	a model

	Args:
		result_dir: path to the folder with the denoising results
	"""
	from os.path import join

	logger = logging.getLogger('testlog')
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(result_dir, 'log.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger

def close_logger(logger):
	'''Closes the logger instance
	'''
	x = list(logger.handlers)
	for i in x:
		logger.removeHandler(i)
		i.flush()
		i.close()

def normalize(data):
	r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
	"""
	return np.float32(data/255.)

def svd_orthogonalization(lyr):
	r"""Applies regularization to the training by performing the
	orthogonalization technique described in the paper "An Analysis and Implementation of
	the FFDNet Image Denoising Method." Tassano et al. (2019).
	For each Conv layer in the model, the method replaces the matrix whose columns
	are the filters of the layer by new filters which are orthogonal to each other.
	This is achieved by setting the singular values of a SVD decomposition to 1.

	This function is to be called by the torch.nn.Module.apply() method,
	which applies svd_orthogonalization() to every layer of the model.
	"""
	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		weights = lyr.weight.data.clone()
		c_out, c_in, f1, f2 = weights.size()
		dtype = lyr.weight.data.type()

		# Reshape filters to columns
		# From (c_out, c_in, f1, f2)  to (f1*f2*c_in, c_out)
		weights = weights.permute(2, 3, 1, 0).contiguous().view(f1*f2*c_in, c_out)

		try:
			# SVD decomposition and orthogonalization
			mat_u, _, mat_v = torch.svd(weights)
			weights = torch.mm(mat_u, mat_v.t())

			lyr.weight.data = weights.view(f1, f2, c_in, c_out).permute(3, 2, 0, 1).type(dtype)
		except:
			pass
	else:
		pass

def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary


	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = v

	return new_state_dict
