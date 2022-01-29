
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
root_path = os.path.dirname(os.path.abspath(__file__))
print(root_path)
os.chdir( root_path )
import time
import argparse
import torch
from tqdm import  tqdm
import imageio as io
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import imageio as io

from models.network_demosaicking import DDnet
from dataset import ValDataset,TrainDataset
# from dataloaders import train_dali_loader
from utils import svd_orthogonalization, close_logger, init_logging, normalize_augment,get_patch
from train_common import resume_training, lr_scheduler, log_train_psnr, \
					validate_and_log, save_model_checkpoint
from torch.utils.data import DataLoader
from utils_mosaic import mosaic_CFA_Bayer_cuda
import tensorboard
def main(**args):
	r"""Performs the main training loop
	"""

	# Load dataset
	print('> Loading datasets ...')
	train_set = TrainDataset(train_files_names=args['dataset_split_files'],trainsetdir=args['trainset_dir'], gray_mode=False)
	loader_train_2 = DataLoader(train_set,batch_size=args['batch_size'], pin_memory=True) #,transforms=[transforms.])

	dataset_val = ValDataset(val_files_names=args['dataset_split_files_val'],valsetdir=args['valset_dir'], gray_mode=False)
	# loader_train = train_dali_loader(batch_size=args['batch_size'],\
	# 								file_root=args['trainset_dir'],\
	# 								sequence_length=args['temp_patch_size'],\
	# 								crop_size=args['patch_size'],\
	# 								epoch_size=args['max_number_patches'],\
	# 								random_shuffle=True,\
	# 								temp_stride=3)

	num_minibatches = int(len(train_set)//args['batch_size'])
	ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
	print("\t# of training samples: %d\n" % int(len(train_set)))

	# Init loggers
	writer, logger = init_logging(args)

	# Define GPU devices
	device_ids = [0,1,2]
	torch.backends.cudnn.benchmark = True # CUDNN optimization

	# Create model
	model = DDnet()
	model = nn.DataParallel(model, device_ids=device_ids).cuda()

	# Define loss
	criterion = nn.MSELoss(reduction='sum')
	criterion.cuda()

	# Optimizer
	optimizer = optim.Adam(model.parameters(), lr=args['lr'])
	# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,10,2,1e-6)

	# Resume training or start anew
	start_epoch, training_params = resume_training(args, model, optimizer)

	# Training
	start_time = time.time()
	max_psnr = 0
	current_lr = args['lr']
	for epoch in range(start_epoch, args['epochs']):
		# Set learning rate
		current_lr, reset_orthog = lr_scheduler(epoch, args)
		if reset_orthog:
			training_params['no_orthog'] = True

		# set learning rate in optimizer
		for param_group in optimizer.param_groups:
			param_group["lr"] = current_lr
		if epoch%50==0:
			print('\nlearning rate %f' % current_lr)

		# train

		for it, data in enumerate(loader_train_2):

			# Pre-training step
			model.train()

			# When optimizer = optim.Optimizer(net.parameters()) we only zero the optim's grads
			optimizer.zero_grad()

			# convert inp to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
			# extract ground truth (central frame)
			img_train, gt_train = normalize_augment(data, ctrl_fr_idx) # 60 5 3 480 800 ,2

			N, _, H, W = img_train.size() # batchsize 5*3 h w

			# std dev of each sequence
			stdn = torch.empty((N, 1, 1, 1)).cuda().uniform_(args['noise_ival'][0], to=args['noise_ival'][1])

			# random number from 5/255 to 55/255
			# draw noise samples from std dev tensor

			# noise = torch.zeros_like(img_train)
			# # if args.use_gpu:
			# noise = noise.cuda()
			img_train = img_train.cuda()
			# noise = torch.normal(mean=noise, std=stdn.expand_as(noise))
			noise = torch.normal(mean=0.0, std=1/255,size=img_train.shape)
			noise = noise.cuda()
			imgn_train = img_train + noise

			for i in range(imgn_train.shape[0]):
				for j in range(imgn_train.shape[1]//3):
					CFA, CFA4, mosaic, mask = mosaic_CFA_Bayer_cuda(imgn_train[i,3*j:3*j+3,:,:].permute(1,2,0))
					imgn_train[i,3*j:3*j+3,:,:] = mosaic.permute(2,0,1)
			# Send tensors to GPUnp

			gt_train = gt_train.cuda(non_blocking=True) +  noise[:,9:12,:,:]
			imgn_train = imgn_train.cuda(non_blocking=True)
			noise = noise.cuda(non_blocking=True)
			noise_map = stdn.expand((N, 1, H, W)).cuda(non_blocking=True) # one channel per image


			# from fastdvdnet import denoise_seq_fastdvdnet_for_train
			# out_train = torch.empty(imgn_train.shape)
			# for i, batch in enumerate(imgn_train):
			# 	out_train[i] = denoise_seq_fastdvdnet(batch, stdn, args['temp_patch_size'], model)
			# temp_patch_size = args['temp_patch_size']
			# out_train = denoise_seq_fastdvdnet_for_train(imgn_train, noise_map, temp_patch_size, model)
			# imgn_train 4 15 h w

			# imgn_train, gt_train = get_patch(imgn_train, gt_train)
			# Evaluate model and optimize it
			out_train = model(imgn_train)
			# out 4 3 h w
			# Compute loss
			loss = criterion(gt_train, out_train) / (N*2)
			#gt 4 3 h w
			loss.backward()
			optimizer.step()

			# Results
			if training_params['step'] % args['save_every'] == 0:
				# Apply regularization by orthogonalizing filters
				if not training_params['no_orthog']:
					model.apply(svd_orthogonalization)

				# Compute training PSNR
				log_train_psnr(out_train, \
								gt_train, \
								loss, \
								writer, \
								epoch, \
								it, \
								num_minibatches, \
								training_params)
			# update step counter
			training_params['step'] += 1

		# Call to model.eval() to correctly set the BN layers before inference
		if epoch % 10 ==0:
			# torch.cuda.empty_cache()
			model.eval()

			# Validation and log images
			psnr_val = validate_and_log(
							model_temp=model, \
							dataset_val=dataset_val, \
							valnoisestd=args['val_noiseL'], \
							temp_psz=args['temp_patch_size'], \
							writer=writer, \
							epoch=epoch, \
							lr=current_lr, \
							logger=logger, \
							trainimg=img_train
							)
			
			# save model and checkpoint
			training_params['start_epoch'] = epoch + 1
			
			model.train()
			if max_psnr<psnr_val:
				max_psnr = psnr_val
				if max_psnr>30:
					save_model_checkpoint(model, args, optimizer, training_params, epoch)


	# Print elapsed time
	elapsed_time = time.time() - start_time
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

	# Close logger file
	close_logger(logger)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train the denoiser")

	#Training parameters

	parser.add_argument("--batch_size", type=int, default=36, 	\
					 help="Training batch size")
	parser.add_argument("--epochs", "--e", type=int, default=1000, \
					 help="Number of total training epochs")
	parser.add_argument("--resume_training", "--r", action='store_true',default=False,
						help="resume training from a previous checkpoint")
	parser.add_argument("--milestone", nargs=2, type=int, default=[200,400,600,800], \
						help="When to decay learning rate; should be lower than 'epochs'")
	parser.add_argument("--lr", type=float, default=1e-3, \
					 help="Initial learning rate")
	parser.add_argument("--no_orthog", action='store_true',\
						help="Don't perform orthogonalization as regularization")
	parser.add_argument("--save_every", type=int, default=200,\
						help="Number of training steps to log psnr and perform \
						orthogonalization")
	parser.add_argument("--save_every_epochs", type=int, default=200,\
						help="Number of training epochs to save state")
	parser.add_argument("--noise_ival", nargs=2, type=int, default=[0.1,1], \
					 help="Noise training interval")
	parser.add_argument("--val_noiseL", type=float, default=1, \
						help='noise level used on validation set')
	parser.add_argument("--use_gpu", "--gpu", action='store_true', default=True, help="use gpu or not")
	# Preprocessing parameters
	parser.add_argument("--patch_size", "--p", type=int, default=4, help="Patch size")
	parser.add_argument("--temp_patch_size", "--tp", type=int, default=6, help="Temporal patch size")
	parser.add_argument("--max_number_patches", "--m", type=int, default=256000, \
						help="Maximum number of patches")
	# Dirs
	parser.add_argument("--log_dir", type=str, default="joint_logs", \
					 help='path of log files')
	parser.add_argument("--trainset_dir", type=str, default='/home/wuzongliang/py/dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/JPEGImages/Full-Resolution/',
					 help='path of trainset')
	parser.add_argument("--valset_dir", type=str, default='/home/wuzongliang/py/dataset/DAVIS-2017-trainval-480p/DAVIS/JPEGImages/480p/',
						 help='path of validation set')
	parser.add_argument("--dataset_split_files", type=str,
						default='/home/wuzongliang/py/dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/ImageSets/2017/',
						help='path of split set file')
	parser.add_argument("--dataset_split_files_val", type=str,
						default='/home/wuzongliang/py/dataset/DAVIS-2017-trainval-480p/DAVIS/ImageSets/2017/',
						help='path of split set file')

	argspar = parser.parse_args()

	# Normalize noise between [0, 1]
	argspar.val_noiseL /= 255.
	argspar.noise_ival[0] /= 255.
	argspar.noise_ival[1] /= 255.

	print("\n### Training DDnet denoiser model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))
