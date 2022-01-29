"""
Trains a FastDVDnet model.

Copyright (C) 2019, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# from apex import amp # [optional] nicholas automated mixed precision (AMP) training from NVIDIA apex
from models import FastDVDnet
from dataset import ValDataset
# from dataloaders import train_dali_loader #nicholas
from utils import svd_orthogonalization, close_logger, init_logging, normalize_augment
from train_common import resume_training, lr_scheduler, log_train_psnr, \
                    validate_and_log, save_model_checkpoint
from torchvision import datasets, transforms
import os

def main(**args):
    r"""Performs the main training loop
    """
    gray_mode = args['gray'] # gray mode indicator
    C = 1 if gray_mode else 3 # number of color channels

    # Load dataset
    print('> Loading datasets ...')
    # dataset_val = ValDataset(valsetdir=args['valset_dir'], gray_mode=gray_mode) # for grayscale/color video nicholas
    # dataset_val = ValDataset(valsetdir=args['valset_dir'], gray_mode=False) # for color videos only

    # loader_train = train_dali_loader(batch_size=args['batch_size'],\
    #                                 file_root=args['trainset_dir'],\
    #                                 sequence_length=args['temp_patch_size'],\
    #                                 crop_size=args['patch_size'],\
    #                                 epoch_size=args['max_number_patches'],\
    #                                 random_shuffle=True,\
    #                                 temp_stride=3,\
    #                                 gray_mode=gray_mode)
    
    # rootpath = 'D:\MyWS\3SCIYuan\packages\fastdvdnet\DAVIS\JPEGImages\self\train'
    trainpath = 'D:/MyWS/3SCIYuan/packages/fastdvdnet/DAVIS/JPEGImages/self/train'
    valpath = 'D:/MyWS/3SCIYuan/packages/fastdvdnet/DAVIS/JPEGImages/self/val'
    # train_dataset = datasets.ImageFolder(trainpath) # 不做transform    
    # train_Dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=args['batch_size'],shuffle=True)
    train_Dataloader = ValDataset(valsetdir=trainpath, gray_mode=gray_mode) # for grayscale/color video
    dataset_val = ValDataset(valsetdir=valpath, gray_mode=gray_mode) # for grayscale/color video

    num_minibatches = int(args['max_number_patches']//args['batch_size'])
    ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
    print("\t# of training samples: %d\n" % int(args['max_number_patches']))

    # Init loggers
    writer, logger = init_logging(args)

    # Define GPU devices
    device_ids = [0]
    torch.backends.cudnn.benchmark = True # CUDNN optimization

    # Create model
    model = FastDVDnet(num_color_channels=C)
    
    model = model.cuda()

    # Define loss
    criterion = nn.MSELoss(reduction='sum')
    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    # [AMP initialization] automated half-precision training
    if args['fp16']:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['amp_opt_level'])
    
    # model = nn.DataParallel(model, device_ids=device_ids).cuda()
    model = nn.DataParallel(model)

    # Resume training or start anew
    start_epoch, training_params = resume_training(args, model, optimizer)

    # Training
    start_time = time.time()
    for epoch in range(start_epoch, args['epochs']):
        # Set learning rate
        current_lr, reset_orthog = lr_scheduler(epoch, args)
        if reset_orthog:
            training_params['no_orthog'] = True

        # set learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('\nlearning rate %f' % current_lr)

        # train

        for i, data in enumerate(train_Dataloader): #(loader_train, 0)

            # Pre-training step
            model.train()

            # When optimizer = optim.Optimizer(net.parameters()) we only zero the optim's grads
            optimizer.zero_grad()

            # convert inp to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
            # extract ground truth (central frame)
            img_train, gt_train = normalize_augment(data[0]['data'], ctrl_fr_idx, gray_mode)
            N, _, H, W = img_train.size()

            # std dev of each sequence
            stdn = torch.empty((N, 1, 1, 1)).cuda().uniform_(args['noise_ival'][0], to=args['noise_ival'][1])
            # draw noise samples from std dev tensor
            noise = torch.zeros_like(img_train)
            noise = torch.normal(mean=noise, std=stdn.expand_as(noise))

            #define noisy input
            imgn_train = img_train + noise

            # Send tensors to GPU
            gt_train = gt_train.cuda(non_blocking=True)
            imgn_train = imgn_train.cuda(non_blocking=True)
            noise = noise.cuda(non_blocking=True)
            noise_map = stdn.expand((N, 1, H, W)).cuda(non_blocking=True) # one channel per image

            # Evaluate model and optimize it
            out_train = model(imgn_train, noise_map)

            # Compute loss
            loss = criterion(gt_train, out_train) / (N*2)

            # [AMP scale loss to avoid overflow of float16] automated mixed precision training
            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
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
                                i, \
                                num_minibatches, \
                                training_params)
            # update step counter
            training_params['step'] += 1
        
        # save model and checkpoint
        training_params['start_epoch'] = epoch + 1
        save_model_checkpoint(model, args, optimizer, training_params, epoch)

        # Call to model.eval() to correctly set the BN layers before inference
        model.eval()

        # Validation and log images
        validate_and_log(
                        model_temp=model, \
                        dataset_val=dataset_val, \
                        valnoisestd=args['val_noiseL'], \
                        temp_psz=args['temp_patch_size'], \
                        writer=writer, \
                        epoch=epoch, \
                        lr=current_lr, \
                        logger=logger, \
                        trainimg=img_train, \
                        gray_mode=gray_mode
                        )

    # Print elapsed time
    elapsed_time = time.time() - start_time
    print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    # Close logger file
    close_logger(logger)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the video denoiser")

    #Training parameters
    parser.add_argument("--batch_size", type=int, default=64, 	\
                     help="Training batch size")
    parser.add_argument("--epochs", "--e", type=int, default=80, \
                     help="Number of total training epochs")
    parser.add_argument("--resume_training", "--r", action='store_true',\
                        help="resume training from a previous checkpoint")
    parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60], \
                        help="When to decay learning rate; should be lower than 'epochs'")
    parser.add_argument("--lr", type=float, default=1e-3, \
                     help="Initial learning rate")
    parser.add_argument("--no_orthog", action='store_true',\
                        help="Don't perform orthogonalization as regularization")
    parser.add_argument("--save_every", type=int, default=10,\
                        help="Number of training steps to log psnr and perform \
                        orthogonalization")
    parser.add_argument("--save_every_epochs", type=int, default=5,\
                        help="Number of training epochs to save state")
    parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 55], \
                     help="Noise training interval")
    parser.add_argument("--val_noiseL", type=float, default=25, \
                        help='noise level used on validation set')
    # Preprocessing parameters
    parser.add_argument("--patch_size", "--p", type=int, default=96, help="Patch size")
    parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
    parser.add_argument("--max_number_patches", "--m", type=int, default=256000, \
                        help="Maximum number of patches")
    # Dirs
    parser.add_argument("--log_dir", type=str, default="logs", \
                     help='path of log files')
    parser.add_argument("--trainset_dir", type=str, default=None, \
                     help='path of trainset')
    parser.add_argument("--valset_dir", type=str, default=None, \
                         help='path of validation set')
    parser.add_argument("--gray", action='store_true',\
                        help='force grayscale (C=1) video in training and validation')
    parser.add_argument("--fp16", action='store_true',\
                        help='half-precision (float16) [mixed precision] training (for Turing graphics cards)')
    parser.add_argument('--amp_opt_level', type=str, default='O1', \
                         help='amp opt_level, default="O1"')
    argspar = parser.parse_args()

    # Normalize noise between [0, 1]
    argspar.val_noiseL /= 255.
    argspar.noise_ival[0] /= 255.
    argspar.noise_ival[1] /= 255.

    print("\n### Training FastDVDnet denoiser model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
