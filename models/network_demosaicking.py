"""
Definition of the FastDVDnet model

Copyright (C) 2019, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""

from matplotlib.pyplot import xlim
import torch
import torch.nn as nn
from torch.nn import functional as F
import functools

import numpy as np

from utils.utils_image import fourCh2OneCh,oneCh2FourCh

base_layer = 20
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())

    return nn.Sequential(*layers)


class CvBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(CvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			# nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			# nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)
class InputCvBlock_2(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch,ch_each_frame=3):
		super(InputCvBlock_2, self).__init__()
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames*(ch_each_frame), num_in_frames*self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			# nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			# nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)


class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames*(3+1), num_in_frames*self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			# nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			# nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class DownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
			# nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			CvBlock(out_ch, out_ch)
		)

	def forward(self, x):
		return self.convblock(x)

class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch),
			nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2)
		)

	def forward(self, x):
		return self.convblock(x)

class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			# nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		return self.convblock(x)

class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, 48, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(48, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=False)
        out = self.conv2(out)
        return identity +  out
class encoder(nn.Module):
    def __init__(self, nf=64, N_RB=5):
        super(encoder, self).__init__()
        RB_f = functools.partial(ResidualBlock_noBN, nf=nf)
        self.conv_first01 =  torch.nn.Sequential(
            nn.Conv2d(4, nf*2, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False)
        )
        
        self.conv_first02 = torch.nn.Sequential(
            nn.Conv2d(nf, nf*2, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False)
        )

        self.conv_first = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.rbs = make_layer(RB_f, N_RB)

        self.d2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.d2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.d4_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.d4_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.py_conv = nn.Conv2d(nf*3, nf, 7, 1, 3, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        x = self.lrelu(self.conv_first01(x))
        x = self.lrelu(self.conv_first02(x))

        fea = self.lrelu(self.conv_first(x))
        fea_lr = self.rbs(fea)

        fea_d2 = self.lrelu(self.d2_conv2(self.lrelu(self.d2_conv1(fea_lr))))
        fea_d4 = self.lrelu(self.d4_conv2(self.lrelu(self.d4_conv1(fea_d2))))
		

        fea_d2 = F.interpolate(fea_d2, size =(x.size()[-2],x.size()[-1]) , mode='bilinear', align_corners=False)
        fea_d4 = F.interpolate(fea_d4, size =(x.size()[-2],x.size()[-1]) , mode='bilinear', align_corners=False)

        out = self.lrelu(self.py_conv(torch.cat([fea_lr, fea_d2, fea_d4],1)))
        return out

class DenBlock(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3,ch_each_frame=3):
		super(DenBlock, self).__init__()
	
		self.chs_lyr0 = base_layer
		self.chs_lyr1 = base_layer*2
		self.chs_lyr2 = base_layer*4

		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		
		self.inc_1 = InputCvBlock_2(num_in_frames=num_input_frames, out_ch=self.chs_lyr0,ch_each_frame=ch_each_frame)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		# self.csSELayer = csSELayer(self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

		# self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2, noise_map=None):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		if noise_map==None:
			x0 = self.inc_1(torch.cat((in0,in1,in2), dim=1))
		else:
			x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# x2 = self.csSELayer(x2)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)

		# Residual
		x = in1 + x

		return x
class DenBlock1ChBayer(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3,ch_each_frame=3):
		super(DenBlock1ChBayer, self).__init__()
	
		self.chs_lyr0 = base_layer
		self.chs_lyr1 = base_layer*2
		self.chs_lyr2 = base_layer*4

		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		
		self.inc_1 = InputCvBlock_2(num_in_frames=num_input_frames, out_ch=self.chs_lyr0,ch_each_frame=ch_each_frame)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		# self.csSELayer = csSELayer(self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=1)
		self.fusion = OutputCvBlock(in_ch=1, out_ch=3)

		# self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2, noise_map=None):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		if noise_map==None:
			x0 = self.inc_1(torch.cat((in0,in1,in2), dim=1))
		else:
			x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# x2 = self.csSELayer(x2)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)

		# Residual
		x = in1 + x
		x = self.fusion(x)

		return x

class DenBlock4ChBayer(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3,ch_each_frame=4):
		super(DenBlock4ChBayer, self).__init__()

		self.chs_lyr0 = base_layer
		self.chs_lyr1 = base_layer*2
		self.chs_lyr2 = base_layer*4

		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		
		self.inc_1 = InputCvBlock_2(num_in_frames=num_input_frames, out_ch=self.chs_lyr0,ch_each_frame=ch_each_frame)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		# self.csSELayer = csSELayer(self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=4)
		self.upscale = nn.UpsamplingBilinear2d(scale_factor=2)
		self.fusion = OutputCvBlock(in_ch=4, out_ch=3)
		

		# self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2, noise_map=None):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		if noise_map==None:
			x0 = self.inc_1(torch.cat((in0,in1,in2), dim=1))
		else:
			x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# x2 = self.csSELayer(x2)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)

		# Residual
		x = in1 + x
		x = self.upscale(x)
		x = self.fusion(x)

		return x

class DDnet(nn.Module):
	""" 
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=5):
		super(DDnet, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each denoising stage
		# self.temp1 = DenBlock1ChBayer(num_input_frames=3,ch_each_frame=1)
		self.temp1 = DenBlock(num_input_frames=3,ch_each_frame=1)
		self.temp2 = DenBlock(num_input_frames=3)

		self.temp11 = DenBlock4ChBayer(num_input_frames=3,ch_each_frame=4)
		# self.temp21 = DenBlock4ChBayer(num_input_frames=3,ch_each_frame=4)
		# self.upscale = nn.C
		# self.fusion2 = FusionBlock(in_ch=3*2,out_ch=3)
		# Init weights
		# self.reset_params()
		self.weight_tensor_in = nn.parameter.Parameter(torch.ones((9,1,1,1,1)))
		self.weight_tensor_in2 = nn.parameter.Parameter(torch.ones((9,1,4,1,1)))
		self.weight_tensor_out = nn.parameter.Parameter(torch.ones((2,1,3,1,1)))


	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x, noise_map=None):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		(x0, x1, x2, x3, x4) = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))
		a = self.weight_tensor_in
		a2 = self.weight_tensor_in2
		a3 = self.weight_tensor_out

		x2_init = x2
		
		x0 = torch.sum(x0,dim=1)
		x1 = torch.sum(x1,dim=1)
		x2 = torch.sum(x2,dim=1)
		x3 = torch.sum(x3,dim=1)
		x4 = torch.sum(x4,dim=1)

		x0_fourCh = oneCh2FourCh(x0.permute(1,2,0)).permute(2,3,0,1)
		x1_fourCh = oneCh2FourCh(x1.permute(1,2,0)).permute(2,3,0,1)
		x2_fourCh = oneCh2FourCh(x2.permute(1,2,0)).permute(2,3,0,1)
		x3_fourCh = oneCh2FourCh(x3.permute(1,2,0)).permute(2,3,0,1)
		x4_fourCh = oneCh2FourCh(x4.permute(1,2,0)).permute(2,3,0,1)

		x0 = x0.unsqueeze(1)
		x1 = x1.unsqueeze(1)
		x2 = x2.unsqueeze(1)
		x3 = x3.unsqueeze(1)
		x4 = x4.unsqueeze(1)




		# First stage
		x20 = self.temp1(x0*a[0], x1*a[1], x2*a[2])#, noise_map)
		x21 = self.temp1(x1*a[3], x2*a[4], x3*a[5])#, noise_map)
		x22 = self.temp1(x2*a[6], x3*a[7], x4*a[8])#, noise_map)

		# First stage
		x20_2 = self.temp11(x0_fourCh*a2[0], x1_fourCh*a2[1], x2_fourCh*a2[2])#, noise_map)
		x21_2 = self.temp11(x1_fourCh*a2[3], x2_fourCh*a2[4], x3_fourCh*a2[5])#, noise_map)
		x22_2 = self.temp11(x2_fourCh*a2[6], x3_fourCh*a2[7], x4_fourCh*a2[8])#, noise_map)



		#Second stage
		x_out1 = self.temp2(x20, x21, x22)#, noise_map)
		x_out2 = self.temp2(x20_2, x21_2, x22_2)#, noise_map)
		x_out = a3[0]*x_out1 + a3[1]*x_out2

		return x_out

	
