"""
Dataset related functions

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import glob
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from utils import open_sequence,get_patch,open_sequence_train

NUMFRXSEQ_VAL = 5	# number of frames of each sequence to include in validation dataset
VALSEQPATT = '*' # pattern for name of validation sequence

NUMFRXSEQ_TRAIN = 5	# number of frames of each sequence to include in TRAINidation dataset
TRAINSEQPATT = '*' # pattern for name of validation sequence
reduce = 0
def shuffle_frame(train_data, num_frames,C, H, W):
	# 1920 1080
    # [num_frames, C, H, W]
	
    processed_data = []
    for i in range(num_frames//NUMFRXSEQ_TRAIN):
        processed_data.append(train_data[NUMFRXSEQ_TRAIN*i:NUMFRXSEQ_TRAIN*i+5,:,:,:]) 

    return processed_data

class TrainDataset(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder on memory.
	"""
	def __init__(self, train_files_names, trainsetdir=None, gray_mode=False, num_input_frames=NUMFRXSEQ_TRAIN):
		self.gray_mode = gray_mode
		split_file = train_files_names + 'train.txt'
		f = open(split_file,'r')
		TRAINSEQPATT = f.readlines()
		f.close()
		# Look for subdirs with individual sequences
		seqs_dirs = []
		for i in range(len(TRAINSEQPATT)-reduce):
			file = os.path.join(trainsetdir, TRAINSEQPATT[i][:-1])
			seqs_dirs.append(file)

		# open individual sequences and append them to the sequence list
		seqs_dirs.sort()
		# seqs_dirs = seqs_dirs[:8]
		sequences = []
		for seq_dir in seqs_dirs:
			seq, _, _ = open_sequence_train(seq_dir, gray_mode, expand_if_needed=False, \
							 max_num_fr=num_input_frames)
			# seq is [num_frames, C, H, W]
			num_frames, C, H, W = seq.shape
			seq = shuffle_frame(seq,num_frames,C, H, W)
			sequences.extend(seq)

			# num_frames, C, H, W = seq.shape
			# patch_size = 128
			# n_len = (H//patch_size)*(W//patch_size)

			# index = np.random.choice(range(num_frames), num_frames)
			# processed_data = np.zeros((num_frames, C,patch_size, patch_size), dtype=np.float32)
			# for i in range(n_len):
			# 	# samples = shuffle_crop(seq,num_frames=num_frames,patch_size=patch_size)
			# 	x_index = np.random.randint(0, H - patch_size)
			# 	y_index = np.random.randint(0, W - patch_size)
			# 	processed_data = seq[:,:,x_index:x_index + patch_size, y_index:y_index + patch_size] 
			# 	# gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
			# 	sequences.append(processed_data)

		self.sequences = sequences

	def __getitem__(self, index):
		samples = torch.from_numpy(self.sequences[index])
		samples = get_patch(samples) #[num_frames, C, H, W]
		return samples

	def __len__(self):
		return len(self.sequences)

class ValDataset(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder on memory.
	"""
	def __init__(self, val_files_names, valsetdir=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL):
		self.gray_mode = gray_mode
		split_file = val_files_names + 'test.txt'
		f = open(split_file, 'r')
		VALSEQPATT = f.readlines()
		f.close()
		# Look for subdirs with individual sequences
		seqs_dirs = []
		for i in range(len(VALSEQPATT)):
			file = os.path.join(valsetdir, VALSEQPATT[i][:-1])
			seqs_dirs.append(file)

		# open individual sequences and append them to the sequence list
		seqs_dirs.sort()
		# seqs_dirs = seqs_dirs[:5]
		# Look for subdirs with individual sequences

		# open individual sequences and append them to the sequence list
		sequences = []
		for seq_dir in seqs_dirs:
			seq, _, _ = open_sequence(seq_dir, gray_mode, expand_if_needed=False, \
							 max_num_fr=num_input_frames)
			# seq is [num_frames, C, H, W]
			sequences.append(seq)
			# num_frames, C, H, W = seq.shape
			# seq = shuffle_frame(seq,num_frames,C, H, W)
			# sequences.extend(seq)

		self.sequences = sequences

	def __getitem__(self, index):
		return torch.from_numpy(self.sequences[index])

	def __len__(self):
		return len(self.sequences)

