import numpy as np
import torch
from torch.utils.data import Dataset
from dnri.utils import data_utils


class BasketballData(Dataset):
	def __init__(self, name, data_path, mode, params, test_full=False, num_in_path=True, has_edges=True, transpose_data=True, max_len=None):
		self.name = name
		self.data_path = data_path
		self.mode = mode
		self.params = params
		self.num_in_path = num_in_path
		# Get preprocessing stats.
		loc_max, loc_min = self._get_normalize_stats()
		self.loc_max = loc_max
		self.loc_min = loc_min
		self.test_full = test_full
		self.max_len = max_len

		# Load data.
		self._load_data(transpose_data)
		
	def __getitem__(self, index):
		if self.max_len is not None:
			inputs = self.feat[index, :self.max_len]
		else:
			inputs = self.feat[index]
		return {'inputs': inputs}

	def __len__(self, ):
		return self.feat.shape[0]

	def _get_normalize_stats(self):
		train_loc = np.load(self._get_npy_path('loc', 'train'))['data'][:79456]
		return train_loc.max(), train_loc.min()

	def _load_data(self, transpose_data):
		# Load data
		# CHECK: Perform train val split if self.mode in ['train', 'valid']
		if self.mode in ['train', 'valid']:
			if self.mode == 'train':
				self.loc_feat = np.load(self._get_npy_path('loc', self.mode))['data'][:79456]
			else:
				self.loc_feat = np.load(self._get_npy_path('loc', self.mode))['data'][79456:]
		else:
			self.loc_feat = np.load(self._get_npy_path('loc', self.mode))['data']

		# Perform preprocessing.
		self.loc_feat = data_utils.normalize(self.loc_feat, self.loc_max, self.loc_min)
		

		# Reshape [num_sims, num_timesteps, num_agents, num_dims]
		self.feat = self.loc_feat.reshape((-1, 50, 11, 2))
		# if transpose_data:
		# 	self.loc_feat = np.transpose(self.loc_feat, [0, 1, 3, 2])
		# self.feat = np.concatenate([self.loc_feat], axis=-1)

		# Convert to pytorch cuda tensor.
		self.feat = torch.from_numpy(
				np.array(self.feat, dtype=np.float32))  # .cuda()

		# Exclude self edges.
		num_atoms = self.params['num_agents']
		off_diag_idx = np.ravel_multi_index(
				np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
				[num_atoms, num_atoms])

	def _get_npy_path(self, feat, mode):
		if self.num_in_path:
			raise ValueError('Provided data path is incorrect.')
		elif self.mode in ['train', 'valid']:
			path = f'{self.data_path}train.npz'
		else: 
			path = f'{self.data_path}test.npz'
		return path