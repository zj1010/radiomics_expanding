from torch.utils import data
from dataset import from_pth2array, read_nii2array4miccai_pancreas
import numpy as np


class my_dataset(data.Dataset):
	def __init__(self, dataset, dataset_key, aim_shape = None):
		self.dataset = dataset
		self.dataset_name = dataset['dataset_name']
		self.dataset_key = dataset_key
		self.aim_shape = aim_shape
	def __len__(self):
		return self.dataset[self.dataset_key+'_num']
	def __getitem__(self, index):
		# 取出一个sample --------------------------------------------------------------------
		sample = self.dataset[self.dataset_key][index]


		# 读取图像（不同数据集有不同的读取方法，但是读出来都是array就对了 -------------------
		if self.dataset_name in ['MNIST','CIFAR100','CIFAR10']:
			# 以上数据集默认直接把图像读取进ram，故不存在后面读取图像
			pass
		elif self.dataset_name in ['imagenette','imagewoof','CIFAR10']:
			if sample['img'] is None:
				img = from_pth2array(path = sample['img_path'],
									 dataset_name = self.dataset_name,
									 aim_shape=self.aim_shape,
									 order=2)
			else:
				img = sample['img']
		elif self.dataset_name in ['miccai2018pancreas','NIH_pancreas']:
			# 这两个胰腺数据集因为预处理比较费时间，所以必须先读到内存里，不支持在线读取！
			if sample['img'] is None:
				raise ValueError('must be preloaded')
		# todo 为了适应不同数据集，有不同的resize方法（待补充）
		# 不同数据集有不同的返回形式
		if self.dataset_name  in ['MNIST', 'CIFAR10', 'imagenette' ,'imagewoof']:
			out_dict = dict(
				img = img,
				img_path = sample['img_path'],
				label = sample['label'],
				)
		elif self.dataset_name == 'CIFAR100':
			out_dict = dict(
				img = img,
				img_path = sample['img_path'],
				coarse_label = sample['coarse_label'],
				fine_label = sample['fine_label'],
				)
		elif self.dataset_name in ['miccai2018pancreas', 'NIH_pancreas','LiTS2017','MedSeg_liver','pancreas','abdomen_1k','epityphlon']:
			if sample['pre_mask'] is not None:
				out_dict = dict(
					img=sample['img'],
					mask=sample['mask'],
					pre_mask=sample['pre_mask'],
					img_path=sample['img_path'],
					mask_path=sample['mask_path'],
					pre_mask_path=sample['pre_mask_path']
				)
			else:
				out_dict = dict(
					img=sample['img'],
					mask=np.array(sample['mask'], dtype='uint8'),
					img_path=sample['img_path'],
					mask_path=sample['mask_path'],
				)


		return out_dict


def get_dataloader(
		dataset,
		dataset_key,
		batch_size,
		num_workers=0,
		pin_memory=False,
		drop_last=True,
		aim_shape = None,
		shuffle=True):
	dataset = my_dataset(dataset =dataset, dataset_key = dataset_key, aim_shape = aim_shape)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=shuffle,
								  num_workers=num_workers,
								  drop_last=drop_last,
								  pin_memory=pin_memory)
	return data_loader


if __name__ == '__main__':
	import config_runtime as config
	from dataset import  get_dataset
	# 读取数据集，构建dataloader
	dataset = get_dataset(
		d_name='imagenette',
		orsize=160,
		preload_cache=False,  # 提前加载到内存会快一些
		aim_shape=32,
		order=0,
		label_level=0
	)
	train_dataloader = get_dataloader(
		dataset=dataset,
		dataset_key='train_set',
		batch_size=config['batch_size'],
		drop_last=True,
		aim_shape=160,
	)
