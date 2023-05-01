# step0: import库 -----------------------------------------------------------------------
from wama.utils import *
from wama.data_augmentation import aug3D
from sklearn.model_selection import StratifiedKFold,KFold
from dataloader import get_dataloader
from wama.utils import *
from wama_modules.utils import load_weights
import random
import os
sep = os.sep

config = dict(
	random_seed = 202020202,

	input_size = [96,128,128],
	batch_size = 1,
	batch_size_test = 2,

	# 学习率与优化器相关参数
	lr_min = 1e-12,
	lr_max = 1e-4,
	epoch_cosinedecay = 400,
	epoch_warmup = 5,
	adam_beta1 = 0.5,
	adam_beta2 = 0.999,

	# 其他参数
	epoch_num = 400,
	save_pth = r'/data/newnas/ZJ/domain_poject/MAILAB_demo/pancreas_segmentation/C3D_backbone_result/traditional',
	gpu_device_index = 0,
	aug_p = 0.6,


	val_flag = True,
	val_save_img = True,
	val_start_epoch = 1,
	val_step_epoch = 1,

	test_flag = True,
	test_save_img = True,
	test_start_epoch = 1,
	test_step_epoch = 1,

	model_save_step = 1,

	logger_print2pic_step_iter = 2,  # 将训练过程中的loss和测试验证指标保存为图片的时间步长

)

# step1: 设置随机数种子
seed = config['random_seed']
random.seed(seed)
os.environ["PYTHONASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False



# step2: device -------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_device_index'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# step3: 读取数据集，构建dataloader -------------------------------------------------------
# dataset_test,_ = load_from_pkl(r"/data/newnas/ZJ/epityphlons_pkl/test_pre_order0_128_128_96_new.pkl")#NIH_pancreas_liver
dataset = load_from_pkl(r"/data/newnas/ZJ/Kidney/tumor layer/train/pre_order0_128_128_96_new.pkl")
# test_dataloader = get_dataloader(dataset=dataset_test,dataset_key='train_set',batch_size=config['batch_size_test'], drop_last=False)
i=0
kf = KFold(n_splits=5,shuffle=True)
train_list = []
val_list = []
for train_index, val_index in kf.split(dataset['train_set']):
	train_list.append(train_index)
	val_list.append(val_index)

for i in range(0,5):
	dataset_train = dict(train_set_num=0, dataset_name='pancreas')
	dataset_val = dict(train_set_num=0, dataset_name='pancreas')
	dataset_train['train_set'] = [dataset['train_set'][n] for n in train_list[i]]
	dataset_train['train_set_num'] = len(dataset_train['train_set'])
	dataset_val['train_set'] = [dataset['train_set'][n] for n in val_list[i]]
	dataset_val['train_set_num'] = len(dataset_val['train_set'])

	train_dataloader = get_dataloader(dataset=dataset_train, dataset_key='train_set', batch_size=config['batch_size'],
									  drop_last=False)
	val_dataloader = get_dataloader(dataset=dataset_val, dataset_key='train_set', batch_size=config['batch_size_test'],
									drop_last=False)
	print('dataset_train:', dataset_train['dataset_name'], dataset_train['train_set_num'])
	print('dataset_val:', dataset_val['dataset_name'], dataset_val['train_set_num'])
	# print('dataset_test:', dataset_test['dataset_name'], dataset_test['train_set_num'])
	# step4: 构建网络 -------------------------------------------------------
	# [32, 64, 128, 256, 512]

	# step9: 训练&验证&测试 --------------------------------------------------------------
	score = -99999
	best_score = -9999
	init_epoch = 0
	current_iter = 0
	auger = aug3D(size=config['input_size'], deformation_scale=0.25)
	print("当前折数 ",str(i+1))
	aug_list = []
		# 训练
	while(len(aug_list)<500):
		for iter, sample in enumerate(train_dataloader):  # 读取训练集图像
			current_iter = current_iter + 1
			# 取出mask和scan
			scan, mask, pre_mask = [tensor2numpy(sample['img']), tensor2numpy(sample['mask']),
									tensor2numpy(sample['pre_mask'])]

			scan = np.expand_dims(scan, axis=1)
			mask = np.expand_dims(mask, axis=1)
			pre_mask = np.expand_dims(pre_mask, axis=1)

			# 一定幾率扩增scan，并将变换应用到mask
			p_transform = random.random()
			if p_transform <= config['aug_p']:
				aug_result = auger.aug(dict(data=scan, seg=mask, pre_mask=pre_mask))  # 注意要以字典形式传入
				scan = aug_result['data']
				mask = aug_result['seg']
				pre_mask = aug_result['pre_mask']

				scan = scan[0][0]
				mask = mask[0][0]
				pre_mask = pre_mask[0][0]
				aug_list.append(dict(
					img=scan,
					mask=mask,
					pre_mask=pre_mask,
					img_path=sample['img_path'][0],
					mask_path=sample['mask_path'][0],
					pre_mask_path=sample['pre_mask_path'][0]
				))
			print("目前已扩增数量 ", len(aug_list))
			if len(aug_list) >= 500:
				new_train = aug_list + dataset_train['train_set']
				train_a = dict(train_set=new_train, train_set_num=len(new_train), dataset_name='pancreas')
				val_a = dict(train_set=dataset_val['train_set'], train_set_num=len(dataset_val['train_set']),
							 dataset_name='pancreas')
				train_path = r'/data/newnas/ZJ/Kidney/tumor layer/fold' + str(i + 1)+ '/train'
				val_path  = r'/data/newnas/ZJ/Kidney/tumor layer/fold' + str(i + 1)+ '/val'
				if not os.path.exists(train_path):
					os.makedirs(train_path)
					os.makedirs(val_path)
				save_as_pkl(train_path+'/pre_order0_128_128_96_new.pkl',
							train_a)
				save_as_pkl(
					val_path+'/pre_order0_128_128_96_new.pkl',
					val_a)
				break

























