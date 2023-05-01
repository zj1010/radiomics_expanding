"""
代码跟着数据集改动，类似的数据集和任务可以使用同一个代码
胰腺数据代码，暂时只支持NIH和miccai 2018的，后面会支持我们的私有胰腺（or PNENs）数据
"""
# import tensorflow

# step0: import库 -----------------------------------------------------------------------
from wama.utils import *
from losses import SmoothCEloss

from wama.data_augmentation import aug3D
from sklearn.model_selection import StratifiedKFold, KFold
from dataloader import get_dataloader
import copy
from wama_modules.utils import load_weights
from decoder import Multi_task_Unet
from torch import optim
from scheduler import get_scheduler
import torch
import numpy as np
from losses_beta import *
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import random
import os
from wama_modules.utils import resizeTensor
sep = os.sep

config = dict(
    random_seed=202020202,

    input_size=[48,128, 128],
    batch_size=2,
    batch_size_test=2,
    window = 250,# 150
    level = 25,# 50

    # 学习率与优化器相关参数
    lr_min=1e-12,
    lr_max=1e-4,
    epoch_cosinedecay=400,
    epoch_warmup=5,
    adam_beta1=0.5,
    adam_beta2=0.999,

    # 其他参数
    epoch_num=400,
    save_pth=r'/data/newnas/ZJ/SFPDK_result/kidney/RelationNet_SFPDK tumor_layer',
    gpu_device_index=1,
    aug_p=0.6,
    fold=5,
    organ = 0,
    val_flag=True,
    val_save_img=True,
    val_start_epoch=1,
    val_step_epoch=1,

    test_flag=True,
    test_save_img=True,
    test_start_epoch=1,
    test_step_epoch=1,

    model_save_step=1,

    logger_print2pic_step_iter=2,  # 将训练过程中的loss和测试验证指标保存为图片的时间步长

)


def myprint(*args):
    """Print & Record while training."""
    print(*args)
    f = open(writer_txt, 'a')
    print(*args, file=f)
    f.close()


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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# step3: 读取数据集，构建dataloader -------------------------------------------------------
dataset_test = load_from_pkl(
    r"/data/newnas/ZJ/Kidney/tumor layer/test/pre_order0_128_128_96_new.pkl")  # NIH_pancreas_liver

test_dataloader = get_dataloader(dataset=dataset_test, dataset_key='train_set', batch_size=config['batch_size_test'],
                                 drop_last=False)

dataset_train = load_from_pkl(r"/data/newnas/ZJ/Kidney/tumor layer/fold" + str(config['fold'])
                              + "/train/pre_order0_128_128_96_new.pkl")
dataset_val = load_from_pkl(r"/data/newnas/ZJ/Kidney/tumor layer/fold" + str(config['fold'] )
                            + "/val/pre_order0_128_128_96_new.pkl")

train_dataloader = get_dataloader(dataset=dataset_train, dataset_key='train_set', batch_size=config['batch_size'],
                                  drop_last=False)
val_dataloader = get_dataloader(dataset=dataset_val, dataset_key='train_set', batch_size=config['batch_size_test'],
                                drop_last=False)
print('dataset_train:', dataset_train['dataset_name'], dataset_train['train_set_num'])
print('dataset_val:', dataset_val['dataset_name'], dataset_val['train_set_num'])
print('dataset_test:', dataset_test['dataset_name'], dataset_test['train_set_num'])
# step4: 构建网络 -------------------------------------------------------
# [32, 64, 128, 256, 512]
model = Multi_task_Unet(in_channel=1,seg_label_category_dict=dict(roi=1,feature_map=1),cls_label_category_dict=dict(tumor=1))
model.cuda()

model.train()

# step5: 构建优化器（默认adam）-------------------------------------------------------
optimizer = optim.Adam(
    list(model.parameters()),
    config['lr_max'],
    [config['adam_beta1'], config['adam_beta2']]
)
save_pth = config['save_pth'] + sep + str(config['fold'] )
# step6: 学习率策略 -------------------------------------------------------
lr_scheduler = get_scheduler(optimizer,
                             lr_warm_epoch=config['epoch_warmup'],
                             lr_cos_epoch=config['epoch_cosinedecay'],
                             lr_min=config['lr_min'],
                             lr_max=config['lr_max'])

# step7: 构建损失函数 -------------------------------------------------------
crion_mse = MSE_loss()
crion_ce = nn.CrossEntropyLoss()

# step8: logger 准备 --------------------------------------------------
# 三种储存方式，优势分别为：txt记录详细log，tensorboard可视化美观，list方便
makedir(save_pth + sep + 'checkpoints')
makedir(save_pth + sep + 'logger_pic')
writer = SummaryWriter(log_dir=save_pth + sep + 'log')
writer.add_scalars('Fold', {'fold': config['fold']})
writer_txt = save_pth + sep + r'record.txt'
writer_4SaveAsPic = dict(lr=[], loss=[], loss_DICE1=[], loss_DICE2=[], loss_LOVAZ=[], loss_BCE=[], score_test=[],
                         score_val=[])

# step9: 训练&验证&测试 --------------------------------------------------------------
score_val = -99999
score_test = -99999
best_test = -9999
best_score = -9999
init_epoch = 0
current_iter = 0
# auger = aug3D(size=config['input_size'], deformation_scale=0.25)
myprint("current fold", config['fold'] )
# 判断是不是被中断的，如果是，那么就重新开始训练
# 重新加载的参数包括：
# 参数部分 1）模型权重；2）optimizer的参数，比如动量之类的；3）schedule的参数；4）epoch
# logger 部分：1）tensorboard不需要；2）writer_4SaveAsPic重新load即可



if os.path.isfile(config['save_pth'] + sep + 'checkpoints' + sep + 'latest.pkl'):
    # Load the pretrained status
    latest_status = torch.load(config['save_pth'] + sep + 'checkpoints' + sep + 'latest.pkl',
                               map_location=torch.device('cpu'))
    model.load_state_dict(latest_status['model'])
    lr_scheduler.load_state_dict(latest_status['lr_scheduler'])
    optimizer.load_state_dict(latest_status['optimizer'])
    init_epoch = latest_status['epoch']

    # Load other logger
    writer_4SaveAsPic = latest_status['writer_4SaveAsPic']

    # print
    print('restart at epoch:', latest_status['epoch'])

if os.path.isfile(config['save_pth'] + sep + 'checkpoints' + sep + 'best.pkl'):
    latest_status = torch.load(config['save_pth'] + sep + 'checkpoints' + sep + 'best.pkl',
                               map_location=torch.device('cpu'))
    best_score = latest_status['score_val']

for epoch in range(init_epoch, config['epoch_num']):
    # 训练
    for iter, sample in enumerate(train_dataloader):  # 读取训练集图像
        current_iter = current_iter + 1

        # 取出mask和scan
        scan, roi_GT,mapGT,feature_GT, class_GT = [sample['img'], tensor2numpy(sample['roi']),
                                                   tensor2numpy(sample['feature_map']),tensor2numpy(sample['feature']),
                                                   tensor2numpy(sample['class_GT'])]

        # 调整scan窗宽窗位，同时每个通道单独归一化
        scan_c0 = (adjustWindow_v2(scan, config['window'],config['level']))
        scan = np.expand_dims(scan_c0, axis=1)
        scan = torch.tensor(scan, device=device, dtype=torch.float)


        mask = torch.tensor(mask, device=device, dtype=torch.long)

        # forward
        seg_logits, cla_logits, feature_logits = model(scan)

        # loss
        roi_loss = crion_mse(seg_logits['roi'], roi_GT)
        map_loss = crion_mse(seg_logits['feature_map'], mapGT)
        feature_loss = crion_mse(feature_logits, mapGT)
        cla_loss = crion_ce(cla_logits['tumor'], class_GT)

        loss = roi_loss + map_loss + feature_loss + cla_loss

        # backward
        model.zero_grad()

        loss.backward()
        optimizer.step()

        # 记录下lr到log里(并且记录到图片里)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalars('Learning rate', {'lr': current_lr}, current_iter)

        # 储存loss（iter）到logger
        writer.add_scalars('Loss', {'loss': loss}, current_iter)

        # 储存loss到list并打印为图片
        writer_4SaveAsPic['loss'].append(loss.data.cpu().numpy())

        writer_4SaveAsPic['lr'].append(current_lr)
        if iter % config['logger_print2pic_step_iter'] == 0:
            print_logger(writer_4SaveAsPic, save_pth + sep + 'logger_pic')

        # 打印loss和lr
        myprint('Epoch [%d/%d] Iter[%d/%d] Loss: %.4f roi_loss:%.4f map_loss:%.4f feature_loss:%.4f cla_loss:%.4f  lr:%.18f' %
                (epoch + 1, config['epoch_num'], iter, len(train_dataloader), loss, roi_loss, map_loss, feature_loss,
                 cla_loss, current_lr))



    # 验证
    if config['val_flag'] and epoch >= config['val_start_epoch'] and epoch % config['val_step_epoch'] == 0:
        cla_result = test_4_Multi_Unet(dataloader=val_dataloader,
                                         model=model,
                                         device=device,
                                         result_savepth=save_pth + sep + 'image',
                                         epoch=epoch,
                                         save_img=False,
                                         save_name='val',
                                         config=config)

        # 计算一些结果，比如所有样本的平均dice，以及方差之类的
        cla_acc = np.mean([i['acc_list'] for i in cla_result])

        score_val = cla_acc

        # 储存指标到pix
        writer_4SaveAsPic['score_val'].append(score_val)
        print_logger(writer_4SaveAsPic, save_pth + sep + 'logger_pic')

        # 储存到tensorboard，并打印到txt
        writer.add_scalars('cla_val', {'cla': cla_acc}, epoch)
        myprint("cla_score ", cla_acc)  # "pancreas",dice_score_val2)

    # 测试
    if config['test_flag'] and epoch >= config['test_start_epoch'] and epoch % config['test_step_epoch'] == 0:
        cla_result = test_4_Multi_Unet(dataloader=test_dataloader,
                                         model=model,
                                         device=device,
                                         result_savepth=save_pth + sep + 'image',
                                         epoch=epoch,
                                         save_img=False,
                                         save_name='test',
                                        config=config
                                         )
        # 这里计算一些结果，比如所有样本的平均dice，以及方差之类的
        cla_acc = np.mean([i['acc_list'] for i in cla_result])

        score_test = cla_acc

        # 储存指标到pix
        writer_4SaveAsPic['score_test'].append(score_val)
        print_logger(writer_4SaveAsPic, save_pth + sep + 'logger_pic')

        # 储存到tensorboard，并打印到txt
        writer.add_scalars('cla_test', {'cla_acc': cla_acc}, epoch)
        myprint("tumor_test", cla_acc)  # ,"pancreas",dice_score_test2)

    # 更新学习率
    if (epoch + 1) <= (config['epoch_warmup'] + config['epoch_cosinedecay']):
        lr_scheduler.step()

    # 保存模型和其他参数，可以用来再次训练
    # 最新模型保存
    if True:
        state = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            lr_scheduler=lr_scheduler.state_dict(),
            epoch=epoch,
            writer_4SaveAsPic=writer_4SaveAsPic,
            score_val=score_val,
            score_test=score_test,
        )
        torch.save(state, save_pth + sep + 'checkpoints' + sep + 'latest.pkl')

    # 最优val模型保存
    if score_val >= best_score:
        best_score = score_val
        state = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            lr_scheduler=lr_scheduler.state_dict(),
            epoch=epoch,
            writer_4SaveAsPic=writer_4SaveAsPic,
            score_val=score_val,
            score_test=score_test
        )
        torch.save(state, save_pth + sep + 'checkpoints' + sep + 'epoch '+str(epoch)+' val '+str(round(score_val,3))+' test '+str(round(score_test,3))+'.pkl')
        print("best score", str(epoch))

    # 最优test模型保存
    if score_test>=best_test:
        best_test=score_test
        state = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            lr_scheduler=lr_scheduler.state_dict(),
            epoch=epoch,
            writer_4SaveAsPic=writer_4SaveAsPic,
            score_val=score_val,
            score_test=score_test
        )
        torch.save(state, save_pth + sep + 'checkpoints' + sep + 'epoch '+str(epoch)+' val '+str(round(score_val,3))+' test '+str(round(score_test,3))+'.pkl')

