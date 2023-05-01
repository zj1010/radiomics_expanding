from torchvision.datasets.mnist import read_image_file, read_label_file
import os
import csv
import openpyxl
import random
import PIL
import shutil
sep = os.sep
from wama.utils import *
import SimpleITK as sitk
import numpy as np
from six.moves import cPickle as pickle
import os
import platform
from sklearn.model_selection import train_test_split

# 读取文件
def load_pickle(f):
    version = platform.python_version_tuple()  # 取python版本号
    if version[0] == '2':
        return pickle.load(f)  # pickle.load, 反序列化为python的数据类型
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)  # dict类型
        X = datadict['data']  # X, ndarray, 像素值
        Y = datadict['labels']  # Y, list, 标签, 分类

        # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
        # transpose，转置
        # astype，复制，同时指定类型
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return [X, Y]


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []  # list
    ys = []

    # 训练集batch 1～5
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)  # 在list尾部添加对象X, x = [..., [X]]
        ys.append(Y)
    Xtr = np.concatenate(xs)  # [ndarray, ndarray] 合并为一个ndarray
    Ytr = np.concatenate(ys)
    del X, Y

    # 测试集
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return [Xtr, Ytr, Xte, Yte]

def unpickleCifar100(file):
    import pickle
    fo = open(file, 'rb')
    # dict = pickle.load(fo)#会报错，改为如下
    dict = pickle.load(fo, encoding='iso-8859-1')
    fo.close()
    return dict


def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines


def get_dataset_imagenet_sub(root_path, preload_cache=False, aim_shape=None, order=0, label_level=0):
    """

    :param root_path:
    :param preload_cache: 是否提前加载到内存
    :param aim_shape:reshape的形状，256、[256]，[256,256]三种格式都可以
    :param order: reshape的阶数，0最近邻，3cubic
    :param label_level: 标签的噪声比例，只能选[0、1、5、25、50]中的一个
    :return:
    """

    # 检查label_level是否在[0、1、5、25、50]里
    if label_level not in [0, 1, 5, 25, 50]:
        raise ValueError('label_level have to be in [0,1,5,25,50]')
    label_index = [0, 1, 5, 25, 50].index(label_level)

    # 首先根据csv文件读取类别和路径信息
    csv_file = get_filelist_frompath(root_path, 'csv')[0]
    CTcsvlines = readCsv(csv_file)
    header = CTcsvlines[0]
    # print('header', header)
    images_labels = CTcsvlines[1:]

    all_cate = list_unique([i[1] for i in images_labels])
    all_cate.sort()  # 排序保证每一次的cate 序号都一样

    train_list = []
    test_list = []
    for case in images_labels:
        if case[-1] == 'False':  # is_valid
            train_list.append(dict(
                img=None,
                img_path=root_path + sep + case[0],
                label=all_cate.index(case[label_index + 1]),
                label_name=case[label_index + 1]))
        else:
            test_list.append(dict(
                img=None,
                img_path=root_path + sep + case[0],
                label=all_cate.index(case[label_index + 1]),
                label_name=case[label_index + 1]))

    if preload_cache:
        print('loading into cache')
        for _item in train_list:
            _item['img'] = readimg(_item['img_path'], aim_shape, order)
        for _item in test_list:
            _item['img'] = readimg(_item['img_path'], aim_shape, order)

    return dict(train_set=train_list,
                test_set=test_list,
                train_set_num=len(train_list),
                test_set_num=len(test_list))


def from_pth2array(path, dataset_name, order=2, aim_shape=None):
    """
    针对不同数据集，有不同的读取方法，不支持mnist和cifar系列！
    # 目前只支持imagenette和imagewoof
    :param path:
    :param dataset_name:
    :return:
    """

    # 检查dataset_name是否有误
    if dataset_name == r'imagenette' or dataset_name == r'imagewoof':
        img = readimg(path, aim_shape, order)
    else:
        raise ValueError
    return img


def resize4pancreasNII(scan, aimspace=[128, 128, 64], order=0, is_mask=False, is_NIH=False):
    """
    # 由于医学图像的特性，xy分辨率较高，所以首先对image x和y进行resize，z等比例保持不变
    # 1）如果z稍微大于这目标z，则resize到目标z ； 如果z大于z太多，则需要把上面的减裁掉（因为胰腺在下面，所以把肺部减去），减到符合标准再resize
    # 2）如果z略微小于目标，则同样resize到目标z； 否则在上面补0，补到符合标准，再resize
    # aim_size 这个要根据resample后的总体尺寸来定，不要瞎搞，[128,128,64]就差不多（对于这个数据集）

    """
    # scan = image_reader.scan['CT']
    # 保持z轴相对比例,先将xy缩放到对应shape
    scan = resize3D(scan, aimspace[:2] + [(scan.shape[-1] / scan.shape[0]) * aimspace[0]], order)

    # 注意，这里以miccai2018 的为标准，调整nii的数据使之方向和miccai一样
    # 也就是需要下面这个操作
    # show3Dslice(scan[:,:,22:])
    # show3Dslice(scan)
    # show3Dslice(scan_NIH[::-1,::-1,::-1])
    # show3Dslice(scan_miccai)
    if is_NIH:
        scan = scan[::-1, ::-1, ::-1]

    if True:
        thresold = (5 / 64) * aimspace[-1]  # todo 自己设定的阈值，64基础上，上下可以差4
        if abs(scan.shape[-1] - aimspace[-1]) <= thresold:
            # 如果z很接近，就直接resize z轴到目标尺寸
            scan = resize3D(scan, aimspace, order)
        elif abs(scan.shape[-1] - aimspace[-1]) > thresold and (scan.shape[-1] - aimspace[-1]) >= 0:
            # 如果层数过多，则删除底部（胯部）到阈值+aimspace(因为胰腺一般靠近肝脏和肺部，而不靠近跨部），之后再resize
            # cut_slices = scan.shape[-1] -
            # scan = scan[:,:,:int((aimspace[-1]+thresold))]  # 注意这个顺序 todo 有点问题 mmp，部分label会被切掉，暂时不要这个操作
            scan = resize3D(scan, aimspace, order)
        elif abs(scan.shape[-1] - aimspace[-1]) > thresold and (scan.shape[-1] - aimspace[-1]) < 0:
            # 如果层数过多，则在顶部（肺部）添加0层到阈值-aimspace，之后再resize
            cut_slices = abs(scan.shape[-1] - (aimspace[-1] - thresold))
            tmp_scan = np.zeros(aimspace[:2] + [int(scan.shape[-1] + cut_slices)], dtype=scan.dtype)
            if is_mask:
                pass  # 如果是分割mask，则赋予0
            else:
                tmp_scan = tmp_scan - 1024  # 如果是CT，则赋予空气值
            tmp_scan[:, :, :scan.shape[-1]] = scan
            scan = resize3D(scan, aimspace, order)
        else:
            scan = resize3D(scan, aimspace, order)

    return scan


def remove_bg4pancreasNII(scan):
    """
    # 此外，CT图像预处理，可以包含“去除非身体的扫描床部分”
    # 也就是去除无关地方，这可以极大减少冗余的地方
    # 这个正好也可以在袁总的数据上用到
    # 一般来说，CT值小于-850 的地方，就可以不要了，不过还是要留一个参数控制阈值
    # 思路：二值化，开操作，取最大联通，外扩，剩下的都不要，完事，记得也要把截取矩阵输出，以供分割使用

    :param scan: 没有卡过窗宽窗外的图像！！！！
    :return:
    """

    # scan = image_reader.scan['CT']
    # scan = resize3D(scan,[256,256,None])
    # show3D(scan)
    # show3D(scan_mask_af)

    scan_mask = (scan > -900).astype(np.int)
    sitk_img = sitk.GetImageFromArray(scan_mask)
    sitk_img = sitk.BinaryMorphologicalOpening(sitk_img != 0, 15)
    scan_mask_af = sitk.GetArrayFromImage(sitk_img)
    # show3Dslice(np.concatenate([scan_mask_af, scan_mask],axis=1))
    scan_mask_af = connected_domain_3D(scan_mask_af)
    # 计算得到bbox，形式为[dim0min, dim0max, dim1min, dim1max, dim2min, dim2max]
    indexx = np.where(scan_mask_af > 0.)
    dim0min, dim0max, dim1min, dim1max, dim2min, dim2max = [np.min(indexx[0]), np.max(indexx[0]),
                                                            np.min(indexx[1]), np.max(indexx[1]),
                                                            np.min(indexx[2]), np.max(indexx[2])]
    return [dim0min, dim0max, dim1min, dim1max]


def read_nii2array4miccai_pancreas(img_pth, mask_pth, aim_spacing, aim_shape, order=3, is_NIH=False, cut_bg=True):
    """

    :param img_pth:
    :param mask_pth:
    :param aim_spacing:
    :param aim_shape:
    :param order:
    :param is_NIH: 如果是NIH数据集，则需要调整各个维度顺序，使之和MICCAI一样
    :return:
    """
    # img_pth = case['img_path']
    # mask_pth = case['mask_path']
    # aim_shape = [128,128,64]
    # aim_spacing = [0.5,0.5,0.8]

    image_reader = wama()  # 构建实例
    image_reader.appendImageAndSementicMaskFromNifti('CT', img_pth, mask_pth)

    # # 修正label(原始数据是错的，一定要先修正，如果使用的是经过修正的，就算了）
    # if is_NIH:
    #     image_reader.sementic_mask['CT'] = image_reader.sementic_mask['CT'][::-1,:,:]

    # (不要在这里调整窗宽窗位，因为可能用到多窗宽窗位）
    # image_reader.adjst_Window('CT', 321, 123)
    # resample
    if aim_spacing is not None:
        print('resampling to ', aim_spacing, 'mm')
        image_reader.resample('CT', aim_spacing, order=order)  # 首先resample没得跑,[0.5,0.5,0.8]就好

    # 去除多余部分
    # scan = image_reader.scan['CT']
    if cut_bg:
        print('cuting bg')
        bbox = remove_bg4pancreasNII(image_reader.scan['CT'])
        image_reader.scan['CT'] = image_reader.scan['CT'][bbox[0]:bbox[1], bbox[2]:bbox[3], :]
        image_reader.sementic_mask['CT'] = image_reader.sementic_mask['CT'][bbox[0]:bbox[1], bbox[2]:bbox[3], :]

    # resize到固定大小
    scan = resize4pancreasNII(image_reader.scan['CT'], aimspace=aim_shape, order=order, is_mask=False, is_NIH=is_NIH)
    mask = resize4pancreasNII(image_reader.sementic_mask['CT'], aimspace=aim_shape, order=0, is_mask=True,
                              is_NIH=is_NIH)  # 注意mask是order 0

    # 由于mask存在肿瘤和胰腺，而我们只需保存胰腺即可，所以要把胰腺和肿瘤合并！
    mask = (mask >= 0.5).astype(mask.dtype)

    return scan, mask



def make_dataset_2D(dataset, remove_none_slice=True):
    """
    仅仅适用与NIH和miccai胰腺数据，其他数据需要改代码
    :param dataset:
    :param remove_none_slice: 是否去除多余层
    :return:
    """

    # dataset = dataset_test
    print('making 2D')

    train_list = []
    # 第几层、属于哪张图片，要保留
    for ii, case in enumerate(dataset['train_set']):
        print(ii + 1, '/', dataset['train_set_num'])
        # 计算需要保留的层的范围
        if remove_none_slice:
            indexx = np.where(case['mask'] > 0.01)
            slice_index_min, slice_index_max = [np.min(indexx[2]), np.max(indexx[2])]
        else:
            slice_index_min, slice_index_max = [0, case['mask'].shape[-1]]

        for slice_index in range(slice_index_min, slice_index_max, 1):
            train_list.append(
                dict(
                    img=case['img'][:, :, slice_index],
                    mask=case['mask'][:, :, slice_index],
                    img_path=case['img_path'],
                    mask_path=case['mask_path'],
                    slice_index=slice_index,
                )
            )

    return dict(train_set=train_list,
                train_set_num=len(train_list),
                dataset_name=dataset['dataset_name'])



def volume_cal(img, spacing):
    s = spacing[0] * spacing[1] * spacing[2]
    vol = np.sum(img == 11)
    vol = vol * s
    return vol


def get_dataset_pancreas_private(preload_cache=False, order=3):
    print('getting abdomen_1k')
    dataset_name = r'pancreas'
    root = r'/data/newnas/ZJ/pnens_data/center2_seg'
    root2 = r'G:\result'
    center1 = os.listdir(root)

    nii_list = []
    mask_list = []
    pre_mask_list = []
    for i in range(len(center1)):
        nii = os.path.join(root, center1[i], 'CT_arterial_phase.nii.gz')
        mask = os.path.join(root, center1[i], 'CT_arterial_phase_ROI.nii.gz')

        pre_mask = os.path.join(root, center1[i], 'organ_fusion.nii.gz')
        nii_list.append(nii)
        mask_list.append(mask)
        pre_mask_list.append(pre_mask)
    # 先把path装进去(没有测试集，只有训练集）
    train_list = []
    for i in range(len(mask_list)):
        train_list.append(dict(
            img=None,
            mask=None,
            pre_mask=None,
            pre_mask_path=pre_mask_list[i],
            img_path=nii_list[i],
            mask_path=mask_list[i],
        ))

    # 预先读取数据
    if preload_cache:
        print('loading.')
        volume = []
        for index, case in enumerate(train_list):
            scan_c0 = []
            mask_c0 = []
            premask_c0 = []

            print('loading ', index + 1, '/', len(train_list), '...')
            scan = sitk.ReadImage(case['img_path'])
            mask = sitk.ReadImage(case['mask_path'])
            premask = sitk.ReadImage(case['pre_mask_path'])
            scan = sitk.GetArrayFromImage(scan)
            mask = sitk.GetArrayFromImage(mask)
            premask = sitk.GetArrayFromImage(premask)
            for i in range(mask.shape[0]):
                if 1 in mask[i]:
                    scan_c0.append(scan[i])
                    mask_c0.append(mask[i])
                    premask_c0.append(premask[i])


            scan = resize3D(np.array(scan_c0), [96, 128, 128], order=0)
            mask = resize3D(np.array(mask_c0), [96, 128, 128], order=0)
            premask = resize3D(np.array(premask_c0), [96, 128, 128], order=0)
            print(scan.shape)

            # show3Dslice(premask)
            case['img'] = scan.astype(np.float32)
            case['mask'] = mask.astype(np.uint8)
            case['pre_mask'] = premask.astype(np.float32)

    return dict(train_set=train_list,
                train_set_num=len(train_list),
                dataset_name=dataset_name)



def trans3D_to_2D(preload_cache=False, order=3):
    print('getting abdomen_1k')
    dataset_name = r'pancreas'
    root = r'F:\zhongjian\center2_seg'
    root2 = r'G:\result'
    center1 = os.listdir(root)

    nii_list = []
    mask_list = []
    pre_mask_list = []
    for i in range(len(center1)):
        nii = os.path.join(root, center1[i], 'CT_arterial_phase.nii.gz')
        mask = os.path.join(root, center1[i], 'CT_arterial_phase_ROI.nii')

        pre_mask = os.path.join(root, center1[i], 'LBC_label.nii.gz')
        nii_list.append(nii)
        mask_list.append(mask)
        pre_mask_list.append(pre_mask)
    # 先把path装进去(没有测试集，只有训练集）
    train_list = []
    for i in range(len(mask_list)):
        train_list.append(dict(
            img=None,
            mask=None,
            pre_mask=None,
            pre_mask_path=pre_mask_list[i],
            img_path=nii_list[i],
            mask_path=mask_list[i],
        ))

    # 预先读取数据
    if preload_cache:
        print('loading.')
        volume = []
        for index, case in enumerate(train_list):
            print('loading ', index + 1, '/', len(train_list), '...')
            scan = sitk.ReadImage(case['img_path'])
            mask = sitk.ReadImage(case['mask_path'])
            premask = sitk.ReadImage(case['pre_mask_path'])

            scan = sitk.GetArrayFromImage(scan)
            mask = sitk.GetArrayFromImage(mask)
            premask = sitk.GetArrayFromImage(premask)

            for m in range(scan.shape[0]):
                if len(np.unique(mask[m])) == 2:
                    show2D(resize2D(mask[m, :, :], [128, 128]).astype(np.uint8))
                volume.append(
                    dict(
                        img=resize2D(scan[m, :, :], [128, 128]).astype(np.float32),
                        mask=resize2D(mask[m, :, :], [128, 128]).astype(np.uint8),
                        pre_mask=resize2D(premask[m, :, :], [128, 128]).astype(np.uint8),

                        pre_mask_path=case['pre_mask_path'],
                        img_path=case['img_path'],
                        mask_path=case['mask_path']))

    return dict(train_set=volume,
                train_set_num=len(volume),
                dataset_name=dataset_name)


def collect_train_val_test(preload_cache=False, order=3, flag_2d=0):
    """
    直接在该函数就分好训练集、验证集和测试集，只生成3D，2D数据根据该函数生成的结果再处理
    """
    print('getting data')
    dataset_name = r'pancreas'
    inter_data_root = r'F:\zhongjian\center1_seg'
    outer_data_root = r'F:\zhongjian\center2_seg'

    center1 = os.listdir(inter_data_root)  # 获取不同中心数据的病例姓名
    center2 = os.listdir(outer_data_root)

    random.shuffle(center1)  # 将病例姓名打乱
    random.shuffle(center2)

    train_nii_list = []  # 存放训练数据的路径
    train_mask_list = []
    train_pre_list = []
    val_nii_list = []  # 存放验证数据的路径
    val_mask_list = []
    val_pre_list = []
    test_nii_list = []  # 存放测试数据的路径
    test_mask_list = []
    test_pre_list = []

    # 将中心1的70%作为训练集
    for i in range(int(len(center1) * 0.7)):
        nii = os.path.join(inter_data_root, center1[i], 'CT_arterial_phase.nii.gz')  # 原图
        mask = os.path.join(inter_data_root, center1[i], 'CT_arterial_phase_ROI.nii.gz')  # 标签
        pre_mask = os.path.join(inter_data_root, center1[i], 'LBC_label.nii.gz')  # 先验标签（如果有）
        train_nii_list.append(nii)
        train_mask_list.append(mask)
        train_pre_list.append(pre_mask)

    # 将中心1剩余数据作为验证集
    for i in range(len(center1)):
        nii = os.path.join(inter_data_root, center1[i], 'CT_arterial_phase.nii.gz')  # 原图
        if nii not in train_nii_list:
            mask = os.path.join(inter_data_root, center1[i], 'CT_arterial_phase_ROI.nii')  # 标签
            pre_mask = os.path.join(inter_data_root, center1[i], 'LBC_label.nii.gz')  # 先验标签（如果有）
            val_nii_list.append(nii)
            val_mask_list.append(mask)
            val_pre_list.append(pre_mask)

    for i in range(len(center2)):
        nii = os.path.join(outer_data_root, center2[i], 'CT_arterial_phase.nii.gz')  # 原图
        mask = os.path.join(outer_data_root, center2[i], 'CT_arterial_phase_ROI.nii')  # 标签
        pre_mask = os.path.join(outer_data_root, center2[i], 'LBC_label.nii.gz')  # 先验标签（如果有）
        test_nii_list.append(nii)
        test_mask_list.append(mask)
        test_pre_list.append(pre_mask)

    # 先把path装进去(没有测试集，只有训练集）
    train_list = []
    for i in range(len(train_nii_list)):
        train_list.append(dict(
            img=None,
            mask=None,
            pre_mask=None,
            pre_mask_path=train_pre_list[i],
            img_path=train_nii_list[i],
            mask_path=train_mask_list[i],
        ))

    val_list = []
    for i in range(len(val_nii_list)):
        val_list.append(dict(
            img=None,
            mask=None,
            pre_mask=None,
            pre_mask_path=val_pre_list[i],
            img_path=val_nii_list[i],
            mask_path=val_mask_list[i],
        ))

    test_list = []
    for i in range(len(test_nii_list)):
        test_list.append(dict(
            img=None,
            mask=None,
            pre_mask=None,
            pre_mask_path=test_pre_list[i],
            img_path=test_nii_list[i],
            mask_path=test_mask_list[i],
        ))
    train_2d = []

    # 预先读取数据
    if preload_cache:
        print('loading train data.')
        for index, case in enumerate(train_list):
            print('loading ', index + 1, '/', len(train_list), '...')
            scan = sitk.ReadImage(case['img_path'])
            mask = sitk.ReadImage(case['mask_path'])
            premask = sitk.ReadImage(case['pre_mask_path'])

            scan = sitk.GetArrayFromImage(scan)
            mask = sitk.GetArrayFromImage(mask)
            premask = sitk.GetArrayFromImage(premask)
            if flag_2d == 0:
                scan = resize3D(scan, [96, 128, 128], order=0)
                mask = resize3D(mask, [96, 128, 128], order=0)
                premask = resize3D(premask, [96, 128, 128], order=0)
                print(scan.shape)
                case['img'] = scan.astype(np.float32)
                case['mask'] = mask.astype(np.uint8)
                case['pre_mask'] = premask.astype(np.float32)
            elif flag_2d == 1:
                for j in range(mask.shape[0]):
                    if len(np.unique(mask[j, :, :])) == 2:
                        scan_2d = resize2D(scan[j, :, :], [128, 128])
                        mask_2d = resize2D(mask[j, :, :], [128, 128])
                        premask_2d = resize2D(premask[j, :, :], [128, 128])
                        # print(scan_2d.shape)
                        train_2d.append(dict(
                            img=scan_2d.astype(np.float32),
                            mask=mask_2d.astype(np.uint8),
                            pre_mask=premask_2d.astype(np.float32),
                            pre_mask_path=case['pre_mask_path'],
                            img_path=case['img_path'],
                            mask_path=case['mask_path'],
                        ))

        print('\nloading val data.')
        for index, case in enumerate(val_list):
            print('loading ', index + 1, '/', len(val_list), '...')
            scan = sitk.ReadImage(case['img_path'])
            mask = sitk.ReadImage(case['mask_path'])
            premask = sitk.ReadImage(case['pre_mask_path'])

            scan = sitk.GetArrayFromImage(scan)
            mask = sitk.GetArrayFromImage(mask)
            premask = sitk.GetArrayFromImage(premask)
            scan = resize3D(scan, [96, 128, 128], order=0)
            mask = resize3D(mask, [96, 128, 128], order=0)
            premask = resize3D(premask, [96, 128, 128], order=0)
            print(scan.shape)
            case['img'] = scan.astype(np.float32)
            case['mask'] = mask.astype(np.uint8)
            case['pre_mask'] = premask.astype(np.float32)

        print('\nloading test data.')
        for index, case in enumerate(test_list):
            print('loading ', index + 1, '/', len(test_list), '...')
            scan = sitk.ReadImage(case['img_path'])
            mask = sitk.ReadImage(case['mask_path'])
            premask = sitk.ReadImage(case['pre_mask_path'])

            scan = sitk.GetArrayFromImage(scan)
            mask = sitk.GetArrayFromImage(mask)
            premask = sitk.GetArrayFromImage(premask)
            scan = resize3D(scan, [96, 128, 128], order=0)
            mask = resize3D(mask, [96, 128, 128], order=0)
            premask = resize3D(premask, [96, 128, 128], order=0)
            print(scan.shape)
            case['img'] = scan.astype(np.float32)
            case['mask'] = mask.astype(np.uint8)
            case['pre_mask'] = premask.astype(np.float32)

    if flag_2d == 0:
        return dict(train_set=train_list, train_set_num=len(train_list), dataset_name=dataset_name), \
               dict(train_set=val_list, train_set_num=len(val_list), dataset_name=dataset_name), \
               dict(train_set=test_list, train_set_num=len(test_list), dataset_name=dataset_name)
    elif flag_2d == 1:
        return dict(train_set=train_2d, train_set_num=len(train_2d), dataset_name=dataset_name), \
               dict(train_set=val_list, train_set_num=len(val_list), dataset_name=dataset_name), \
               dict(train_set=test_list, train_set_num=len(test_list), dataset_name=dataset_name)


def make_train_2_2d():
    root = r'F:\zhongjian\pancreas\SFDK_pkl\3D_data\LBC\train\pre_order0_128_128_96_new.pkl'
    data = load_from_pkl(root)
    data_train = data[0]['train_set']
    mask_list = []
    pre_mask_list = []
    img_list = []
    train_list = []
    for i in data_train:
        for j in range(i['mask'].shape[0]):
            if len(np.unique(i['mask'][j, :, :])) == 2:
                train_list.append(
                    dict(img=i['img'][j, :, :],
                         mask=i['mask'][j, :, :],
                         pre_mask=i['pre_mask'][j, :, :],
                         pre_mask_path=i['pre_mask_path'],
                         img_path=i['img_path'],
                         mask_path=i['mask_path']
                         ))
    return dict(train_set=train_list,
                train_set_num=len(train_list),
                dataset_name='pancreas')



def correct_name():
    path = r'G:\dataset\pi\PI\AllDeal\center2\Done'
    names = os.listdir(os.path.join(path, 'Image'))
    for name in names:
        print(name)
        nii_path = os.path.join(path, 'Image', name, 't2 haste cor.nii.gz')
        mask_path = os.path.join(path, 'Mask', name, 't2 haste cor.nii.gz')
        if not os.path.exists(nii_path):
            nii = os.listdir(os.path.join(path, 'Image', name))
            os.rename(os.path.join(path, 'Image', name, nii[0]), nii_path)
        if not os.path.exists(mask_path):
            mask = os.listdir(os.path.join(path, 'Mask', name))
            os.rename(os.path.join(path, 'Mask', name, mask[0]), mask_path)
    print(2)


def get_dataset_epityphlon_private(preload_cache=False, order=3):
    print('getting abdomen_1k')
    dataset_name = r'pancreas'
    root = r'F:\zhongjian\center2_seg'
    root2 = r'G:\result'
    center1 = os.listdir(root)

    nii_list = []
    mask_list = []
    pre_mask_list = []
    for i in range(len(center1)):
        nii = os.path.join(root, center1[i], 'CT_arterial_phase.nii.gz')
        mask = os.path.join(root, center1[i], 'CT_arterial_phase_ROI.nii.gz')

        pre_mask = os.path.join(root, center1[i], 'LBC_label.nii.gz')
        nii_list.append(nii)
        mask_list.append(mask)
        pre_mask_list.append(pre_mask)
    # 先把path装进去(没有测试集，只有训练集）
    train_list = []
    for i in range(len(mask_list)):
        train_list.append(dict(
            img_2d=None,
            mask_2d=None,
            pre_2d=None,
            img=None,
            mask=None,
            pre_mask=None,
            pre_mask_path=pre_mask_list[i],
            img_path=nii_list[i],
            mask_path=mask_list[i],
        ))

    # 预先读取数据
    if preload_cache:
        print('loading.')
        volume = []
        for index, case in enumerate(train_list):
            print('loading ', index + 1, '/', len(train_list), '...')
            scan = sitk.ReadImage(case['img_path'])
            mask = sitk.ReadImage(case['mask_path'])
            premask = sitk.ReadImage(case['pre_mask_path'])
            scan = sitk.GetArrayFromImage(scan)
            mask = sitk.GetArrayFromImage(mask)
            premask = sitk.GetArrayFromImage(premask)

            if index > 114:
                for i in range(mask.shape[0]):
                    if len(np.unique(mask[i, :, :])) == 2:
                        scan_2d = resize2D(scan[i, :, :], [128, 128], order=0)
                        mask_2d = resize2D(mask[i, :, :], [128, 128], order=0)
                        premask_2d = resize2D(premask[i, :, :], [128, 128], order=0)
                        print(scan_2d.shape)

            scan = resize3D(scan, [96, 128, 128], order=0)
            mask = resize3D(mask, [96, 128, 128], order=0)
            premask = resize3D(premask, [96, 128, 128], order=0)
            print(scan.shape)

            # show3Dslice(premask)
            case['img'] = scan.astype(np.float32)
            case['mask'] = mask.astype(np.uint8)
            case['pre_mask'] = premask.astype(np.float32)

    return dict(train_set=train_list,
                train_set_num=len(train_list),
                dataset_name=dataset_name)


def set_LWY_data():
    root = r'G:\dataset\epityphlon\IMG'
    mask = r'G:\dataset\epityphlon\MASK'
    pth = r'G:\dataset\epityphlons'
    names = os.listdir(root)
    for name in names:
        os.makedirs(os.path.join(pth, name[:-7]))
        os.makedirs(os.path.join(pth, name[:-7], "organs"))
        shutil.copy(os.path.join(root, name), os.path.join(pth, name[:-7], "original.nii.gz"))
        shutil.copy(os.path.join(mask, name), os.path.join(pth, name[:-7], "mask.nii.gz"))
        print(2)



def get_dataset_epityphlons_private(preload_cache=False, order=3):
    print('getting abdomen_1k')
    dataset_name = r'pancreas'
    root = r'/data/newnas/ZJ/epityphlons_pkl/test'
    root2 = r'G:\result'
    center1 = os.listdir(root)

    nii_list = []
    mask_list = []
    pre_mask_list = []
    scan_c0 = []
    premask_c0 = []
    mask_c0 = []
    for i in range(len(center1)):
        nii = os.path.join(root, center1[i], 'original.nii.gz')
        mask = os.path.join(root, center1[i], 'mask.nii.gz')

        pre_mask = os.path.join(root, center1[i], 'organ_fusion.nii.gz')
        nii_list.append(nii)
        mask_list.append(mask)
        pre_mask_list.append(pre_mask)
    # 先把path装进去(没有测试集，只有训练集）
    train_list = []
    for i in range(len(mask_list)):
        train_list.append(dict(
            img=None,
            mask=None,
            pre_mask=None,
            pre_mask_path=pre_mask_list[i],
            img_path=nii_list[i],
            mask_path=mask_list[i],
        ))

    # 预先读取数据
    if preload_cache:
        print('loading.')
        volume = []
        for index, case in enumerate(train_list):
            pre_mask_list = []
            scan_c0 = []
            premask_c0 = []
            mask_c0 = []
            print('loading ', index + 1, '/', len(train_list), '...')
            scan = sitk.ReadImage(case['img_path'])
            mask = sitk.ReadImage(case['mask_path'])
            premask = sitk.ReadImage(case['pre_mask_path'])
            scan = sitk.GetArrayFromImage(scan)
            mask = sitk.GetArrayFromImage(mask)
            premask = sitk.GetArrayFromImage(premask)
            for i in range(mask.shape[0]):
                if 1 in mask[i]:
                    scan_c0.append(scan[i])
                    mask_c0.append(mask[i])
                    premask_c0.append(premask[i])

            scan = resize3D(np.array(scan_c0), [96, 128, 128], order=0)
            mask = resize3D(np.array(mask_c0), [96, 128, 128], order=0)
            premask = resize3D(np.array(premask_c0), [96, 128, 128], order=0)
            print(scan.shape)

            # show3Dslice(premask)
            case['img'] = scan.astype(np.float32)
            case['mask'] = mask.astype(np.uint8)
            case['pre_mask'] = premask.astype(np.float32)

    return dict(train_set=train_list,
                train_set_num=len(train_list),
                dataset_name=dataset_name)

def get_dataset_KiTS19(preload_cache=False, order=3):
    print('getting abdomen_1k')
    dataset_name = r'pancreas'
    root = r'/data/newnas/ZJ/Kidney/data'
    root2 = r'G:\result'
    center1 = os.listdir(root)
    train, test = train_test_split(center1,test_size=0.3,random_state=42,shuffle=True)
    nii_list = []
    mask_list = []
    pre_mask_list = []

    nii_test_list = []
    mask_test_list = []
    pre_mask_test_list = []

    for i in range(len(train)):
        nii = os.path.join(root, train[i], 'imaging.nii.gz')
        mask = os.path.join(root, train[i], 'segmentation.nii.gz')

        pre_mask = os.path.join(root, train[i], 'organ_fusion.nii.gz')
        nii_list.append(nii)
        mask_list.append(mask)
        pre_mask_list.append(pre_mask)

    for i in range(len(test)):
        nii = os.path.join(root, test[i], 'imaging.nii.gz')
        mask = os.path.join(root, test[i], 'segmentation.nii.gz')
        pre_mask = os.path.join(root, test[i], 'organ_fusion.nii.gz')
        nii_test_list.append(nii)
        mask_test_list.append(mask)
        pre_mask_test_list.append(pre_mask)
    # 先把path装进去(没有测试集，只有训练集）
    train_list = []
    test_list = []
    for i in range(len(mask_list)):
        train_list.append(dict(
            img=None,
            mask=None,
            pre_mask=None,
            pre_mask_path=pre_mask_list[i],
            img_path=nii_list[i],
            mask_path=mask_list[i],
        ))

    for i in range(len(mask_test_list)):
        test_list.append(dict(
            img=None,
            mask=None,
            pre_mask=None,
            pre_mask_path=pre_mask_test_list[i],
            img_path=nii_test_list[i],
            mask_path=mask_test_list[i],
        ))

    # 预先读取数据
    if preload_cache:
        print('loading.')
        volume = []
        for index, case in enumerate(train_list):
            pre_mask_list = []
            scan_c0 = []
            premask_c0 = []
            mask_c0 = []
            print('loading ', index + 1, '/', len(train_list), '...')
            scan = sitk.ReadImage(case['img_path'])
            mask = sitk.ReadImage(case['mask_path'])
            premask = sitk.ReadImage(case['pre_mask_path'])
            scan = sitk.GetArrayFromImage(scan)
            mask = sitk.GetArrayFromImage(mask)
            mask[mask==1]=0
            mask[mask==2]=1
            premask = sitk.GetArrayFromImage(premask)
            for i in range(mask.shape[0]):
                if 1 in mask[i]:
                    scan_c0.append(scan[i])
                    mask_c0.append(mask[i])
                    premask_c0.append(premask[i])

            scan = resize3D(np.array(scan_c0), [96, 128, 128], order=0)
            mask = resize3D(np.array(mask_c0), [96, 128, 128], order=0)
            premask = resize3D(np.array(premask_c0), [96, 128, 128], order=0)
            print(scan.shape)

            # show3Dslice(premask)
            case['img'] = scan.astype(np.float32)
            case['mask'] = mask.astype(np.uint8)
            case['pre_mask'] = premask.astype(np.float32)

    if preload_cache:
        print('loading.')
        volume = []
        for index, case in enumerate(test_list):
            pre_mask_list = []
            scan_c0 = []
            premask_c0 = []
            mask_c0 = []
            print('loading ', index + 1, '/', len(test_list), '...')
            scan = sitk.ReadImage(case['img_path'])
            mask = sitk.ReadImage(case['mask_path'])
            premask = sitk.ReadImage(case['pre_mask_path'])
            scan = sitk.GetArrayFromImage(scan)
            mask = sitk.GetArrayFromImage(mask)
            mask[mask == 1] = 0
            mask[mask == 2] = 1
            premask = sitk.GetArrayFromImage(premask)
            for i in range(mask.shape[0]):
                if 1 in mask[i]:
                    scan_c0.append(scan[i])
                    mask_c0.append(mask[i])
                    premask_c0.append(premask[i])

            scan = resize3D(np.array(scan_c0), [96, 128, 128], order=0)
            mask = resize3D(np.array(mask_c0), [96, 128, 128], order=0)
            premask = resize3D(np.array(premask_c0), [96, 128, 128], order=0)
            print(scan.shape)

            # show3Dslice(premask)
            case['img'] = scan.astype(np.float32)
            case['mask'] = mask.astype(np.uint8)
            case['pre_mask'] = premask.astype(np.float32)

    return dict(train_set=train_list,train_set_num=len(train_list),dataset_name=dataset_name),\
           dict(train_set=test_list,train_set_num=len(test_list),dataset_name=dataset_name),


def move_nii():
    root = r'/data/newnas/ZJ/epityphlons'
    save = r'/data/newnas/ZJ/epityphlons_pkl'
    excel_pth = r'/data/newnas/ZJ/LWY_Finally.xlsx'
    excel = openpyxl.load_workbook(excel_pth)
    sheet = excel.active
    cell1 = sheet['A']
    cell2 = sheet['D']
    names = os.listdir(root)
    for n in names:
        for i in range(len(cell1)):
            if n == str(cell1[i].value):
                if cell2[i].value == "train":
                    shutil.copytree(os.path.join(root, n), os.path.join(save, "train", n))
                else:
                    shutil.copytree(os.path.join(root, n), os.path.join(save, "test", n))


if __name__ == '__main__':
    print("on going")
    x,y  = get_dataset_KiTS19(preload_cache=True, order=0)
    save_as_pkl(r'/data/newnas/ZJ/Kidney/tumor layer/train/pre_order0_128_128_96_new.pkl', x)
    save_as_pkl(r'/data/newnas/ZJ/Kidney/tumor layer/test/pre_order0_128_128_96_new.pkl', y)
    # move_nii()
    # correct_name()
    # make_multiorgan_label()
    # make_new_PI()
    # x = make_train_2_2d()
    # x = get_dataset_LiTS2017(preload_cache=True, order=0)
    # save_as_pkl(r'D:\z\innovation\data\liver\MedSeg\pre_order0_128_128_96_new.pkl', x)










