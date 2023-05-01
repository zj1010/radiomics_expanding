import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy.metric import binary
import numpy as np


# 主旨是简单，所以maybe loss不需要太复杂
# 但是有时侯，搞出来一个很精致的loss，其实也是很强的
# loss的设计，要看你的目的是什么，度量学习有点重要貌似

# l1 loss  or  mse loss


# l2 loss


# NCC loss or CC loss


# SIMM loss


# cosine 相似度 loss


# 一些分割的loss

# dice loss
class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):
        # probs = torch.zeros([3,250])
        # targets = torch.zeros([3,250])
        # probs[:,175:] = 1
        # targets[:,125:] = 1
        num = targets.size(0)
        smooth = 1

        m1 = probs
        m2 = targets
        intersection = (m1 * m2)

        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)
    pred[pred < 0.5] = 0
    pred[pred > 0.5] = 1
    N = gt.size(0)
    pred_flat = pred.contiguous().view(N, -1)
    gt_flat = gt.contiguous().view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    loss_b = loss.sum() / N
    return loss_b


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.contiguous().view(num, -1)  # Flatten
    m2 = target.contiguous().view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class DiceMean(nn.Module):
    def __init__(self):
        super(DiceMean, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return 1 - (dice_sum / class_num)


class KDLoss():
    def __init__(self, temp=2):
        self.temp = temp
        self.log_sotfmax = nn.LogSoftmax(dim=-1)

    def __call__(self, preds, gts, strategy):
        preds = F.softmax(preds, dim=-1)
        preds = torch.pow(preds, 1. / self.temp)

        # l_preds = F.softmax(preds, dim=-1)
        l_preds = self.log_sotfmax(preds)

        if strategy == "lwf":
            gts = F.softmax(gts, dim=-1)
            gts = torch.pow(gts, 1. / self.temp)
            l_gts = self.log_sotfmax(gts)

        elif strategy == "lwf_eq_prob":
            eq_prob = 1. / gts.size()[1]
            gts = torch.empty(gts.size(), device=gts.device).fill_(eq_prob)
            gts = torch.pow(gts, 1. / self.temp)
            l_gts = self.log_sotfmax(gts)

        l_preds = torch.log(l_preds)
        l_preds[l_preds != l_preds] = 0.  # Eliminate NaN values
        loss = torch.mean(torch.sum(-l_gts * l_preds, axis=1))
        return loss


class KnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', alpha=1., kd_cil_weights=False):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.kd_cil_weights = kd_cil_weights
        self.temp = 2.0

    def forward(self, inputs, targets, mask=None):
        # inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs / self.temp, dim=1)
        ax = torch.unique(outputs)
        # print(ax)
        labels = torch.softmax(targets * self.alpha / self.temp, dim=1)

        loss = (outputs * labels).sum(dim=1).mean() * self.temp ** 2
        # print(loss)
        if self.kd_cil_weights:
            w = -(torch.softmax(targets, dim=1) * torch.log_softmax(targets, dim=1)).sum(dim=1) + 1.0
            loss = loss * w[:, None]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            x = torch.unique(loss)
            # print(x)
            output = -torch.mean(loss)
        elif self.reduction == 'sum':
            output = -torch.sum(loss)
        else:
            output = -loss

        return output


class SoftDiceLoss2(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, activation='sigmoid'):
        super(SoftDiceLoss2, self).__init__()
        self.activation = activation

    def forward(self, y_pr, y_gt):
        return 1 - dice_coeff(y_pr, y_gt)


# dice loss
class SoftDiceLoss3(nn.Module):
    """
    probs_n = torch.zeros([8,1,96,96,48])
    targets_n = torch.zeros([8,1,96,96,48])
    probs_n[0,0,0,0,0] = 1
    probs_n[:,:,:200] = 1
    targets_n[:,:,160:] = 1

    probs_n = torch.zeros([2,2])
    targets_n = torch.zeros([2,2])
    probs_n[:,1] = 1
    targets_n[:,0] = 1
    a = SoftDiceLoss()
    loss = a(probs_n, targets_n)
    probs_n = targets_n
    loss = a(probs_n, targets_n)
    """

    def __init__(self):
        super(SoftDiceLoss3, self).__init__()

    def forward(self, probs_n, targets_n):
        # print(probs_n.max(), probs_n.min())
        # print(targets_n.max(), targets_n.min())
        # 先flatten(保留前两个维度，bz，c，-1）
        probs = probs_n.view(*probs_n.shape[:2], -1)  # （bz，c，-1）
        targets = targets_n.view(*targets_n.shape[:2], -1)

        # smooth = 1e-12
        smooth = 1

        m1 = probs
        m2 = targets
        intersection = (m1 * m2)

        score = (2. * (intersection.sum(2)) + smooth) / (m1.sum(2) + m2.sum(2) + smooth)

        # 先对通道取mean
        score_channel_mean = score.mean(1)

        # 再对case取mean
        score_case_mean = score_channel_mean.mean()

        score_final = 1 - score_case_mean
        return score_final


class HardDiceLoss(nn.Module):
    def __init__(self):
        super(HardDiceLoss, self).__init__()

    def forward(self, probs_n, targets_n):
        # print(probs_n.max(), probs_n.min())
        # print(targets_n.max(), targets_n.min())
        # 先flatten(保留前两个维度，bz，c，-1）
        probs_n[probs_n < 0.5] = 0
        probs_n[probs_n >= 0.5] = 1
        probs = probs_n.view(*probs_n.shape[:2], -1)  # （bz，c，-1）
        targets = targets_n.view(*targets_n.shape[:2], -1)

        # smooth = 1e-12
        smooth = 1

        m1 = probs
        m2 = targets
        intersection = (m1 * m2)

        score = (2. * (intersection.sum(2)) + smooth) / (m1.sum(2) + m2.sum(2) + smooth)
        # 先对通道取mean
        score_channel_mean = score.mean(1)
        # 再对case取mean
        score_case_mean = score_channel_mean.mean()

        return score_case_mean


class Hausdorff_distance(nn.Module):
    def __init__(self):
        super(Hausdorff_distance, self).__init__()

    def forward(self, probs_n, targets_n):
        probs_n[probs_n < 0.5] = 0
        probs_n[probs_n >= 0.5] = 1
        # probs = probs_n.view(*probs_n.shape[:2], -1)  # （bz，c，-1）
        # targets = targets_n.view(*targets_n.shape[:2], -1)
        #

        # # smooth = 1e-12

        hd95 = binary.hd95(probs_n, targets_n)

        return hd95


class MSE_loss(nn.Module):
    def __init__(self):
        super(MSE_loss, self).__init__()

    def forward(self, probs_n, targets_n):
        probs = probs_n.view(*probs_n.shape[:2], -1)  # （bz，c，-1）
        targets = targets_n.view(*targets_n.shape[:2], -1)
        loss = ((probs-targets)**2).mean()
        return loss

class cross_entrophy_loss(nn.Module):
    def __init__(self):
        super(cross_entrophy_loss, self).__init__()
    def forward(self, probs_n, targets_n):
        probs = probs_n.view(*probs_n.shape[:2], -1)  # （bz，c，-1）
        targets = targets_n.view(*targets_n.shape[:1], -1)
        loss = torch.nn.functional.cross_entropy(probs,targets)
        return loss

def h_dice(probs,targets):
    probs[probs < 0.5] = 0
    probs[probs >= 0.5] = 1
    probs = probs.reshape(4,1,-1)  # （bz，c，-1）
    targets = targets.reshape(4,1,-1)

    # smooth = 1e-12
    smooth = 1

    m1 = probs
    m2 = targets
    intersection = (m1 * m2)

    score = (2. * (intersection.sum(2)) + smooth) / (m1.sum(2) + m2.sum(2) + smooth)
    # 先对通道取mean
    score_channel_mean = score.mean(1)
    # 再对case取mean
    score_case_mean = score_channel_mean.mean()

    return score_case_mean