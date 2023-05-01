import torch

def SmoothCEloss(logits, labels, probs = None,  C=2, alpha=0.0, mean = True, printflag = False, beta = 1, weights = None):
    """
    # 交叉熵： -q*log(p) q 金标准 p预测概率
    # focal 交叉熵：-(1-p)^gamma *q * log(p)
    :param logits:  读热编码的logits
    :param labels: 不用独热编码
    :param C: class num
    :param alpha: smooth parameter
    :param beta: >1,则关注困难样本
    weights: 例如有5类，C=5，则 weights=[1,2,1,1,1]
    :return:
    """

    N = labels.size(0)  # batchsize
    if printflag:
        print('batchsize:', N)
        print('logits:', logits)
        print('labels:', labels)

    smoothed_labels = torch.full(size=(N, C), fill_value=alpha / (C - 1)).to(logits.device)   # 注意这个to,hhhhh
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1 - alpha)

    if printflag:
        print('@or     label', '\n', labels)
        print('@smooth label', '\n', smoothed_labels)

    if probs is None:
        probs = torch.nn.functional.softmax(logits, dim=1)
    log_prob = torch.nn.functional.log_softmax(logits, dim=1)


    scaler = (1-probs)**beta
    loss_ = (scaler*log_prob * smoothed_labels) # per ins loss

    if weights is not None:
        loss_ = loss_ * torch.tensor(weights).to(logits.device)
    loss = -torch.sum(loss_, dim=1) # per ins loss
    if mean:
        return loss.mean()
    else:
        return loss

if __name__ == '__main__':
    class_num = 5   # 类别数
    logits = torch.randn(3, class_num, requires_grad=True)
    labels = torch.empty(3, dtype=torch.long).random_(5)
    loss = SmoothCEloss(
        logits, labels,
        C=class_num,
        alpha=0.0,
        mean=True,
        beta=1,  # beta>1,则关注困难样本
        weights=[1,1,1,1,2],  # 每个类别的权重
    )
    loss.backward()
