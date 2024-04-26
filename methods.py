import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelAwareSmoothing(nn.Module):
    def __init__(self, cls_num_list, smooth_head, smooth_tail, shape='concave', power=None):
        super(LabelAwareSmoothing, self).__init__()

        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)

        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(cls_num_list) - n_K) / (n_1 - n_K)

        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'exp' and power is not None:
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.power((np.array(cls_num_list) - n_K) / (n_1 - n_K), power)

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()
        if torch.cuda.is_available():
            self.smooth = self.smooth.cuda()

    def forward(self, x, target):
        smoothing = self.smooth[target]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss

        return loss.mean()


class LearnableWeightScaling(nn.Module):
    def __init__(self, num_classes):
        super(LearnableWeightScaling, self).__init__()
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x):
        return self.learned_norm * x

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.ceil(W * cut_rat).astype(int)
    cut_h = np.ceil(H * cut_rat).astype(int)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# 混合数据增强 mixup+cutmix
def GLMC_mixed(org1, org2, invs1, invs2, label_org, label_invs, alpha=1):
    lam = np.random.beta(alpha, alpha)

    # mixup
    mixup_x = lam * org1 + (1 - lam) * invs1
    mixup_y = lam * label_org + (1 - lam) * label_invs
    # mixup_y_w = lam * label_org_w + (1 - lam) * label_invs_w

    # cutmix    使用 CutMix 方法对输入数据进行切割和混合
    bbx1, bby1, bbx2, bby2 = rand_bbox(org2.size(), lam)    #bbx1、bby1、bbx2、bby2 是切割框的坐标，通过调用 rand_bbox 函数生成
    org2[:, :, bbx1:bbx2, bby1:bby2] = invs2[:, :, bbx1:bbx2, bby1:bby2]     #将 invs2 的一部分混合到 org2 中
 
    lam_cutmix = lam
    cutmix_y = lam_cutmix * label_org + (1 - lam_cutmix) * label_invs
    # cutmix_y_w = lam_cutmix * label_org_w + (1 - lam_cutmix) * label_invs_w

    return mixup_x, org2, mixup_y, cutmix_y

def SimSiamLoss(p,z, version='simplified'):  # negative cosine similarity   余弦相似度
    z = z.detach()  # 停止对z的梯度计算

    if version == 'original':       #余弦相似度的计算
        p = F.normalize(p, dim=1)  # 进行L2正则化，确保范数为1
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()   #计算点积，取平均值，求负

    elif version == 'simplified':  #使用 PyTorch 提供的 F.cosine_similarity 计算余弦相似度
        return - F.cosine_similarity(p, z, dim=-1).mean()
    else:
        raise Exception

class KPSLoss(nn.Module):
    r"""Implement of KPS Loss :
    Args:
    """

    def __init__(self, cls_num_list, max_m=0.5, weighted=False, weight= None, s=30):
        super(KPSLoss, self).__init__()
        assert s > 0

        s_list = torch.cuda.FloatTensor(cls_num_list)
        s_list = s_list*(50/s_list.min())
        s_list = torch.log(s_list) #torch.log(s_list) #s_list**(1/4) #torch.log(s_list) #s_list**(1/4)#s_list = torch.log(s_list)**2  #s_list**(1/5)
        s_list = s_list*(1/s_list.min()) #s+ s_list #
        self.s_list = s_list
        self.s = s

        m_list =  torch.flip(self.s_list, dims=[0])
        m_list = m_list * (max_m / m_list.max())
        self.m_list = m_list
                
        self.weighted = weighted
        self.weight = weight
        

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = input*self.s_list
        phi = cosine - self.m_list
        # --------------------------- convert label to one-hot ---------------------------
        index = torch.zeros_like(input, dtype=torch.uint8)
        index.scatter_(1, label.data.view(-1, 1), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        #output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output = torch.where(index, phi, cosine)

        if self.weighted == False:
            output *= self.s
        elif self.weighted == True:
            index_float = index.type(torch.cuda.FloatTensor)
            batch_s = torch.flip(self.s_list, dims=[0])*self.s
            batch_s = torch.clamp(batch_s, self.s, 50)    #s过大不好。          
            batch_s = torch.matmul(batch_s[None, :], index_float.transpose(0,1)) 
            batch_s = batch_s.view((-1, 1))           
            output *= batch_s
        else:
            output *= self.s
        return F.cross_entropy(output, label, weight= self.weight)
    
class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

# 平衡logits损失，得到logits
def balancedcrossEntropy(logit):   #手动计算交叉熵，torch封装好的交叉熵会自动进行softmax和one-hot编码转换

    freq = [500, 477, 455, 434, 415, 396, 378, 361, 344, 328, 314, 299, 286, 273, 260, 248, 237, 226, 216, 206, 197, 188, 179, 171, 163, 156, 149, 142, 135, 129, 123, 118, 112, 107, 102, 98, 93, 89, 85, 81, 77, 74, 70, 67, 64, 61, 58, 56, 53, 51, 48, 46, 44, 42, 40, 38, 36, 35, 33, 32, 30, 29, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 15, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5]
    freq = torch.tensor(freq)
    sample_per_class = freq
    spc = sample_per_class.type_as(logit)    #每个样本数量的向量，转换为logits相同的类型
    spc = spc.unsqueeze(0).expand(logit.shape[0], -1)   #将其扩展为与logits相同维度
    logit = logit + spc.log()    #取对数与原来logits相加，进行logits进行纠正
    
    # loss = - (weight * (target * torch.log(softmax(logit)+1e-7)).sum(dim=1)).sum()
    return logit