'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.0017M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):    

    def __init__(self, block, num_blocks, num_classes=10, use_norm=False):
        super(ResNet_s, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if use_norm:
            self.linear = NormedLinear(64, num_classes)
        else:
            self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet_fe(nn.Module):    #使用的模型

    def __init__(self, block, num_blocks):
        super(ResNet_fe, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    #经过卷积和池化的输出，没有经过全连接层，分类器是全连接层，经过全连接层输出的是logits
    def forward(self, x):    
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out

class Classifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(Classifier, self).__init__()
        self.feat_in = feat_in * 4 
        self.fc = nn.Linear(self.feat_in, num_classes)
        self.apply(_weights_init)

    def forward(self, x):
        # print(x)
        # print("进来了！！！！")
        x = self.fc(x)
        return x

 #GLMC使用的模型，以cifar100为例实验，设定num_classes=100
class ResNet_modify(nn.Module):    

    def __init__(self, block, num_blocks, nf=64):
        super(ResNet_modify, self).__init__()
        self.in_planes = nf
        num_classes=100

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
    
        self.layer1 = self._make_layer(block, 1 * nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * nf, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, num_blocks[2], stride=2)

        self.out_dim = 4 * nf * block.expansion     #输出维度

        self.fc = nn.Linear(self.out_dim, num_classes)
        # self.fc_cb = torch.nn.utils.weight_norm(nn.Linear(512 * block.expansion, num_class), dim=0)
        hidden_dim1 = 512
        hidden_dim2 = 128
        self.fc_cb = nn.Linear(self.out_dim, num_classes)
        self.contrast_head = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2),
            
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim2),         #线性层（输入维度，输出维度）
            # nn.BatchNorm1d(hidden_dim2),
            # nn.ReLU()
            # nn.Linear(hidden_dim1, hidden_dim2)
        )
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, train=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        feature = out.view(out.size(0), -1)
        # print(feature.size())

        if train is True:
            out = feature
            out_cb = feature
            z = self.projection_head(feature)    #投影向量 
            p = self.contrast_head(z)    #传递给对比头得到对比向量
            return out, out_cb, z, p
        else:
            return feature

    @property
    # 返回模型的最后一层
    def last_layer(self):
        return self.layer3
    
    #返回模型的第二层输出
    def forward_1(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out

    #接受第二层输出，返回最终的out
    def forward_2(self, out):
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out

    
class ResNet_modify10(nn.Module):    

    def __init__(self, block, num_blocks, nf=64):
        super(ResNet_modify10, self).__init__()
        self.in_planes = nf
        num_classes=10

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
    
        self.layer1 = self._make_layer(block, 1 * nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * nf, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, num_blocks[2], stride=2)

        self.out_dim = 4 * nf * block.expansion     #输出维度

        self.fc = nn.Linear(self.out_dim, num_classes)
        # self.fc_cb = torch.nn.utils.weight_norm(nn.Linear(512 * block.expansion, num_class), dim=0)
        hidden_dim = 128
        self.fc_cb = nn.Linear(self.out_dim, num_classes)
        self.contrast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim),

        )
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, train=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        feature = out.view(out.size(0), -1)
        # print(feature.size())

        if train is True:
            out = feature
            out_cb = feature
            z = self.projection_head(feature)    #投影向量 
            p = self.contrast_head(z)    #传递给对比头得到对比向量,预测头
            return out, out_cb, z, p
        else:
            return feature

class Classifier1(nn.Module):      #返回是logits，形状是（batch_size,num_classes)，交叉熵计算logit与target的差异
    def __init__(self, feat_in, num_classes):
        super(Classifier1, self).__init__()
        self.feat_in = feat_in * 4 
        self.fc = nn.Linear(self.feat_in, num_classes)
        self.apply(_weights_init)

    def forward(self, x):
        # print(x)
        # print("进来了！！！！")
        x = self.fc(x)
        return x

    @property 
    # 获取全连接层的权重
    def weight(self):
        return self.fc.weight

class Classifier2(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(Classifier2, self).__init__()
        self.feat_in = feat_in * 4 
        self.fc = nn.Linear(self.feat_in, num_classes)
        self.apply(_weights_init)

    def forward(self, x):
        # print(x)
        # print("进来了！！！！")
        x = self.fc(x)
        return x

    @property 
    # 获取全连接层的权重
    def weight(self):
        return self.fc.weight



def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])

def resnet32_fe():      #使用的模型
    return ResNet_fe(BasicBlock, [5, 5, 5])

def resnet32():         #GLMC使用的模型
    return ResNet_modify(BasicBlock, [5, 5, 5])

def resnet32_10():         #GLMC使用的模型
    return ResNet_modify10(BasicBlock, [5, 5, 5])

# def resnet32(num_classes=10, use_norm=False):
#     return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()