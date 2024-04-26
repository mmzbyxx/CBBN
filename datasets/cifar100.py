import numpy as np
from .sampler import ClassAwareSampler

import torch
import torchvision
from torchvision import transforms
import torchvision.datasets
from methods import TwoCropTransform

class IMBALANCECIFAR100(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



class CIFAR100_LT(object):
    def __init__(self, distributed, root='./data/cifar100', imb_type='exp',
                    imb_factor=0.01, batch_size=128, num_works=40):

        train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])     

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # 单倍数据
        train_balance_dataset  = IMBALANCECIFAR100(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0, train=True, download=True, transform=train_transform)
        
        # 双倍数据
        train_dataset = IMBALANCECIFAR100(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0, train=True, download=True, transform=TwoCropTransform(train_transform))
        eval_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=eval_transform)
        self.targets = train_dataset.targets
        
        cls_num_list = train_dataset.get_cls_num_list()
        self.cls_num_lists = train_dataset.get_cls_num_list()

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        
        resample_weighting = 0.0
        cls_weight = 1.0 / (np.array(cls_num_list) ** resample_weighting)      #计算类别权重1.0 / (样本数量 ^ args.resample_weighting)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)   #权重归一化
        samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])    #计算每个样本权重
        samples_weight = torch.from_numpy(samples_weight)    #样本权重转换为pytorch张量
        samples_weight = samples_weight.double()    #样本权重的数据类型转换为 double
    #权重采样器   加权采样，replacement=True 表示采样时可以有放回地从数据集中抽取样本
        self.weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=False)

        # 双倍数据
        self.train_instance = torch.utils.data.DataLoader(             #加载数据集
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

        # 单倍数据采样
        self.train_single = torch.utils.data.DataLoader(             #加载数据集
            train_balance_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

       #平衡单倍数据采样
        balance_sampler = ClassAwareSampler(train_balance_dataset)
        self.train_balance = torch.utils.data.DataLoader(
            train_balance_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=balance_sampler)

        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)
        
        # 权重加载器，双倍样本
        self.weighted_train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size,num_workers=num_works, 
            persistent_workers=True,pin_memory=True,sampler=self.weighted_sampler)
