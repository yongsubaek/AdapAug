import logging

import numpy as np
import os, copy

import math
import random
import torch
import torchvision
from PIL import Image

from torch.utils.data import Dataset, SubsetRandomSampler, Sampler, Subset, ConcatDataset
import torch.distributed as dist
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit, PredefinedSplit
from theconf import Config as C

from AdapAug.archive import arsaug_policy, autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10, fa_reduced_svhn, fa_resnet50_rimagenet
from AdapAug.augmentations import *
from AdapAug.common import get_logger
from AdapAug.imagenet import ImageNet
from AdapAug.networks.efficientnet_pytorch.model import EfficientNet
from collections import Counter
op_list = augment_list(False)

logger = get_logger('Fast AutoAugment')
logger.setLevel(logging.INFO)

_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
_CIFAR_STD2 = (0.2470, 0.2435, 0.2616)
_SVHN_MEAN, _SVHN_STD = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)

class AdapAugData(Dataset):
    def __init__(self, dataname, controller=None, transform=None, given_policy=None, target_transform=None, clean_transform=None, batch_multiplier=1, **kargs):
        dataset = torchvision.datasets.__dict__[dataname](transform=None, **kargs)
        self.dataname = dataname
        self.data = dataset.data if dataname != "SVHN" else np.transpose(dataset.data, (0,2,3,1))
        self.targets = self.labels = dataset.targets if dataname != "SVHN" else dataset.labels
        self.transform = transform

        # for i,x in enumerate(transform.transforms):
        #     if type(x) == transforms.ToTensor:
        #         split_i = i
        #         break
        # else:
        #     raise ValueError(f"transform has no ToTensor: {transform}")
        # self.before_transform = transforms.Compose(transform.transforms[:split_i])
        # self.after_transform = transforms.Compose(transform.transforms[split_i:])
        self.clean_transform = clean_transform
        self.target_transform = target_transform

        self.controller = controller
        self.given_policy = given_policy
        self.log_probs = None
        self.policies = None

        self.batch_multiplier = batch_multiplier

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            if self.policies is not None: # CTL Training
                log_prob = self.log_probs[index] # [M]
                policy = self.policies[index] # [M]
                if self.batch_multiplier > 1:
                    aug_imgs = []
                    for pol in policy:
                        # aug_img = self.before_transform(img)
                        # aug_img = Augmentation(pol)(aug_img)
                        # aug_img = self.after_transform(aug_img)
                        aug_img = Augmentation(pol)(img)
                        aug_img = self.transform(aug_img)
                        aug_imgs.append(aug_img)
                    aug_img =  torch.stack(aug_imgs) # [M, 3, 32, 32]
                else:
                    # aug_img = self.before_transform(img)
                    # aug_img = Augmentation(policy)(aug_img)
                    # aug_img = self.after_transform(aug_img)
                    aug_img = Augmentation(policy)(img)
                    aug_img = self.transform(aug_img)
                img = self.clean_transform(img)
            else:
                if self.controller is None: # Adversarial AutoAugment
                    if self.batch_multiplier > 1:
                        imgs = []
                        for policy in self.given_policy:
                            # aug_img = self.before_transform(img)
                            # aug_img = Augmentation(policy)(aug_img)
                            # aug_img = self.after_transform(aug_img)
                            aug_img = Augmentation(policy)(img)
                            aug_img = self.transform(aug_img)
                            imgs.append(aug_img)
                        img = torch.stack(imgs) # [M, 3, 32, 32]
                    else:
                        img = self.transform(img)
                else: # AdapAug temp_loader
                    img = self.clean_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.policies is not None:
            return (aug_img, img, log_prob, policy), target
        else:
            return img, target
def get_dataloaders(dataset, batch, dataroot, split=0.15, split_idx=0, controller=None, _transform=None, rand_val=False, batch_multiplier=1, validation=False):
    if _transform is None:
        _transform = C.get()['aug']
    if 'cifar' in dataset or 'svhn' in dataset:
        if "cifar" in dataset:
            _mean, _std = _CIFAR_MEAN, _CIFAR_STD
        else:
            _mean, _std = _SVHN_MEAN, _SVHN_STD
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std),
        ])
    elif 'imagenet' in dataset:
        input_size = 224
        sized_size = 256

        if 'efficientnet' in C.get()['model']['type']:
            input_size = EfficientNet.get_image_size(C.get()['model']['type'])
            sized_size = input_size + 32    # TODO
            # sized_size = int(round(input_size / 224. * 256))
            # sized_size = input_size
            logger.info('size changed to %d/%d.' % (input_size, sized_size))

        transform_train = transforms.Compose([
            EfficientNetRandomCrop(input_size),
            transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            # transforms.RandomResizedCrop(input_size, scale=(0.1, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            EfficientNetCenterCrop(input_size),
            transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    else:
        raise ValueError('dataset=%s' % dataset)

    if batch_multiplier > 1:
        logger.debug('Large Batch augmentation provided.')
        pass
    elif isinstance(_transform, list):
        logger.debug('augmentation provided.')
        transform_train.transforms.insert(0, Augmentation(_transform))
    elif isinstance(_transform, dict):
        # group version
        logger.debug('group augmentation provided.')
    else:
        logger.debug('augmentation: %s' % _transform)
        if _transform == 'fa_reduced_cifar10':
            transform_train.transforms.insert(0, Augmentation(fa_reduced_cifar10()))

        elif _transform == 'fa_reduced_imagenet':
            transform_train.transforms.insert(0, Augmentation(fa_resnet50_rimagenet()))

        elif _transform == 'fa_reduced_svhn':
            transform_train.transforms.insert(0, Augmentation(fa_reduced_svhn()))

        elif _transform == 'arsaug':
            transform_train.transforms.insert(0, Augmentation(arsaug_policy()))
        elif _transform == 'autoaug_cifar10':
            transform_train.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        elif _transform == 'autoaug_extend':
            transform_train.transforms.insert(0, Augmentation(autoaug_policy()))
        elif _transform in ['default', "clean", "nonorm", "nocut"]:
            pass
        else:
            raise ValueError('not found augmentations. %s' % _transform)

    if C.get()['cutout'] > 0 and _transform != "nocut":
        transform_train.transforms.append(CutoutDefault(C.get()['cutout']))
    if _transform == "clean":
        transform_train = transform_test
    elif _transform == "nonorm":
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
    train_idx = valid_idx = None
    if dataset == 'cifar10':
        if controller is not None or batch_multiplier > 1:
            total_trainset = AdapAugData("CIFAR10", root=dataroot, controller=controller, train=True, download=False, transform=transform_train, clean_transform=transform_test, given_policy=_transform, batch_multiplier=batch_multiplier)
        else:
            total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=False, transform=transform_test)
    elif dataset == 'reduced_cifar10':
        if controller is not None or batch_multiplier > 1:
            total_trainset = AdapAugData("CIFAR10", root=dataroot, controller=controller, train=True, download=False, transform=transform_train, clean_transform=transform_test, given_policy=_transform, batch_multiplier=batch_multiplier)
        else:
            total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=False, transform=transform_train)
        sss = StratifiedShuffleSplit(n_splits=5, train_size=4000, random_state=0)   # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        for _ in range(split_idx+1):
            train_idx, valid_idx = next(sss)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=False, transform=transform_test)
    elif dataset == 'cifar100':
        if controller is not None or batch_multiplier > 1:
            total_trainset = AdapAugData("CIFAR100", root=dataroot, controller=controller, train=True, download=False, transform=transform_train, clean_transform=transform_test, given_policy=_transform, batch_multiplier=batch_multiplier)
        else:
            total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=False, transform=transform_test)
    elif dataset == 'svhn': #TODO
        trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=False, transform=transform_train)
        extraset = torchvision.datasets.SVHN(root=dataroot, split='extra', download=False, transform=transform_train)
        total_trainset = ConcatDataset([trainset, extraset])
        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=False, transform=transform_test)
    elif dataset == 'reduced_svhn':
        if controller is not None or batch_multiplier > 1:
            total_trainset = AdapAugData("SVHN", root=dataroot, controller=controller, split='train', download=False, transform=transform_train, clean_transform=transform_test, given_policy=_transform, batch_multiplier=batch_multiplier)
        else:
            total_trainset = torchvision.datasets.SVHN(root=dataroot, split='train', download=False, transform=transform_train)
        sss = StratifiedShuffleSplit(n_splits=5, train_size=1000, test_size=7325, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.labels)
        for _ in range(split_idx+1):
            train_idx, valid_idx = next(sss)
        # targets = [total_trainset.labels[idx] for idx in train_idx]
        # total_trainset = Subset(total_trainset, train_idx)
        # total_trainset.targets = targets
        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=False, transform=transform_test)
    elif dataset == 'imagenet':
        total_trainset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), transform=transform_train)
        testset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), split='val', transform=transform_test)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]
    elif dataset == 'reduced_imagenet':
        # randomly chosen indices
        # idx120 = sorted(random.sample(list(range(1000)), k=120))
        idx120 = [16, 23, 52, 57, 76, 93, 95, 96, 99, 121, 122, 128, 148, 172, 181, 189, 202, 210, 232, 238, 257, 258, 259, 277, 283, 289, 295, 304, 307, 318, 322, 331, 337, 338, 345, 350, 361, 375, 376, 381, 388, 399, 401, 408, 424, 431, 432, 440, 447, 462, 464, 472, 483, 497, 506, 512, 530, 541, 553, 554, 557, 564, 570, 584, 612, 614, 619, 626, 631, 632, 650, 657, 658, 660, 674, 675, 680, 682, 691, 695, 699, 711, 734, 736, 741, 754, 757, 764, 769, 770, 780, 781, 787, 797, 799, 811, 822, 829, 830, 835, 837, 842, 843, 845, 873, 883, 897, 900, 902, 905, 913, 920, 925, 937, 938, 940, 941, 944, 949, 959]
        total_trainset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), transform=transform_train)
        testset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), split='val', transform=transform_test)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=len(total_trainset) - 50000, random_state=0)  # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)

        # filter out
        train_idx = list(filter(lambda x: total_trainset.labels[x] in idx120, train_idx))
        valid_idx = list(filter(lambda x: total_trainset.labels[x] in idx120, valid_idx))
        test_idx = list(filter(lambda x: testset.samples[x][1] in idx120, range(len(testset))))

        targets = [idx120.index(total_trainset.targets[idx]) for idx in train_idx]
        for idx in range(len(total_trainset.samples)):
            if total_trainset.samples[idx][1] not in idx120:
                continue
            total_trainset.samples[idx] = (total_trainset.samples[idx][0], idx120.index(total_trainset.samples[idx][1]))
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        for idx in range(len(testset.samples)):
            if testset.samples[idx][1] not in idx120:
                continue
            testset.samples[idx] = (testset.samples[idx][0], idx120.index(testset.samples[idx][1]))
        testset = Subset(testset, test_idx)
        print('reduced_imagenet train=', len(total_trainset))
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    if isinstance(total_trainset, AdapAugData) and total_trainset.policies is None and total_trainset.controller is not None:
        with torch.no_grad():
            temp_loader = torch.utils.data.DataLoader(
                        total_trainset, batch_size=batch*batch_multiplier, shuffle=False, num_workers=4,
                        drop_last=False)
            policies = []
            log_probs = []
            total_trainset.controller.eval()
            for data, _ in temp_loader:
                mpolicy = []
                mlog_prob = []
                for m in range(batch_multiplier):
                    log_prob, _, sampled_policies = total_trainset.controller(data.cuda())
                    mpolicy.append(sampled_policies.detach().cpu())
                    mlog_prob.append(log_prob.detach().cpu())
                policies.append(torch.stack(mpolicy)) # [M, datalen, ...]
                log_probs.append(torch.stack(mlog_prob)) # [M, datalen]
            total_trainset.policies  = torch.cat(policies, dim=1)
            total_trainset.log_probs = torch.cat(log_probs, dim=1)
            if batch_multiplier > 1:
                total_trainset.policies  = total_trainset.policies.permute(1,0,2,3,4).cpu().numpy()
                total_trainset.log_probs = total_trainset.log_probs.T.cpu().numpy()
            else:
                total_trainset.policies  = total_trainset.policies[0].cpu().numpy()
                total_trainset.log_probs = total_trainset.log_probs[0].cpu().numpy()
    if split > 0.0:
        if train_idx is None or valid_idx is None:
            # filter by split ratio
            sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
            sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
            for _ in range(split_idx + 1):
                train_idx, valid_idx = next(sss)
            if validation:
                sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
                sss = sss.split(list(range(len(valid_idx))), [total_trainset.targets[idx] for idx in valid_idx])
                for _ in range(split_idx + 1):
                    _val_idx, _test_idx = next(sss)
                test_idx  = [valid_idx[idx] for idx in _test_idx]
                valid_idx = [valid_idx[idx] for idx in _val_idx] # D_A
                # build testset
                total_trainset.controller = None
                testset = copy.deepcopy(total_trainset)
                testset.transform = transform_test
                testset.policies = None
                testset = Subset(testset, test_idx)
            if controller is not None: # Adv AA
                train_idx = list(train_idx) + list(valid_idx) # D_M + D_A
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx) if not rand_val else SubsetRandomSampler(valid_idx)

    else:
        train_sampler = None
        valid_sampler = SubsetSampler([])

        if train_idx is not None and valid_idx is not None:
            total_trainset = Subset(total_trainset, train_idx)

    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True if train_sampler is None else False, num_workers=8 if torch.cuda.device_count()==8 else 4, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    validloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True,
        sampler=valid_sampler, drop_last=rand_val)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=8 if torch.cuda.device_count()==8 else 4, pin_memory=True,
        drop_last=False)
    return train_sampler, trainloader, validloader, testloader


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if type(name) != str:
                    name, pr, level = (op_list[name][0].__name__, pr/10., level/10.+.1)
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
            # self.policy = policy
        return img

class EfficientNetRandomCrop:
    def __init__(self, imgsize, min_covered=0.1, aspect_ratio_range=(3./4, 4./3), area_range=(0.08, 1.0), max_attempts=10):
        assert 0.0 < min_covered
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]
        assert 0 < area_range[0] <= area_range[1]
        assert 1 <= max_attempts

        self.min_covered = min_covered
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self._fallback = EfficientNetCenterCrop(imgsize)

    def __call__(self, img):
        # https://github.com/tensorflow/tensorflow/blob/9274bcebb31322370139467039034f8ff852b004/tensorflow/core/kernels/sample_distorted_bounding_box_op.cc#L111
        original_width, original_height = img.size
        min_area = self.area_range[0] * (original_width * original_height)
        max_area = self.area_range[1] * (original_width * original_height)

        for _ in range(self.max_attempts):
            aspect_ratio = random.uniform(*self.aspect_ratio_range)
            height = int(round(math.sqrt(min_area / aspect_ratio)))
            max_height = int(round(math.sqrt(max_area / aspect_ratio)))

            if max_height * aspect_ratio > original_width:
                max_height = (original_width + 0.5 - 1e-7) / aspect_ratio
                max_height = int(max_height)
                if max_height * aspect_ratio > original_width:
                    max_height -= 1

            if max_height > original_height:
                max_height = original_height

            if height >= max_height:
                height = max_height

            height = int(round(random.uniform(height, max_height)))
            width = int(round(height * aspect_ratio))
            area = width * height

            if area < min_area or area > max_area:
                continue
            if width > original_width or height > original_height:
                continue
            if area < self.min_covered * (original_width * original_height):
                continue
            if width == original_width and height == original_height:
                return self._fallback(img)      # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L102

            x = random.randint(0, original_width - width)
            y = random.randint(0, original_height - height)
            return img.crop((x, y, x + width, y + height))

        return self._fallback(img)


class EfficientNetCenterCrop:
    def __init__(self, imgsize):
        self.imgsize = imgsize

    def __call__(self, img):
        """Crop the given PIL Image and resize it to desired size.

        Args:
            img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
        Returns:
            PIL Image: Cropped image.
        """
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)
