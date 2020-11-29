import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import itertools
import json
import logging
import math, time
import os
from collections import OrderedDict

import torch
from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torchvision import transforms

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser

from AdapAug.common import get_logger, EMA, add_filehandler, get_optimizer
from AdapAug.data import get_dataloaders, get_post_dataloader, Augmentation, CutoutDefault
from AdapAug.lr_scheduler import adjust_learning_rate_resnet
from AdapAug.metrics import accuracy, Accumulator, CrossEntropyLabelSmooth, Tracker
from AdapAug.networks import get_model, num_class
from AdapAug.tf_port.rmsprop import RMSpropTF
from AdapAug.aug_mixup import CrossEntropyMixUpLabelSmooth, mixup
from warmup_scheduler import GradualWarmupScheduler
import random, numpy as np
from AdapAug.augmentations import get_augment, augment_list
from torchvision.utils import save_image
from AdapAug.archive import fa_reduced_cifar10
from AdapAug.controller import Controller

logger = get_logger('Adap AutoAugment')
logger.setLevel(logging.INFO)
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
_SVHN_MEAN, _SVHN_STD = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)

class ExponentialMovingAverage(object):
  """Class that maintains an exponential moving average."""

  def __init__(self, momentum):
    self._numerator   = 0
    self._denominator = 0
    self._momentum    = momentum

  def update(self, value):
    self._numerator = self._momentum * self._numerator + (1 - self._momentum) * value
    self._denominator = self._momentum * self._denominator + (1 - self._momentum)

  def value(self):
    """Return the current value of the moving average"""
    return self._numerator / self._denominator


class AdapAugloader(object):
    """
    Wraper loader
    """
    def __init__(self, dataloader, controller=None):
        self.dataloader = dataloader
        self.controller = controller
        if self.controller:
            self.controller.eval()

    def __iter__(self):
        self.loader_iter = iter(self.dataloader)
        return self

    def __next__(self):
        inputs, labels = next(self.loader_iter)
        if self.controller:
            # ! original image to controller(only normalized)
            # ! augmented image to model
            _, _, sampled_policies = self.controller(inputs.cuda())
            batch_policies = batch_policy_decoder(sampled_policies) # (list:list:list:tuple) [batch, num_policy, n_op, 3]
            aug_inputs, applied_policy = augment_data(inputs, batch_policies)
            self.applied_policy = applied_policy
        else:
            aug_inputs = []
            for img in inputs:
                pil_img = transforms.ToPILImage()(UnNormalize()(img.cpu()))
                transform_img = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
                ])
                if C.get()['cutout'] > 0:
                    transform_img.transforms.append(CutoutDefault(C.get()['cutout']))
                if C.get()['aug'] == 'fa_reduced_cifar10':
                    transform_img.transforms.insert(0, Augmentation(fa_reduced_cifar10())) ###
                aug_img = transform_img(pil_img)
                aug_inputs.append(aug_img)
            aug_inputs = torch.stack(aug_inputs)
        return (aug_inputs, labels)

    def __len__(self):
        return len(self.dataloader)

class UnNormalize(object):
    def __init__(self, mean=_CIFAR_MEAN, std=_CIFAR_STD):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def save_pic(inputs, aug_inputs, labels, policies, batch_policies, step, verbose=False):
    if verbose:
        for i, ori_img in enumerate(inputs):
            aug_img = aug_inputs[i]
            label = labels[i]
            applied_policy = policies[i]
            policy = batch_policies[i]
            print("step: %d" % i)
            print(policy)
            print(applied_policy)
    save_path = "./results/{}/".format(C.get()['exp_name'])
    os.makedirs(save_path, exist_ok=True)
    # np.savez(save_path + "{}_policy.npz".format(step), labels=labels, policies=policies, batch_policies=batch_policies)
    save_image(inputs, save_path + "{}_ori.png".format(step))
    save_image(aug_inputs, save_path + "{}_aug.png".format(step))

def augment_data(imgs, policys, transform):
    """
    arguments
        imgs: (tensor) [batch, h, w, c]; [(image)->ToTensor->Normalize]
        policys: (list:list:list:tuple) [batch, num_policy, n_op, 3]
    return
        aug_imgs: (tensor) [batch, h, w, c];
        [(image)->(policys)->RandomResizedCrop->RandomHorizontalFlip->ToTensor->Normalize->CutOut]
    """
    if "cifar" in C.get()['dataset']:
        mean, std = _CIFAR_MEAN, _CIFAR_STD
    elif "svhn" in C.get()['dataset']:
        mean, std = _SVHN_MEAN, _SVHN_STD
    aug_imgs = []
    applied_policy = []
    for img, policy in zip(imgs, policys):
        # policy: (list:list:tuple) [num_policy, n_op, 3]
        pil_img = transforms.ToPILImage()(UnNormalize(mean, std)(img.cpu()))
        augment = Augmentation(policy)
        aug_img = augment(pil_img)
        # apply original training/valid transforms
        aug_img = transform(aug_img)
        aug_imgs.append(aug_img)
        applied_policy.append(augment.policy)
    aug_imgs = torch.stack(aug_imgs)
    return aug_imgs, applied_policy

def batch_policy_decoder(augment): # augment: [batch, num_policy, n_op, 3]
    op_list = augment_list(False)
    batch_policies = []
    for policy in augment:      # policy: [num_policy, n_op, 3]
        policies = []
        for subpolicies in policy: # subpolicies: [n_op, 3]
            ops = []
            for op in subpolicies:
                op_idx, op_prob, op_level = op
                op_prob = op_prob / 10.
                op_level = op_level / 10.0 + 0.1
                # assert (0.0 <= op_prob <= 1.0) and (0.0 <= op_level <= 1.0), f"prob {op_prob}, level {op_level}"
                ops.append((op_list[op_idx][0].__name__, op_prob, op_level))
            policies.append(ops)
        batch_policies.append(policies)
    return batch_policies # (list:list:list:tuple) [batch, num_policy, n_op, 3]

def train_controller(controller, config, load_search=False):
    ori_aug = C.get()["aug"]

    dataset = C.get()['test_dataset']
    target_path = config['target_path']
    ctl_save_path = config['ctl_save_path']
    childnet_paths = config['childnet_paths']
    mode = config['mode']
    childaug = config['childaug']
    eps_clip = 0.1
    # ctl_train_steps = 1500
    ctl_num_aggre = 1
    ctl_entropy_w = 1e-5
    ctl_ema_weight = 0.95

    controller.train()
    c_optimizer = optim.Adam(controller.parameters(), lr = config['c_lr'])#, weight_decay=1e-6)
    controller = DataParallel(controller).cuda()

    # load childnet weights
    childnet = get_model(C.get()['model'], num_class(dataset), local_rank=-1)
    cv_id = 0
    data = torch.load(childnet_paths[cv_id])
    key = 'model' if 'model' in data else 'state_dict'
    if 'epoch' not in data:
        childnet.load_state_dict(data)
    else:
        logger.info('checkpoint epoch@%d' % data['epoch'])
        if not isinstance(childnet, (DataParallel, DistributedDataParallel)):
            childnet.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
        else:
            childnet.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
            del data
    childnet = DataParallel(childnet).cuda()
    childnet.eval()

    # create a TargetNetwork
    t_net = get_model(C.get()['model'], num_class(dataset), local_rank=-1)
    t_optimizer, t_scheduler = get_optimizer(t_net)
    wd = C.get()['optimizer']['decay']
    grad_clip = C.get()['optimizer'].get('clip', 5.0)
    params_without_bn = [params for name, params in t_net.named_parameters() if not ('_bn' in name or '.bn' in name)]
    criterion = CrossEntropyLabelSmooth(num_class(dataset), C.get().conf.get('lb_smooth', 0), reduction="batched_sum").cuda()
    t_net = DataParallel(t_net).cuda()

    trace = {'affinity': Tracker(),
               'diversity': Tracker(),
               'test': Tracker()}
    # load TargetNetwork weights
    if load_search and os.path.isfile(target_path):
        data = torch.load(target_path)
        key = 'model' if 'model' in data else 'state_dict'
        if 'epoch' not in data:
            t_net.load_state_dict(data)
        else:
            logger.info('checkpoint epoch@%d' % data['epoch'])
            if not isinstance(t_net, (DataParallel, DistributedDataParallel)):
                t_net.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
            else:
                t_net.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
                del data
        t_optimizer.load_state_dict(data['optimizer_state_dict'])
    # load ctl weights and results
    if load_search and os.path.isfile(ctl_save_path):
        logger.info('------Controller load------')
        checkpoint = torch.load(ctl_save_path)
        controller.module.load_state_dict(checkpoint['ctl_state_dict'])
        c_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trace['affinity'].trace = checkpoint['aff_trace']
        trace['diversity'].trace = checkpoint['div_trace']
        trace['test'].trace = checkpoint['test_trace']
    else:
        logger.info('------Train Controller from scratch------')
    # get dataloaders
    C.get()["aug"] = "clean"
    valid_loader, default_transform, child_transform = get_post_dataloader(C.get()['dataset'], C.get()['batch'], config['dataroot'], config['split_ratio'], split_idx=cv_id, rand_val=True, childaug=childaug)
    C.get()["aug"] = childaug
    _, total_loader, _, test_loader = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], 0.0)

    for epoch in range(C.get()['epoch']):
        t_net.train()
        for train_type in ['affinity', 'diversity']:
            loader = valid_loader if train_type == 'affinity' else total_loader
            # print(len(loader))
            # 0. Given Target Network
            baseline = ExponentialMovingAverage(ctl_ema_weight)
            for step, (inputs, labels) in enumerate(loader):
                batch_size = len(labels)
                inputs, labels = inputs.cuda(), labels.cuda()
                st = time.time()
                # 1. Policy search
                log_probs, entropys, sampled_policies = controller(inputs)
                with torch.no_grad():
                    # 2. Augment
                    batch_policies = batch_policy_decoder(sampled_policies) # (list:list:list:tuple) [batch, num_policy, n_op, 3]
                    aug_inputs, applied_policy = augment_data(inputs, batch_policies, default_transform)
                    aug_inputs = aug_inputs.cuda()
                # TODO: childaug != clean
                if train_type == 'affinity':
                    # 3. Get affinity(1/loss) with childnet
                    with torch.no_grad():
                        aug_preds = childnet(aug_inputs)
                        aug_loss = criterion(aug_preds, labels) # (tensor)[batch]
                        top1, top5 = accuracy(aug_preds, labels, (1, 5))
                        clean_loss = criterion(childnet(inputs), labels) # clean data loss
                        reward = 0.1/(aug_loss - clean_loss + 1e-3) # affinity approximation
                        baseline.update(reward.mean())
                        advantages = reward - baseline.value()
                        advantages += ctl_entropy_w * entropys
                else: # diversity
                    # 3. TargetNetwork Training
                    aug_preds = t_net(aug_inputs)
                    aug_loss = criterion(aug_preds, labels) # (tensor)[batch]
                    top1, top5 = accuracy(aug_preds, labels, (1, 5))
                    with torch.no_grad():
                        clean_loss = criterion(t_net(inputs), labels) # clean data loss
                        reward = aug_loss - clean_loss # diversity approximation
                        baseline.update(reward.mean())
                        advantages = reward - baseline.value()
                        advantages += ctl_entropy_w * entropys
                    t_loss = aug_loss.mean()
                    t_loss += wd * (1. / 2.) * sum([torch.sum(p ** 2) for p in params_without_bn])
                    t_loss.backward()
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(t_net.parameters(), grad_clip)
                    t_optimizer.step()
                    t_optimizer.zero_grad()

                if mode == "reinforce":
                    pol_loss = -1 * (log_probs * advantages).sum() #scalar tensor
                elif mode == 'ppo':
                    old_log_probs = log_probs.detach()
                    ratios = (log_probs - old_log_probs).exp()
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                    pol_loss = -torch.min(surr1, surr2).sum()
                # 4. Train Controller
                # Average gradient over controller_num_aggregate samples
                pol_loss = pol_loss / ctl_num_aggre
                pol_loss.backward(retain_graph=ctl_num_aggre>1)
                if step % ctl_num_aggre == 0:
                    torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
                    c_optimizer.step()
                    c_optimizer.zero_grad()
                trace[train_type].add_dict({
                    'cnt': batch_size,
                    'time': time.time()-st,
                    'acc': top1.cpu().detach().item()*batch_size,
                    'pol_loss': pol_loss.cpu().detach().item(),
                    'reward': reward.sum().cpu().detach().item(),
                    'advantages': advantages.sum().cpu().detach().item()
                    })
                # print(f"{trace[train_type] / 'cnt'}")
            logger.info(f"\n[Train Controller {epoch+1:3d}/{C.get()['epoch']:3d}] ({train_type}) {trace[train_type] / 'cnt'}")
        if (epoch+1) % 10 == 0 or epoch == C.get()['epoch']-1:
            for data, label in test_loader:
                _batch_size = len(data)
                data, label = data.cuda(), label.cuda()
                pred = t_net(data)
                loss = criterion(pred, label).sum()
                top1, top5 = accuracy(pred, label, (1,5))
                trace['test'].add_dict({
                    'cnt': _batch_size,
                    'loss': loss.detach().cpu().item(),
                    'top1': top1.detach().cpu().item()*_batch_size,
                    'top5': top5.detach().cpu().item()*_batch_size,
                })
            logger.info(f"\n[Test T {epoch+1:3d}/{C.get()['epoch']:3d}] {trace['test'] / 'cnt'}")
            # update cv_id
            cv_id = (cv_id+1) % config['cv_num']
            data = torch.load(childnet_paths[cv_id])
            key = 'model' if 'model' in data else 'state_dict'
            if 'epoch' not in data:
                childnet.load_state_dict(data)
            else:
                if not isinstance(childnet, (DataParallel, DistributedDataParallel)):
                    childnet.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
                else:
                    childnet.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
                    del data
            torch.save({
                        'epoch': epoch,
                        'model':t_net.state_dict(),
                        'optimizer_state_dict': t_optimizer.state_dict(),
                        }, target_path)
            torch.save({
                        'epoch': epoch,
                        'ctl_state_dict': controller.state_dict(),
                        'optimizer_state_dict': c_optimizer.state_dict(),
                        'aff_trace': dict(trace['affinity'].trace),
                        'div_trace': dict(trace['diversity'].trace),
                        'last_policy': batch_policies
                        }, ctl_save_path)
        for k in trace:
            trace[k].reset_accum()

    C.get()["aug"] = ori_aug
    return trace, t_net

def train_and_eval_ctl(tag, controller, dataroot, test_ratio=0.0, cv_fold=0, reporter=None, metric='last', save_path=None, only_eval=False, local_rank=-1, evaluation_interval=5):
    """
    training on augmented data
    """
    controller.eval()
    total_batch = C.get()["batch"]
    dataset = C.get()['test_dataset']
    is_master = local_rank < 0 or dist.get_rank() == 0
    if is_master:
        add_filehandler(logger, save_path + '.log')

    if not reporter:
        reporter = lambda **kwargs: 0

    max_epoch = C.get()['epoch']
    trainsampler, trainloader, validloader, testloader_ = get_dataloaders(dataset, C.get()['batch'], dataroot, test_ratio, split_idx=cv_fold, multinode=(local_rank >= 0))
    trainloader = AdapAugloader(trainloader, controller)
    # create a model & an optimizer
    model = get_model(C.get()['model'], num_class(dataset), local_rank=local_rank)
    model_ema = get_model(C.get()['model'], num_class(dataset), local_rank=-1)
    model_ema.eval()
    criterion_ce = criterion = CrossEntropyLabelSmooth(num_class(dataset), 0)
    if C.get()['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=C.get()['lr'],
            momentum=C.get()['optimizer'].get('momentum', 0.9),
            weight_decay=0.0,
            nesterov=C.get()['optimizer'].get('nesterov', True)
        )
    elif C.get()['optimizer']['type'] == 'rmsprop':
        optimizer = RMSpropTF(
            model.parameters(),
            lr=C.get()['lr'],
            weight_decay=0.0,
            alpha=0.9, momentum=0.9,
            eps=0.001
        )
    else:
        raise ValueError('invalid optimizer type=%s' % C.get()['optimizer']['type'])

    lr_scheduler_type = C.get()['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.get()['epoch'], eta_min=0.)
    elif lr_scheduler_type == 'resnet':
        scheduler = adjust_learning_rate_resnet(optimizer)
    elif lr_scheduler_type == 'efficientnet':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.97 ** int((x + C.get()['lr_schedule']['warmup']['epoch']) / 2.4))
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

    if C.get()['lr_schedule'].get('warmup', None) and C.get()['lr_schedule']['warmup']['epoch'] > 0:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
            total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )

    if not tag or not is_master:
        from AdapAug.metrics import SummaryWriterDummy as SummaryWriter
        logger.warning('tag not provided, no tensorboard log.')
    else:
        from tensorboardX import SummaryWriter
    writers = [SummaryWriter(log_dir='./logs/%s/%s' % (tag, x)) for x in ['train', 'valid', 'test']]

    if C.get()['optimizer']['ema'] > 0.0 and is_master:
        # https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4?u=ildoonet
        ema = EMA(C.get()['optimizer']['ema'])
    else:
        ema = None

    result = OrderedDict()
    epoch_start = 1
    if save_path != 'test.pth':     # and is_master: --> should load all data(not able to be broadcasted)
        if save_path and os.path.exists(save_path):
            logger.info('%s file found. loading...' % save_path)
            data = torch.load(save_path)
            key = 'model' if 'model' in data else 'state_dict'

            if 'epoch' not in data:
                model.load_state_dict(data)
            else:
                logger.info('checkpoint epoch@%d' % data['epoch'])
                if not isinstance(model, (DataParallel, DistributedDataParallel)):
                    model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
                else:
                    model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
                logger.info('optimizer.load_state_dict+')
                optimizer.load_state_dict(data['optimizer'])
                if data['epoch'] < C.get()['epoch']:
                    epoch_start = data['epoch']
                else:
                    only_eval = True
                if ema is not None:
                    ema.shadow = data.get('ema', {}) if isinstance(data.get('ema', {}), dict) else data['ema'].state_dict()
            del data
        else:
            logger.info('"%s" file not found. skip to pretrain weights...' % save_path)
            if only_eval:
                logger.warning('model checkpoint not found. only-evaluation mode is off.')
            only_eval = False

    tqdm_disabled = bool(os.environ.get('TASK_NAME', '')) and local_rank != 0  # KakaoBrain Environment

    if only_eval:
        logger.info('evaluation only+')
        model.eval()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, None, desc_default='train', epoch=0, writer=writers[0], is_master=is_master)

        with torch.no_grad():
            rs['valid'] = run_epoch(model, validloader, criterion, None, desc_default='valid', epoch=0, writer=writers[1], is_master=is_master)
            rs['test'] = run_epoch(model, testloader_, criterion, None, desc_default='*test', epoch=0, writer=writers[2], is_master=is_master)
            if ema is not None and len(ema) > 0:
                model_ema.load_state_dict({k.replace('module.', ''): v for k, v in ema.state_dict().items()})
                rs['valid'] = run_epoch(model_ema, validloader, criterion_ce, None, desc_default='valid(EMA)', epoch=0, writer=writers[1], verbose=is_master, tqdm_disabled=tqdm_disabled)
                rs['test'] = run_epoch(model_ema, testloader_, criterion_ce, None, desc_default='*test(EMA)', epoch=0, writer=writers[2], verbose=is_master, tqdm_disabled=tqdm_disabled)
        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
            if setname not in rs:
                continue
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result

    # train loop
    best_top1 = 0
    for epoch in range(epoch_start, max_epoch + 1):
        model.train()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, optimizer, desc_default='train', epoch=epoch, writer=writers[0], verbose=(is_master and local_rank <= 0), scheduler=scheduler, ema=ema, wd=C.get()['optimizer']['decay'], tqdm_disabled=tqdm_disabled)
        model.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if ema is not None and C.get()['optimizer']['ema_interval'] > 0 and epoch % C.get()['optimizer']['ema_interval'] == 0:
            logger.info(f'ema synced+ rank={dist.get_rank()}')
            if ema is not None:
                model.load_state_dict(ema.state_dict())
            for name, x in model.state_dict().items():
                # print(name)
                dist.broadcast(x, 0)
            torch.cuda.synchronize()
            logger.info(f'ema synced- rank={dist.get_rank()}')

        if is_master and (epoch % evaluation_interval == 0 or epoch == max_epoch):
            with torch.no_grad():
                rs['valid'] = run_epoch(model, validloader, criterion_ce, None, desc_default='valid', epoch=epoch, writer=writers[1], verbose=is_master, tqdm_disabled=tqdm_disabled)
                rs['test'] = run_epoch(model, testloader_, criterion_ce, None, desc_default='*test', epoch=epoch, writer=writers[2], verbose=is_master, tqdm_disabled=tqdm_disabled)

                if ema is not None:
                    model_ema.load_state_dict({k.replace('module.', ''): v for k, v in ema.state_dict().items()})
                    rs['valid'] = run_epoch(model_ema, validloader, criterion_ce, None, desc_default='valid(EMA)', epoch=epoch, writer=writers[1], verbose=is_master, tqdm_disabled=tqdm_disabled)
                    rs['test'] = run_epoch(model_ema, testloader_, criterion_ce, None, desc_default='*test(EMA)', epoch=epoch, writer=writers[2], verbose=is_master, tqdm_disabled=tqdm_disabled)

            logger.info(
                f'epoch={epoch} '
                f'[train] loss={rs["train"]["loss"]:.4f} top1={rs["train"]["top1"]:.4f} '
                f'[valid] loss={rs["valid"]["loss"]:.4f} top1={rs["valid"]["top1"]:.4f} '
                f'[test] loss={rs["test"]["loss"]:.4f} top1={rs["test"]["top1"]:.4f} '
            )

            if metric == 'last' or rs[metric]['top1'] > best_top1:
                if metric != 'last':
                    best_top1 = rs[metric]['top1']
                for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
                    result['%s_%s' % (key, setname)] = rs[setname][key]
                result['epoch'] = epoch

                writers[1].add_scalar('valid_top1/best', rs['valid']['top1'], epoch)
                writers[2].add_scalar('test_top1/best', rs['test']['top1'], epoch)

                reporter(
                    loss_valid=rs['valid']['loss'], top1_valid=rs['valid']['top1'],
                    loss_test=rs['test']['loss'], top1_test=rs['test']['top1']
                )

                # save checkpoint
                if is_master and save_path:
                    logger.info('save model@%d to %s, err=%.4f' % (epoch, save_path, 1 - result['top1_test']))#best_top1))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict(),
                        'ema': ema.state_dict() if ema is not None else None,
                    }, save_path)

    del model

    # result['top1_test'] = best_top1
    return result
