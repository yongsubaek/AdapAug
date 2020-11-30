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
from AdapAug.data import get_dataloaders, get_post_dataloader, Augmentation
from AdapAug.lr_scheduler import adjust_learning_rate_resnet
from AdapAug.metrics import accuracy, Accumulator, CrossEntropyLabelSmooth, Tracker
from AdapAug.networks import get_model, num_class
from warmup_scheduler import GradualWarmupScheduler
import random, numpy as np
from AdapAug.augmentations import augment_list
from torchvision.utils import save_image
from AdapAug.controller import Controller
from AdapAug.train import run_epoch

logger = get_logger('Adap AutoAugment')
logger.setLevel(logging.INFO)
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
_SVHN_MEAN, _SVHN_STD = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)

def train_controller(controller, config):
    """
    1. Training Controller (Affinity)
    2. for 1 epoch:
        2-1. Training TargetNetwork 1 batch
        2-2. Training Controller 1 batch (Diversity)
    """
    ori_aug = C.get()["aug"]

    dataset = C.get()['test_dataset']
    target_path = config['target_path']
    ctl_save_path = config['ctl_save_path']
    childnet_paths = config['childnet_paths']
    mode = config['mode']
    load_search = config['load_search']
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
    # C.get()["aug"] = childaug
    _, total_loader, _, test_loader = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], 0.0)
    test_metrics = []
    for epoch in range(C.get()['epoch']):
        for train_type in ['affinity', 'diversity']:
            loader = valid_loader if train_type == 'affinity' else total_loader
            repeat = len(total_loader) // len(valid_loader) if train_type == 'affinity' else 1
            # 0. Given Target Network
            for _ in range(repeat):
                baseline = ExponentialMovingAverage(ctl_ema_weight)
                for step, (inputs, labels) in enumerate(loader):
                    batch_size = len(labels)
                    inputs, labels = inputs.cuda(), labels.cuda()
                    st = time.time()
                    # 1. Policy search
                    controller.eval()
                    log_probs, entropys, sampled_policies = controller(inputs)
                    # 2. Augment
                    with torch.no_grad():
                        batch_policies = batch_policy_decoder(sampled_policies) # (list:list:list:tuple) [batch, num_policy, n_op, 3]
                        aug_inputs, applied_policy = augment_data(inputs, batch_policies, default_transform)
                        aug_inputs = aug_inputs.cuda()
                    # TODO: childaug != clean
                    controller.train()
                    if train_type == 'affinity':
                        # 3. Get affinity with childnet
                        with torch.no_grad():
                            aug_preds = childnet(aug_inputs)
                            aug_loss = criterion(aug_preds, labels) # (tensor)[batch]
                            top1, top5 = accuracy(aug_preds, labels, (1, 5))
                            clean_loss = criterion(childnet(inputs), labels) # clean data loss
                            reward = clean_loss - aug_loss # affinity approximation
                            baseline.update(reward.mean())
                            advantages = reward - baseline.value()
                            advantages += ctl_entropy_w * entropys
                    else: # diversity
                        # 3. TargetNetwork Training
                        t_net.train()
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
                    trace['diversity'].add_dict({
                        'cnt': batch_size,
                        'time': time.time()-st,
                        'acc': top1.cpu().detach().item()*batch_size,
                        'pol_loss': pol_loss.cpu().detach().item(),
                        'reward': reward.sum().cpu().detach().item(),
                        'advantages': advantages.sum().cpu().detach().item()
                        })
                logger.info(f"\n[Train Controller {epoch+1:3d}/{C.get()['epoch']:3d}] (diversity) {trace['diversity'] / 'cnt'}")
        if (epoch+1) % 10 == 0 or epoch == C.get()['epoch']-1:
            t_net.eval()
            for data, label in test_loader:
                _batch_size = len(data)
                data, label = data.cuda(), label.cuda()
                pred = t_net(data)
                loss = criterion(pred, label).mean()
                top1, top5 = accuracy(pred, label, (1,5))
                trace['test'].add_dict({
                    'cnt': _batch_size,
                    'loss': loss.detach().cpu().item()*_batch_size,
                    'top1': top1.detach().cpu().item()*_batch_size,
                    'top5': top5.detach().cpu().item()*_batch_size,
                })
            test_metric = trace['test'] / 'cnt'
            test_metrics.append(test_metric.get_dict())
            logger.info(f"\n[Test T {epoch+1:3d}/{C.get()['epoch']:3d}] {test_metric}")
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
                        'test_trace': dict(trace['test'].trace),
                        'test_metrics': test_metrics,
                        'last_policy': batch_policies
                        }, ctl_save_path)
        for k in trace:
            trace[k].reset_accum()

    C.get()["aug"] = ori_aug
    return trace, t_net

def train_controller2(controller, config):
    """
    Adv AA training scheme
    0. Training Controller (Affinity)
    1. Training TargetNetwork 1 epoch
    2. Training Controller 1 epoch (Diversity)
    """
    ori_aug = C.get()["aug"]

    dataset = C.get()['test_dataset']
    target_path = config['target_path']
    ctl_save_path = config['ctl_save_path']
    childnet_paths = config['childnet_paths']
    mode = config['mode']
    load_search = config['load_search']
    childaug = config['childaug']
    eps_clip = 0.1
    # ctl_train_steps = 1500
    ctl_num_aggre = 1
    ctl_entropy_w = 1e-5
    ctl_ema_weight = 0.95

    controller.train()
    c_optimizer = optim.Adam(controller.parameters(), lr = config['c_lr'])#, weight_decay=1e-6)
    # controller = DataParallel(controller).cuda()

    # load childnet weights
    childnet = get_model(C.get()['model'], num_class(dataset), local_rank=-1).cuda()
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
    # childnet = DataParallel(childnet).cuda()
    childnet.eval()

    # create a TargetNetwork
    t_net = get_model(C.get()['model'], num_class(dataset), local_rank=-1).cuda()
    t_optimizer, t_scheduler = get_optimizer(t_net)
    wd = C.get()['optimizer']['decay']
    grad_clip = C.get()['optimizer'].get('clip', 5.0)
    params_without_bn = [params for name, params in t_net.named_parameters() if not ('_bn' in name or '.bn' in name)]
    criterion = CrossEntropyLabelSmooth(num_class(dataset), C.get().conf.get('lb_smooth', 0), reduction="batched_sum").cuda()
    _criterion = CrossEntropyLabelSmooth(num_class(dataset), C.get().conf.get('lb_smooth', 0)).cuda()
    # t_net = DataParallel(t_net).cuda()

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
    C.get()["aug"] = "default"#childaug
    _, total_loader, _, test_loader = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], 0.0, controller=controller)
    ### Training Loop
    test_metrics = []
    total_t_train_time = 0.
    # Warm-up
    for epoch in range(C.get()['lr_schedule']['warmup']['epoch']):
        ts = time.time()
        t_net.train()
        t_tracker, metrics = run_epoch(t_net, total_loader, criterion, t_optimizer, desc_default='T-train', epoch=epoch+1, scheduler=t_scheduler, wd=C.get()['optimizer']['decay'], verbose=False, trace=True)
        total_t_train_time += time.time() - ts
        logger.info(f"[T-train] {epoch+1}/{C.get()['epoch']} (time {total_t_train_time:.1f}) {metrics}")

    for epoch in range(C.get()['lr_schedule']['warmup']['epoch'], C.get()['epoch']):
        ## Affinity Training
        controller.train()
        baseline = ExponentialMovingAverage(ctl_ema_weight)
        for _ in range(len(total_loader.dataset)//len(valid_loader.dataset)):
            for step, (inputs, labels) in enumerate(valid_loader):
                st = time.time()
                batch_size = len(labels)
                inputs, labels = inputs.cuda(), labels.cuda()
                log_probs, entropys, sampled_policies = controller(inputs)
                with torch.no_grad():
                    aug_inputs, _ = augment_data(inputs, sampled_policies, default_transform)
                    aug_inputs = aug_inputs.cuda()
                # TODO: childaug != clean
                with torch.no_grad():
                    aug_preds = childnet(aug_inputs)
                    aug_loss = criterion(aug_preds, labels) # (tensor)[batch]
                    top1, top5 = accuracy(aug_preds, labels, (1, 5))
                    clean_loss = criterion(childnet(inputs), labels) # clean data loss
                    reward = clean_loss - aug_loss  # affinity approximation
                    baseline.update(reward.mean())
                    advantages = reward - baseline.value()
                    advantages += ctl_entropy_w * entropys
                if mode == "reinforce":
                    pol_loss = -1 * (log_probs * advantages).sum() #scalar tensor
                elif mode == 'ppo':
                    old_log_probs = log_probs.detach()
                    ratios = (log_probs - old_log_probs).exp()
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                    pol_loss = -torch.min(surr1, surr2).sum()
                pol_loss = pol_loss / ctl_num_aggre
                pol_loss.backward(retain_graph=ctl_num_aggre>1)
                if step % ctl_num_aggre == 0:
                    torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
                    c_optimizer.step()
                    c_optimizer.zero_grad()
                trace['affinity'].add_dict({
                    'cnt': batch_size,
                    'time': time.time()-st,
                    'acc': 100*top1.cpu().detach().item()*batch_size,
                    'pol_loss': pol_loss.cpu().detach().item(),
                    'reward': reward.sum().cpu().detach().item(),
                    'advantages': advantages.sum().cpu().detach().item()
                    })
            logger.info(f"\n(Affinity)[Train Controller {epoch+1:3d}/{C.get()['epoch']:3d}] {trace['affinity'] / 'cnt'}")
        ## TargetNetwork Training
        ts = time.time()
        _, total_loader, _, _ = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], 0.0, controller=controller)
        t_net.train()
        t_tracker, metrics = run_epoch(t_net, total_loader, criterion, t_optimizer, desc_default='T-train', epoch=epoch+1, scheduler=t_scheduler, wd=C.get()['optimizer']['decay'], verbose=False, trace=True)
        total_t_train_time += time.time() - ts
        logger.info(f"[T-train] {epoch+1}/{C.get()['epoch']} (time {total_t_train_time:.1f}) {metrics}")
        ## Diversity Training from TargetNetwork trace
        controller.train()
        t_dict = t_tracker.get_dict()
        baseline = ExponentialMovingAverage(ctl_ema_weight)
        for step, (inputs, labels) in enumerate(t_dict['clean_data']):
            batch_size = len(labels)
            inputs, labels = inputs.cuda(), labels.cuda()
            aug_loss = t_dict['loss'][step].cuda()
            policy = t_dict['policy'][step].cuda()
            top1 = t_dict['acc'][step]
            st = time.time()
            log_probs, entropys, sampled_policies = controller(inputs, policy)
            with torch.no_grad():
                # clean_loss = criterion(t_net(inputs), labels) # clean data loss
                reward = aug_loss #- clean_loss
                baseline.update(reward.mean())
                advantages = reward - baseline.value()
                advantages += ctl_entropy_w * entropys
            if mode == "reinforce":
                pol_loss = -1 * (log_probs * advantages).sum() #scalar tensor
            elif mode == 'ppo':
                old_log_probs = t_dict['log_probs'][step].cuda()
                ratios = (log_probs - old_log_probs).exp()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                pol_loss = -torch.min(surr1, surr2).sum()
            pol_loss = pol_loss / ctl_num_aggre
            pol_loss.backward(retain_graph=ctl_num_aggre>1)
            if step % ctl_num_aggre == 0:
                torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
                c_optimizer.step()
                c_optimizer.zero_grad()
            trace['diversity'].add_dict({
                'cnt': batch_size,
                'time': time.time()-st,
                'acc': top1*batch_size,
                'pol_loss': pol_loss.cpu().detach().item(),
                'reward': reward.sum().cpu().detach().item(),
                'advantages': advantages.sum().cpu().detach().item()
                })
        logger.info(f"\n(Diversity)[Train Controller {epoch+1:3d}/{C.get()['epoch']:3d}] {trace['diversity'] / 'cnt'}")

        if (epoch+1) % 10 == 0 or epoch == C.get()['epoch']-1:
            # TargetNetwork Test
            t_net.eval()
            for data, label in test_loader:
                _batch_size = len(data)
                data, label = data.cuda(), label.cuda()
                pred = t_net(data)
                loss = criterion(pred, label).mean()
                top1, top5 = accuracy(pred, label, (1,5))
                trace['test'].add_dict({
                    'cnt': _batch_size,
                    'loss': loss.detach().cpu().item()*_batch_size,
                    'top1': top1.detach().cpu().item()*_batch_size,
                    'top5': top5.detach().cpu().item()*_batch_size,
                })
            test_metric = trace['test'] / 'cnt'
            test_metrics.append(test_metric.get_dict())
            logger.info(f"[Test T {epoch+1:3d}/{C.get()['epoch']:3d}] {test_metric}")
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
                        'policy': t_dict['policy']
                        }, target_path)
            torch.save({
                        'epoch': epoch,
                        'ctl_state_dict': controller.state_dict(),
                        'optimizer_state_dict': c_optimizer.state_dict(),
                        'aff_trace': dict(trace['affinity'].trace),
                        'div_trace': dict(trace['diversity'].trace),
                        'test_trace': dict(trace['test'].trace),
                        'test_metrics': test_metrics
                        }, ctl_save_path)
        for k in trace:
            trace[k].reset_accum()
    C.get()["aug"] = ori_aug
    return trace, t_net

def train_controller3(controller, config):
    """
    Minimal Backward
    0. Training TargetNetwork 1 epoch
    1. Training Controller (weigheted sum of Affinity & Diversity)
    """
    ori_aug = C.get()["aug"]
    dataset = C.get()['test_dataset']
    target_path = config['target_path']
    ctl_save_path = config['ctl_save_path']
    childnet_paths = config['childnet_paths']
    mode = config['mode']
    load_search = config['load_search']
    childaug = config['childaug']
    eps_clip = 0.1
    # ctl_train_steps = 1500
    ctl_num_aggre = 1
    ctl_entropy_w = 1e-5
    ctl_ema_weight = 0.95

    aff_w = 1.
    div_w = 1.

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
    v_loader_iter = iter(valid_loader)
    # C.get()["aug"] = childaug
    _, total_loader, _, test_loader = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], 0.0)
    ### Training Loop
    test_metrics = []
    total_t_train_time = 0.
    for epoch in range(C.get()['epoch']):
        ## TargetNetwork Training
        t_net.train()
        ts = time.time()
        t_loader = AdapAugloader(total_loader, controller, default_transform)
        t_tracker, metric = run_epoch(t_net, t_loader, criterion, t_optimizer, desc_default='T-train', epoch=epoch+1, scheduler=t_scheduler, wd=C.get()['optimizer']['decay'], verbose=False, trace=True)
        total_t_train_time += time.time() - ts
        logger.info(f"[T-train] {epoch+1}/{C.get()['epoch']} (time {total_t_train_time:.1f}) {metric}")
        ## Controller Training
        controller.train()
        aff_baseline = ExponentialMovingAverage(ctl_ema_weight)
        div_baseline = ExponentialMovingAverage(ctl_ema_weight)
        t_dict = t_tracker.get_dict()
        for step, (t_inputs, t_labels) in enumerate(t_dict['clean_data']):
            try:
                v_inputs, v_labels = next(v_loader_iter)
            except StopIteration:
                v_loader_iter = iter(valid_loader)
                v_inputs, v_labels = next(v_loader_iter)
            t_batch_size = len(t_inputs)
            v_batch_size = len(v_inputs)
            # Get Affinity
            t_inputs, t_labels, v_inputs, v_labels = t_inputs.cuda(), t_labels.cuda(), v_inputs.cuda(), v_labels.cuda()
            st = time.time()
            log_probs, entropys, sampled_policies = controller(v_inputs)
            with torch.no_grad():
                batch_policies = batch_policy_decoder(sampled_policies) # (list:list:list:tuple) [batch, num_policy, n_op, 3]
                aug_inputs, applied_policy = augment_data(v_inputs, batch_policies, default_transform)
                aug_inputs = aug_inputs.cuda()
            # TODO: childaug != clean
            with torch.no_grad():
                aug_preds = childnet(aug_inputs)
                aug_loss = criterion(aug_preds, v_labels) # (tensor)[batch]
                top1, top5 = accuracy(aug_preds, v_labels, (1, 5))
                clean_loss = criterion(childnet(v_inputs), v_labels) # clean data loss
                reward = clean_loss - aug_loss # affinity approximation
                aff_baseline.update(reward.mean())
                advantages = reward - aff_baseline.value()
                advantages += ctl_entropy_w * entropys
            if mode == "reinforce":
                pol_loss = -1 * (log_probs * advantages).sum() #scalar tensor
            elif mode == 'ppo':
                old_log_probs = log_probs.detach()
                ratios = (log_probs - old_log_probs).exp()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                pol_loss = -torch.min(surr1, surr2).sum()
            aff_loss = aff_w * pol_loss
            trace['affinity'].add_dict({
                'cnt': v_batch_size,
                'acc': top1.cpu().detach().item()*v_batch_size,
                'pol_loss': pol_loss.cpu().detach().item(),
                'reward': reward.sum().cpu().detach().item(),
                'advantages': advantages.sum().cpu().detach().item()
                })
            logger.info(f"\n[Train Controller {epoch+1:3d}/{C.get()['epoch']:3d}] (affinity) {trace['affinity'] / 'cnt'}")
            ## Get Diversity from TargetNetwork trace
            aug_loss = t_dict['loss'][step].cuda()
            policy = t_dict['policy'][step].cuda()
            st = time.time()
            log_probs, entropys, sampled_policies = controller(t_inputs, policy)
            with torch.no_grad():
                # clean_loss = criterion(t_net(inputs), labels) # clean data loss
                reward = aug_loss #- clean_loss
                div_baseline.update(reward.mean())
                advantages = reward - div_baseline.value()
                advantages += ctl_entropy_w * entropys
            if mode == "reinforce":
                pol_loss = -1 * (log_probs * advantages).sum() #scalar tensor
            elif mode == 'ppo':
                old_log_probs = t_dict['log_probs'][step].cuda()
                ratios = (log_probs - old_log_probs).exp()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                pol_loss = -torch.min(surr1, surr2).sum()
            div_loss = div_w * pol_loss
            total_pol_loss = aff_loss + div_loss
            total_pol_loss = total_pol_loss / ctl_num_aggre
            total_pol_loss.backward(retain_graph=ctl_num_aggre>1)
            if step % ctl_num_aggre == 0:
                torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
                c_optimizer.step()
                c_optimizer.zero_grad()
            trace['diversity'].add_dict({
                'cnt': t_batch_size,
                'time': time.time()-st,
                'acc': top1.cpu().detach().item()*t_batch_size,
                'pol_loss': pol_loss.cpu().detach().item(),
                'reward': reward.sum().cpu().detach().item(),
                'advantages': advantages.sum().cpu().detach().item()
                })
        logger.info(f"\n[Train Controller {epoch+1:3d}/{C.get()['epoch']:3d}] (diversity) {trace['diversity'] / 'cnt'}")
        if (epoch+1) % 10 == 0 or epoch == C.get()['epoch']-1:
            # TargetNetwork Test
            t_net.eval()
            for data, label in test_loader:
                _batch_size = len(data)
                data, label = data.cuda(), label.cuda()
                pred = t_net(data)
                loss = criterion(pred, label).mean()
                top1, top5 = accuracy(pred, label, (1,5))
                trace['test'].add_dict({
                    'cnt': _batch_size,
                    'loss': loss.detach().cpu().item()*_batch_size,
                    'top1': top1.detach().cpu().item()*_batch_size,
                    'top5': top5.detach().cpu().item()*_batch_size,
                })
            test_metric = trace['test'] / 'cnt'
            test_metrics.append(test_metric.get_dict())
            logger.info(f"\n[Test T {epoch+1:3d}/{C.get()['epoch']:3d}] {test_metric}")
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
                        'test_trace': dict(trace['test'].trace),
                        'test_metrics': test_metrics,
                        'last_policy': batch_policies
                        }, ctl_save_path)
        for k in trace:
            trace[k].reset_accum()

    C.get()["aug"] = ori_aug
    return trace, t_net

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
        transform: transfroms after augment
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
        aug_img = Augmentation(policy)(pil_img)
        aug_img = transform(aug_img)
        aug_imgs.append(aug_img)
        # applied_policy.append(augment.policy)
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
