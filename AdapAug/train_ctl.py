import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import itertools
import json
import logging
import math, time
import os, copy
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
    Adversarial AutoAugment training scheme without image
    1. Training TargetNetwork 1 epoch
    2. Training Controller 1 step (Diversity)
    """
    controller = controller.cuda()
    # ori_aug = C.get()["aug"]

    dataset = C.get()['test_dataset']
    target_path = config['target_path']
    ctl_save_path = config['ctl_save_path']
    mode = config['mode']
    load_search = config['load_search']
    batch_multiplier = config['M']

    eps_clip = 0.2
    ctl_entropy_w = 1e-5
    ctl_ema_weight = 0.95

    controller.train()
    c_optimizer = optim.Adam(controller.parameters(), lr = config['c_lr'])#, weight_decay=1e-6)

    # create a TargetNetwork
    t_net = get_model(C.get()['model'], num_class(dataset), local_rank=-1).cuda()
    t_optimizer, t_scheduler = get_optimizer(t_net)
    wd = C.get()['optimizer']['decay']
    grad_clip = C.get()['optimizer'].get('clip', 5.0)
    params_without_bn = [params for name, params in t_net.named_parameters() if not ('_bn' in name or '.bn' in name)]
    criterion = CrossEntropyLabelSmooth(num_class(dataset), C.get().conf.get('lb_smooth', 0), reduction="batched_sum").cuda()
    _criterion = CrossEntropyLabelSmooth(num_class(dataset), C.get().conf.get('lb_smooth', 0)).cuda()
    if batch_multiplier > 1:
        t_net = DataParallel(t_net).cuda()
        if controller.img_input:
            controller = DataParallel(controller).cuda()
    trace = {'diversity': Tracker()}
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
        trace['diversity'].trace = checkpoint['div_trace']
    else:
        logger.info('------Train Controller from scratch------')
    ### Training Loop
    test_metrics = []
    total_t_train_time = 0.
    baseline = ExponentialMovingAverage(ctl_ema_weight)
    for epoch in range(C.get()['epoch']):
        ## TargetNetwork Training
        ts = time.time()
        log_probs=[]
        entropys=[]
        sampled_policies=[]
        for m in range(batch_multiplier):
            log_prob, entropy, sampled_policy = controller()
            log_probs.append(log_prob)
            entropys.append(entropy)
            sampled_policies.append(sampled_policy.detach().cpu())
        log_probs = torch.cat(log_probs)
        entropys = torch.cat(entropys)
        sampled_policies = list(torch.cat(sampled_policies).numpy()) if batch_multiplier > 1 else list(sampled_policies[0][0].numpy()) # (M, num_op, num_p, num_m)
        _, total_loader, _, test_loader = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], 0.0, _transform=sampled_policies, batch_multiplier=batch_multiplier)
        t_net.train()
        # training and return M normalized moving averages of losses
        metrics = run_epoch(t_net, total_loader, criterion if batch_multiplier>1 else _criterion, t_optimizer, desc_default='T-train', epoch=epoch+1, scheduler=t_scheduler, wd=C.get()['optimizer']['decay'], verbose=False, \
                            batch_multiplier=batch_multiplier)
        total_t_train_time += time.time() - ts
        logger.info(f"[T-train] {epoch+1}/{C.get()['epoch']} (time {total_t_train_time:.1f}) {metrics}")
        ## Diversity Training from TargetNetwork trace
        st = time.time()
        controller.train()
        rewards = metrics['loss']
        if batch_multiplier > 1:
            advantages = metrics.norm_loss.cuda()
        else:
            baseline.update(rewards)
            advantages = rewards - baseline.value()
        if mode == "reinforce":
            pol_loss = -1 * (log_probs * advantages)
        elif mode == 'ppo':
            old_log_probs = log_probs.detach()
            ratios = (log_probs - old_log_probs).exp()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
            pol_loss = -torch.min(surr1, surr2)
        pol_loss = (pol_loss - ctl_entropy_w * entropys).mean()
        pol_loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        c_optimizer.step()
        c_optimizer.zero_grad()
        trace['diversity'].add_dict({
            'cnt' : 1,
            'time': time.time()-st,
            'acc': metrics["top1"],
            'pol_loss': pol_loss.cpu().detach().item(),
            'reward': rewards,
            })
        logger.info(f"(Diversity){epoch+1:3d}/{C.get()['epoch']:3d} {trace['diversity'].last()}")

        if (epoch+1) % 10 == 0 or epoch == C.get()['epoch']-1:
            # TargetNetwork Test
            t_net.eval()
            test_metric = run_epoch(t_net, test_loader, _criterion, None, desc_default='test T', epoch=epoch+1, verbose=False)
            test_metrics.append(test_metric.get_dict())
            logger.info(f"[Test T {epoch+1:3d}/{C.get()['epoch']:3d}] {test_metric}")
            torch.save({
                        'epoch': epoch,
                        'model':t_net.state_dict(),
                        'optimizer_state_dict': t_optimizer.state_dict(),
                        'policy': sampled_policies,
                        'test_metrics': test_metrics
                        }, target_path)
            torch.save({
                        'epoch': epoch,
                        'ctl_state_dict': controller.state_dict(),
                        'optimizer_state_dict': c_optimizer.state_dict(),
                        'div_trace': dict(trace['diversity'].trace),
                        }, ctl_save_path)
    # C.get()["aug"] = ori_aug
    return trace, test_metrics

def train_controller2(controller, config):
    """
    Adv AA training scheme with image
    0. Training Controller (Affinity)
    1. Training TargetNetwork 1 epoch
    2. Training Controller 1 epoch (Diversity)
    """
    controller = controller.cuda()
    # ori_aug = C.get()["aug"]

    dataset = C.get()['test_dataset']
    target_path = config['target_path']
    ctl_save_path = config['ctl_save_path']
    childnet_paths = config['childnet_paths']
    mode = config['mode']
    load_search = config['load_search']
    childaug = config['childaug']
    ctl_train_steps = config['ctl_train_steps']
    batch_multiplier = config['M']

    eps_clip = 0.2
    ctl_num_aggre = config['ctl_num_aggre']
    ctl_entropy_w = 1e-5
    ctl_ema_weight = 0.95
    cv_id = 0 if config['cv_id'] is None else config['cv_id']

    controller.train()
    c_optimizer = optim.Adam(controller.parameters(), lr = config['c_lr'])#, weight_decay=1e-6)
    # controller = DataParallel(controller).cuda()

    # load childnet weights
    childnet = get_model(C.get()['model'], num_class(dataset), local_rank=-1).cuda()
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
             'diversity': Tracker()}
             # 'test': Tracker()}
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
        # trace['test'].trace = checkpoint['test_trace']
    else:
        logger.info('------Train Controller from scratch------')
    # get dataloaders
    # C.get()["aug"] = "clean"
    # valid_loader, default_transform, child_transform = get_post_dataloader(C.get()['dataset'], C.get()['batch'], config['dataroot'], config['split_ratio'], split_idx=cv_id, rand_val=True, childaug=childaug)
    _, _, valid_loader, _ = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], config['split_ratio'], split_idx=cv_id, rand_val=True, _transform=childaug)#, controller=controller)
    _, total_loader, _, test_loader = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], 0.0, _transform="default")#, controller=controller)
    ### Training Loop
    if ctl_train_steps is not None:
        aff_step = div_step = ctl_train_steps
    else:
        aff_step = config['aff_step']
        div_step = config['div_step']
    aff_loader_len = len(valid_loader)
    div_loader_len = len(total_loader)
    aff_train_len = aff_loader_len if aff_step is None else min(aff_loader_len, int(aff_step))
    div_train_len = div_loader_len if div_step is None else min(div_loader_len, int(div_step))

    test_metrics = []
    total_t_train_time = 0.
    for epoch in range(C.get()['epoch']):
        ## Affinity Training
        baseline = ExponentialMovingAverage(ctl_ema_weight)
        repeat = 1#len(total_loader.dataset)//len(valid_loader.dataset) if aff_step is None else 1
        for _ in range(repeat):
            _, _, valid_loader, _ = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], config['split_ratio'], split_idx=cv_id, rand_val=True, controller=controller, _transform=childaug)
            a_tracker, _ = run_epoch(childnet, valid_loader, criterion, None, desc_default='childnet tracking', epoch=epoch+1, verbose=False, \
                                     trace=True)
            controller.train()
            a_dict = a_tracker.get_dict()
            for step, (inputs, labels) in enumerate(a_dict['clean_data']):
                batch_size = len(labels)
                inputs, labels = inputs.cuda(), labels.cuda()
                aug_loss = a_dict['loss'][step].cuda()
                if step >= aff_loader_len - aff_train_len:
                    policy = a_dict['policy'][step].cuda()
                    top1 = a_dict['acc'][step]
                    st = time.time()
                    log_probs, entropys, sampled_policies = controller(inputs, policy)
                with torch.no_grad():
                    clean_loss = criterion(childnet(inputs), labels) # clean data loss
                    reward = clean_loss.detach() - aug_loss  # affinity approximation
                    baseline.update(reward.mean())
                    if step < aff_loader_len - aff_train_len: continue
                    advantages = reward - baseline.value()
                if mode == "reinforce":
                    pol_loss = -1 * (log_probs * advantages).sum() #scalar tensor
                elif mode == 'ppo':
                    old_log_probs = log_probs.detach()
                    ratios = (log_probs - old_log_probs).exp()
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                    pol_loss = -torch.min(surr1, surr2).sum()
                pol_loss -= ctl_entropy_w * entropys
                if (step+1)==aff_loader_len:
                    length = ctl_num_aggre if aff_train_len % ctl_num_aggre == 0 else aff_train_len % ctl_num_aggre
                    pol_loss = pol_loss / length
                else:
                    pol_loss = pol_loss / ctl_num_aggre
                pol_loss.backward(retain_graph=ctl_num_aggre>1)
                if (step+1) % ctl_num_aggre == 0 or (step+1)==aff_loader_len:
                    torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
                    c_optimizer.step()
                    c_optimizer.zero_grad()
                trace['affinity'].add_dict({
                    'cnt': batch_size,
                    'time': time.time()-st,
                    'acc': top1*batch_size,
                    'pol_loss': pol_loss.cpu().detach().item(),
                    'reward': reward.sum().cpu().detach().item(),
                    'advantages': advantages.sum().cpu().detach().item()
                    })
            if step >= aff_loader_len - aff_train_len:
                logger.info(f"(Affinity)[Train Controller {epoch+1:3d}/{C.get()['epoch']:3d}] {trace['affinity'] / 'cnt'}")
        ## TargetNetwork Training
        ts = time.time()
        _, total_loader, _, _ = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], 0.0, controller=controller, _transform="default")
        t_net.train()
        t_tracker, metrics = run_epoch(t_net, total_loader, criterion, t_optimizer, desc_default='T-train', epoch=epoch+1, scheduler=t_scheduler, wd=C.get()['optimizer']['decay'], verbose=False, \
                                        trace=True)
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
            if step >= div_loader_len - div_train_len:
                policy = t_dict['policy'][step].cuda()
                top1 = t_dict['acc'][step]
                st = time.time()
                log_probs, entropys, sampled_policies = controller(inputs, policy)
            with torch.no_grad():
                # clean_loss = criterion(t_net(inputs), labels) # clean data loss
                reward = aug_loss #- clean_loss.detach()
                baseline.update(reward.mean())
                if step < div_loader_len - div_train_len: continue
                advantages = reward - baseline.value()
            if mode == "reinforce":
                pol_loss = -1 * (log_probs * advantages).sum() #scalar tensor
            elif mode == 'ppo':
                old_log_probs = t_dict['log_probs'][step].cuda()
                ratios = (log_probs - old_log_probs).exp()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                pol_loss = -torch.min(surr1, surr2).sum()
            pol_loss -= ctl_entropy_w * entropys
            if (step+1)==aff_loader_len:
                length = ctl_num_aggre if aff_train_len % ctl_num_aggre == 0 else aff_train_len % ctl_num_aggre
                pol_loss = pol_loss / length
            else:
                pol_loss = pol_loss / ctl_num_aggre
            pol_loss.backward(retain_graph=ctl_num_aggre>1)
            if (step+1) % ctl_num_aggre == 0 or (step+1)==div_loader_len:
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
        if step >= div_loader_len - div_train_len:
            logger.info(f"(Diversity)[Train Controller {epoch+1:3d}/{C.get()['epoch']:3d}] {trace['diversity'] / 'cnt'}")

        if (epoch+1) % 10 == 0 or epoch == C.get()['epoch']-1:
            # TargetNetwork Test
            t_net.eval()
            test_metric = run_epoch(t_net, test_loader, _criterion, None, desc_default='test T', epoch=epoch+1, verbose=False)
            test_metrics.append(test_metric.get_dict())
            logger.info(f"[Test T {epoch+1:3d}/{C.get()['epoch']:3d}] {test_metric}")
            # update cv_id
            if config['cv_id'] is None:
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
                        'policy': t_dict['policy'],
                        'test_metrics': test_metrics
                        }, target_path)
            torch.save({
                        'epoch': epoch,
                        'ctl_state_dict': controller.state_dict(),
                        'optimizer_state_dict': c_optimizer.state_dict(),
                        'aff_trace': dict(trace['affinity'].trace),
                        'div_trace': dict(trace['diversity'].trace),
                        # 'test_trace': dict(trace['test'].trace)
                        }, ctl_save_path)
        if epoch < C.get()['epoch']-1:
            for k in trace:
                trace[k].reset_accum()
    # C.get()["aug"] = ori_aug
    return trace, test_metrics

def train_controller3(controller, config):
    """
    Weighted Sum of Affinity and Diversity
    0. Training TargetNetwork 1 epoch
    1. Training Controller (weigheted sum of Affinity & Diversity)
    """
    controller = controller.cuda()
    # ori_aug = C.get()["aug"]

    dataset = C.get()['test_dataset']
    target_path = config['target_path']
    ctl_save_path = config['ctl_save_path']
    childnet_paths = config['childnet_paths']
    mode = config['mode']
    load_search = config['load_search']
    childaug = config['childaug']

    eps_clip = 0.2
    ctl_entropy_w = 1e-5
    ctl_ema_weight = 0.95
    cv_id = 0 if config['cv_id'] is None else config['cv_id']
    aff_w = config['aff_w']
    div_w = config['div_w']
    aff_step = config['aff_step']
    div_step = config['div_step']
    reward_type = config["reward_type"]
    controller.train()
    c_optimizer = optim.Adam(controller.parameters(), lr = config['c_lr'])#, weight_decay=1e-6)
    # controller = DataParallel(controller).cuda()

    # load childnet weights
    childnet = get_model(C.get()['model'], num_class(dataset), local_rank=-1).cuda()
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
             'diversity': Tracker()}
             # 'test': Tracker()}
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
        # trace['test'].trace = checkpoint['test_trace']
    else:
        logger.info('------Train Controller from scratch------')
    ### Training Loop
    test_metrics = []
    # policies = []
    total_t_train_time = 0.
    for epoch in range(C.get()['epoch']):
        ## TargetNetwork Training
        ts = time.time()
        _, total_loader, _, test_loader = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], 0.0, controller=controller, _transform="default")
        t_net.train()
        d_tracker, metrics = run_epoch(t_net, total_loader, criterion, t_optimizer, desc_default='T-train', epoch=epoch+1, scheduler=t_scheduler, wd=C.get()['optimizer']['decay'], verbose=False, \
                                        trace=True, get_clean_loss=reward_type==2)
        total_t_train_time += time.time() - ts
        logger.info(f"[T-train] {epoch+1}/{C.get()['epoch']} (time {total_t_train_time:.1f}) {metrics}")
        d_dict = d_tracker.get_dict()
        ## Childnet BackTracking
        _, _, valid_loader, _ = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], config['split_ratio'], split_idx=cv_id, rand_val=True, controller=controller, _transform=childaug)
        a_tracker, _ = run_epoch(childnet, valid_loader, criterion, None, desc_default='childnet tracking', epoch=epoch+1, verbose=False, \
                                 trace=True, get_clean_loss=True)
        a_dict = a_tracker.get_dict()
        # policies.append(d_dict['policy'])
        ## Get Affinity & Diversity Rewards from traces
        with torch.no_grad():
            d_rewards = torch.stack(d_dict['loss']).cuda() # [train_len_d, batch_d]
            a_rewards = torch.stack(a_dict['loss']).cuda() # [train_len_a, batch_a]
            a_clean_loss = torch.stack(a_dict['clean_loss']).cuda()
            a_rewards = a_clean_loss - a_rewards # affinity approximation (usually negative)
            if reward_type > 1:
                if reward_type == 2:
                    d_clean_loss = torch.stack(d_dict['clean_loss']).cuda() # [train_len, batch]
                    d_rewards = d_rewards - d_clean_loss # information gain
                # normalization
                d_rewards = (d_rewards - d_rewards.mean(0)) / (d_rewards.std(0) + 1e-6)
                a_rewards = (a_rewards - a_rewards.mean(0)) / (a_rewards.std(0) + 1e-6)
        # d_loss = 0.
        # Get diversity loss
        controller.train()
        for step, reward in enumerate(d_rewards):
            if div_step is not None and step >= div_step: break
            st = time.time()
            inputs, labels = d_dict['clean_data'][step]
            batch_size = len(labels)
            inputs, labels = inputs.cuda(), labels.cuda()
            policy = d_dict['policy'][step].cuda()
            top1 = d_dict['acc'][step]
            log_probs, d_entropys, _ = controller(inputs, policy)
            if mode == "reinforce":
                pol_loss = -1 * (log_probs * reward)
            elif mode == 'ppo':
                old_log_probs = d_dict['log_probs'][step].cuda()
                ratios = (log_probs - old_log_probs).exp()
                surr1 = ratios * reward
                surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * reward
                pol_loss = -torch.min(surr1, surr2)
            # d_loss += (div_w * pol_loss - ctl_entropy_w * d_entropys).sum()
            d_loss = (div_w * pol_loss - ctl_entropy_w * d_entropys).mean()
            d_loss.backward(retain_graph=True)
            trace['diversity'].add_dict({
                'cnt': batch_size,
                'time': time.time()-st,
                'acc': top1*batch_size,
                'pol_loss': pol_loss.cpu().detach().sum().item(),
                'reward': reward.cpu().detach().sum().item(),
                })
        # d_loss /= sum(trace['diversity']['cnt'])
        logger.info(f"(Diversity){epoch+1:3d}/{C.get()['epoch']:3d} {trace['diversity'] / 'cnt'}")
        # a_loss = 0.
        # Get affinity loss
        for step, reward in enumerate(a_rewards):
            if aff_step is not None and step >= aff_step: break
            st = time.time()
            inputs, labels = a_dict['clean_data'][step]
            batch_size = len(labels)
            inputs, labels = inputs.cuda(), labels.cuda()
            policy = a_dict['policy'][step].cuda()
            top1 = a_dict['acc'][step]
            log_probs, entropys, _ = controller(inputs, policy)
            if mode == "reinforce":
                pol_loss = -1 * (log_probs * reward)
            elif mode == 'ppo':
                old_log_probs = a_dict['log_probs'][step].cuda()
                ratios = (log_probs - old_log_probs).exp()
                surr1 = ratios * reward
                surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * reward
                pol_loss = -torch.min(surr1, surr2)
            # a_loss += (aff_w * pol_loss - ctl_entropy_w * entropys).sum()
            a_loss = (aff_w * pol_loss - ctl_entropy_w * entropys).mean()
            a_loss.backward(retain_graph=True)
            trace['affinity'].add_dict({
                'cnt': batch_size,
                'time': time.time()-st,
                'acc': top1*batch_size,
                'pol_loss': pol_loss.cpu().detach().sum().item(),
                'reward': reward.cpu().detach().sum().item(),
                })
        # a_loss /= sum(trace['affinity']['cnt'])
        logger.info(f"(Affinity) {epoch+1:3d}/{C.get()['epoch']:3d} {trace['affinity'] / 'cnt'}")
        # pol_loss = a_loss + d_loss
        # pol_loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 5.0)
        c_optimizer.step()
        c_optimizer.zero_grad()

        if (epoch+1) % 10 == 0 or epoch == C.get()['epoch']-1:
            # TargetNetwork Test
            t_net.eval()
            test_metric = run_epoch(t_net, test_loader, _criterion, None, desc_default='test T', epoch=epoch+1, verbose=False)
            test_metrics.append(test_metric.get_dict())
            logger.info(f"[Test T {epoch+1:3d}/{C.get()['epoch']:3d}] {test_metric}")
            # update cv_id
            if config['cv_id'] is None:
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
                        'policy': d_dict['policy'],
                        'test_metrics': test_metrics
                        }, target_path)
            torch.save({
                        'epoch': epoch,
                        'ctl_state_dict': controller.state_dict(),
                        'optimizer_state_dict': c_optimizer.state_dict(),
                        'aff_trace': dict(trace['affinity'].trace),
                        'div_trace': dict(trace['diversity'].trace),
                        }, ctl_save_path)
        if epoch < C.get()['epoch']-1:
            for k in trace:
                trace[k].reset_accum()
    # C.get()["aug"] = ori_aug
    return trace, test_metrics

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

class MovingAverage(object):
  """Class that maintains an exponential moving average."""

  def __init__(self, dummy=None):
    self._numerator   = 0
    self._denominator = 0

  def update(self, value):
    self._numerator += value
    self._denominator += 1

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
