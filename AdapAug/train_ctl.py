import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import itertools
import json
import logging
import math, time
import os, copy
from collections import OrderedDict, defaultdict

import torch
from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torchvision import transforms

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser

from AdapAug.common import get_logger, EMA, add_filehandler, get_optimizer
from AdapAug.data import get_dataloaders, Augmentation
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
    ctl_entropy_w = config['ctl_entropy_w']
    ctl_ema_weight = 0.95

    controller.train()
    # c_optimizer = optim.Adam(controller.parameters(), lr = config['c_lr'])#, weight_decay=1e-6)
    c_optimizer = optim.SGD(controller.parameters(),
                            lr=config['c_lr'],
                            momentum=C.get()['optimizer'].get('momentum', 0.9),
                            weight_decay=0.0,
                            nesterov=C.get()['optimizer'].get('nesterov', True)
                            )
    c_scheduler = GradualWarmupScheduler(
        c_optimizer,
        multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
        total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
        after_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(c_optimizer, T_max=C.get()['epoch'], eta_min=0.)
    )
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
        t_optimizer.load_state_dict(data['optimizer_state_dict'])
        start_epoch = data['epoch']
        policies = data['policy']
        test_metrics = data['test_metrics']
        del data
    else:
        start_epoch = 0
        policies = []
        test_metrics = []
    # load ctl weights and results
    if load_search and os.path.isfile(ctl_save_path):
        logger.info('------Controller load------')
        checkpoint = torch.load(ctl_save_path)
        controller.load_state_dict(checkpoint['ctl_state_dict'])
        c_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trace['diversity'].trace = checkpoint['div_trace']
        train_metrics = checkpoint['train_metrics']
        del checkpoint
    else:
        logger.info('------Train Controller from scratch------')
        train_metrics = {"affinity":[], "diversity": []}
    ### Training Loop
    baseline = ZeroBase(ctl_ema_weight)
    # baseline = ExponentialMovingAverage(ctl_ema_weight)
    total_t_train_time = 0.
    for epoch in range(start_epoch, C.get()['epoch']):
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
        policies.append(sampled_policies)
        _, total_loader, _, test_loader = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], 0.0, _transform=sampled_policies, batch_multiplier=batch_multiplier)
        t_net.train()
        # training and return M normalized moving averages of losses
        metrics = run_epoch(t_net, total_loader, criterion if batch_multiplier>1 else _criterion, t_optimizer, desc_default='T-train', epoch=epoch+1, scheduler=t_scheduler, wd=C.get()['optimizer']['decay'], verbose=False, \
                            batch_multiplier=batch_multiplier)
        if batch_multiplier > 1:
            tracker, metrics = metrics
            track = tracker.get_dict()
        train_metrics['diversity'].append(metrics.get_dict())
        total_t_train_time += time.time() - ts
        logger.info(f"[T-train] {epoch+1}/{C.get()['epoch']} (time {total_t_train_time:.1f}) {metrics}")
        ## Diversity Training from TargetNetwork trace
        st = time.time()
        controller.train()
        with torch.no_grad():
            if batch_multiplier > 1:
                rewards = torch.stack(track['loss']).mean(0).reshape(batch_multiplier, -1).mean(1).cuda() # [M]
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6) # [M]
                rewards = rewards.mean().item()
            else:
                rewards = metrics['loss']
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
        c_scheduler.step(epoch)
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
                        'policy': policies,
                        'test_metrics': test_metrics,
                        }, target_path)
            torch.save({
                        'epoch': epoch,
                        'ctl_state_dict': controller.state_dict(),
                        'optimizer_state_dict': c_optimizer.state_dict(),
                        'div_trace': dict(trace['diversity'].trace),
                        'train_metrics': train_metrics,
                        }, ctl_save_path)
    # C.get()["aug"] = ori_aug
    train_metrics['affinity'] = [{'top1': 0.}]
    return trace, train_metrics, test_metrics

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

    train_metrics = {"affinity":[], "diversity": []}
    test_metrics = []
    total_t_train_time = 0.
    for epoch in range(C.get()['epoch']):
        ## Affinity Training
        baseline = ExponentialMovingAverage(ctl_ema_weight)
        repeat = 1#len(total_loader.dataset)//len(valid_loader.dataset) if aff_step is None else 1
        for _ in range(repeat):
            _, _, valid_loader, _ = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], config['split_ratio'], split_idx=cv_id, rand_val=True, controller=controller, _transform=childaug)
            a_tracker, a_metrics = run_epoch(childnet, valid_loader, criterion, None, desc_default='childnet tracking', epoch=epoch+1, verbose=False, \
                                     trace=True)
            train_metrics["affinity"].append(a_metrics.get_dict())
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
        t_tracker, d_metrics = run_epoch(t_net, total_loader, criterion, t_optimizer, desc_default='T-train', epoch=epoch+1, scheduler=t_scheduler, wd=C.get()['optimizer']['decay'], verbose=False, \
                                        trace=True)
        total_t_train_time += time.time() - ts
        logger.info(f"[T-train] {epoch+1}/{C.get()['epoch']} (time {total_t_train_time:.1f}) {d_metrics}")
        train_metrics["diversity"].append(d_metrics.get_dict())
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
                        'train_metrics': train_metrics,
                        }, ctl_save_path)
        if epoch < C.get()['epoch']-1:
            for k in trace:
                trace[k].reset_accum()
    # C.get()["aug"] = ori_aug
    return trace, train_metrics, test_metrics

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
    ctl_entropy_w = config['ctl_entropy_w']
    ctl_ema_weight = 0.95
    cv_id = 0 if config['cv_id'] is None else config['cv_id']
    aff_w = config['aff_w']
    div_w = config['div_w']
    aff_step = config['aff_step']
    div_step = config['div_step']
    reward_type = config["reward_type"] # 0. ema, 1: none, 2: diversity=info_gain + batch norm, 3. batch norm
    batch_multiplier = config['M']

    controller.train()
    if controller.img_input:
        c_optimizer = optim.Adam([
                                    {'params':controller.conv_input.parameters(), 'lr': 0.01, 'weight_decay': 5e-4},
                                    {'params':controller.rnn_params()},
                                 ], lr = config['c_lr'])#, weight_decay=1e-6)
        # c_optimizer = optim.SGD([
        #                         {'params':controller.conv_input.parameters(), 'lr': C.get()['lr'], 'weight_decay': 5e-4},
        #                         {'params':controller.rnn_params()},
        #                         ],
        #                         lr=config['c_lr'],
        #                         momentum=C.get()['optimizer'].get('momentum', 0.9),
        #                         weight_decay=0.0,
        #                         nesterov=C.get()['optimizer'].get('nesterov', True)
        #                         )
    else:
        # c_optimizer = optim.Adam(controller.parameters(), lr = config['c_lr'])#, weight_decay=1e-6)
        c_optimizer = optim.SGD(controller.parameters(),
                                lr=config['c_lr'],
                                momentum=C.get()['optimizer'].get('momentum', 0.9),
                                weight_decay=0.0,
                                nesterov=C.get()['optimizer'].get('nesterov', True)
                                )
    c_scheduler = GradualWarmupScheduler(
        c_optimizer,
        multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
        total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
        after_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(c_optimizer, T_max=C.get()['epoch'], eta_min=0.)
    )
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
    childnet.eval()

    # create a TargetNetwork
    t_net = get_model(C.get()['model'], num_class(dataset), local_rank=-1).cuda()
    t_optimizer, t_scheduler = get_optimizer(t_net)
    criterion = CrossEntropyLabelSmooth(num_class(dataset), C.get().conf.get('lb_smooth', 0), reduction="batched_sum").cuda()
    _criterion = CrossEntropyLabelSmooth(num_class(dataset), C.get().conf.get('lb_smooth', 0)).cuda()
    if batch_multiplier > 1:
        t_net = DataParallel(t_net).cuda()
        if controller.img_input:
            controller = DataParallel(controller).cuda()
        if aff_w != 0.:
            childnet = DataParallel(childnet).cuda()
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
        t_optimizer.load_state_dict(data['optimizer_state_dict'])
        start_epoch = data['epoch']
        policies = data['policy']
        test_metrics = data['test_metrics']
        del data
    else:
        start_epoch = 0
        policies = []
        test_metrics = []
    # load ctl weights and results
    if load_search and os.path.isfile(ctl_save_path):
        logger.info('------Controller load------')
        checkpoint = torch.load(ctl_save_path)
        controller.load_state_dict(checkpoint['ctl_state_dict'])
        c_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trace['affinity'].trace = checkpoint['aff_trace']
        trace['diversity'].trace = checkpoint['div_trace']
        train_metrics = checkpoint['train_metrics']
        del checkpoint
    else:
        logger.info('------Train Controller from scratch------')
        train_metrics = {"affinity":[], "diversity": []}
    ### Training Loop
    total_t_train_time = 0.
    for epoch in range(start_epoch, C.get()['epoch']):
        ## TargetNetwork Training
        ts = time.time()
        _, total_loader, valid_loader, test_loader = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], config['split_ratio'], split_idx=cv_id, \
                                                     rand_val=True, controller=controller, _transform="default", validation=config['validation'], batch_multiplier=batch_multiplier)
        t_net.train()
        d_tracker, d_metrics = run_epoch(t_net, total_loader, criterion, t_optimizer, desc_default='T-train', epoch=epoch+1, scheduler=t_scheduler, wd=C.get()['optimizer']['decay'], verbose=False, \
                                        trace=True, get_trace=['clean_loss'] if reward_type==2 else [], batch_multiplier=batch_multiplier)
        total_t_train_time += time.time() - ts
        logger.info(f"[T-train] {epoch+1}/{C.get()['epoch']} (time {total_t_train_time:.1f}) {d_metrics}")
        train_metrics["diversity"].append(d_metrics.get_dict())
        d_dict = d_tracker.get_dict()
        del d_tracker, d_metrics
        policies.append(d_dict['policy'])
        with torch.no_grad():
            d_rewards = torch.stack(d_dict['loss']).cuda() # [train_len_d, M*batch]
            _d_rewards = d_rewards.cpu().detach()
            if reward_type > 1:
                if reward_type == 2:
                    d_clean_loss = torch.stack(d_dict['clean_loss']).cuda() # [train_len, M*batch]
                    d_rewards = d_rewards - d_clean_loss.repeat(1,batch_multiplier) # information gain
                # normalization
                d_rewards = (d_rewards - d_rewards.mean(1).reshape(-1,1).expand(d_rewards.size(0), d_rewards.size(1))) / (d_rewards.std(1).reshape(-1,1).expand(d_rewards.size(0), d_rewards.size(1)) + 1e-6)
            else:
                d_baseline = ExponentialMovingAverage(ctl_ema_weight)
        ## Childnet BackTracking
        if aff_w != 0.:
            # _, _, valid_loader, _ = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], config['split_ratio'], split_idx=cv_id, \
            #                                         rand_val=True, controller=controller, _transform=childaug, validation=config['validation'])
            a_tracker, a_metrics = run_epoch(childnet, valid_loader, criterion, None, desc_default='childnet tracking', epoch=epoch+1, verbose=False, \
                                            trace=True, get_trace=['logits', 'clean_logits'] if reward_type in [0,1,4] else ['clean_loss'], batch_multiplier=batch_multiplier)
            train_metrics["affinity"].append(a_metrics.get_dict())
            a_dict = a_tracker.get_dict()
            del a_tracker, a_metrics
            ## Get Affinity & Diversity Rewards from traces
            with torch.no_grad():
                if reward_type in [2,3]:
                    a_rewards = torch.stack(a_dict['loss']).cuda() # [train_len_a, M*batch]
                    a_clean_loss = torch.stack(a_dict['clean_loss']).cuda() # [train_len_a, batch]
                    a_rewards = a_clean_loss.repeat(1,batch_multiplier) - a_rewards # affinity approximation (usually negative)
                else: # reward_type in [0,1,4]
                    a_rewards = torch.stack(a_dict['logits']).max(-1)[1].cuda() # [train_len_a, M*batch]
                    a_clean_logits = torch.stack(a_dict['clean_logits']).max(-1)[1].cuda() # [train_len_a, batch]
                    a_rewards = (a_clean_logits.repeat(1,batch_multiplier) == a_rewards).float() # [train_len_a, M*batch]
                _a_rewards = a_rewards.cpu().detach()
                if reward_type > 1:
                    # normalization
                    a_rewards = (a_rewards - a_rewards.mean(1).reshape(-1,1).expand(a_rewards.size(0), a_rewards.size(1))) / (a_rewards.std(1).reshape(-1,1).expand(a_rewards.size(0), a_rewards.size(1)) + 1e-6)
                else:
                    a_baseline = ExponentialMovingAverage(ctl_ema_weight)

        # Get diversity loss
        controller.train()
        if div_w != 0.:
            for step, reward in enumerate(d_rewards):
                if div_step is not None and step >= div_step: break
                st = time.time()
                inputs, labels = d_dict['clean_data'][step]
                batch_size = len(labels)*batch_multiplier
                inputs, labels = inputs.cuda(), labels.cuda()
                policy = d_dict['policy'][step].cuda() # [batch*M, n_subpolicy, n_op, 3]
                top1 = d_dict['acc'][step]
                log_probs, entropys, _ = controller(inputs.repeat(batch_multiplier,1,1,1), policy) # [batch*M]
                if reward_type == 0:
                    d_baseline.update(reward.mean())
                    advantages = reward - d_baseline.value()
                else:
                    advantages = reward
                if mode == "reinforce":
                    pol_loss = -1 * (log_probs * advantages)
                elif mode == 'ppo':
                    old_log_probs = d_dict['log_probs'][step].cuda() # [batch*M]
                    ratios = (log_probs - old_log_probs).exp()
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                    pol_loss = -torch.min(surr1, surr2)
                d_loss = (div_w * pol_loss - ctl_entropy_w * entropys).sum()
                d_loss.backward(retain_graph=True)
                trace['diversity'].add_dict({
                    'cnt': batch_size,
                    'time': time.time()-st,
                    'acc': top1*batch_size,
                    'pol_loss': pol_loss.cpu().detach().sum().item(),
                    'reward': _d_rewards[step].sum().item(),
                    })
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 5.0)
            c_optimizer.step()
            c_optimizer.zero_grad()
            logger.info(f"(Diversity){epoch+1:3d}/{C.get()['epoch']:3d} {trace['diversity'] / 'cnt'}")
        # Get affinity loss
        if aff_w != 0.:
            for step, reward in enumerate(a_rewards):
                if aff_step is not None and step >= aff_step: break
                st = time.time()
                inputs, labels = a_dict['clean_data'][step]
                batch_size = len(labels)*batch_multiplier
                inputs, labels = inputs.cuda(), labels.cuda()
                policy = a_dict['policy'][step].cuda()
                top1 = a_dict['acc'][step]
                log_probs, entropys, _ = controller(inputs.repeat(batch_multiplier,1,1,1), policy)
                if reward_type == 0:
                    a_baseline.update(reward.mean())
                    advantages = reward - a_baseline.value()
                else:
                    advantages = reward
                if mode == "reinforce":
                    pol_loss = -1 * (log_probs * advantages)
                elif mode == 'ppo':
                    old_log_probs = a_dict['log_probs'][step].cuda()
                    ratios = (log_probs - old_log_probs).exp()
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                    pol_loss = -torch.min(surr1, surr2)
                # a_loss += (aff_w * pol_loss - ctl_entropy_w * entropys).sum()
                a_loss = (len(total_loader)/len(valid_loader))*(aff_w*(1-top1)*pol_loss - ctl_entropy_w * entropys).sum()
                a_loss.backward(retain_graph=True)
                trace['affinity'].add_dict({
                    'cnt': batch_size,
                    'time': time.time()-st,
                    'acc': top1*batch_size,
                    'pol_loss': pol_loss.cpu().detach().sum().item(),
                    'reward': _a_rewards[step].sum().item(),
                    })
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 5.0)
            c_optimizer.step()
            c_optimizer.zero_grad()
            logger.info(f"(Affinity) {epoch+1:3d}/{C.get()['epoch']:3d} {trace['affinity'] / 'cnt'}")
        c_scheduler.step(epoch)

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
                        'policy': policies,
                        'test_metrics': test_metrics,
                        }, target_path)
            torch.save({
                        'epoch': epoch,
                        'ctl_state_dict': controller.state_dict(),
                        'optimizer_state_dict': c_optimizer.state_dict(),
                        'aff_trace': dict(trace['affinity'].trace),
                        'div_trace': dict(trace['diversity'].trace),
                        'train_metrics': train_metrics,
                        }, ctl_save_path)
        if epoch < C.get()['epoch']-1:
            for k in trace:
                trace[k].reset_accum()
    # C.get()["aug"] = ori_aug
    if len(train_metrics["affinity"])==0:
        train_metrics["affinity"].append(defaultdict(lambda: 0.))
    return trace, train_metrics, test_metrics

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

class ZeroBase(object):
  """Dummy class for non-baseline."""

  def __init__(self, dummy=None):
      pass
  def update(self, dummy=None):
      pass
  def value(self):
    return 0.
