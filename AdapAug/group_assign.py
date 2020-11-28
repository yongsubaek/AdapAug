import numpy as np
import copy, os, math, time
from collections import defaultdict
from collections.abc import Iterable
import torch
from torch import nn, optim
from torch.distributions import Categorical
from torchvision.transforms import transforms
from AdapAug.data import get_dataloaders, Augmentation
from AdapAug.train import run_epoch
from AdapAug.networks import get_model, num_class
from warmup_scheduler import GradualWarmupScheduler
from theconf import Config as C
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
_SVHN_MEAN, _SVHN_STD = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
class ModelWrapper(nn.Module):
    def __init__(self, backbone, gr_num, mode="reinforce"):
        super(ModelWrapper, self).__init__()
        backbone = backbone.cuda()
        feature_extracter_list = list(backbone.children())[:-1]
        num_features = feature_extracter_list[-1].num_features # last: bn
        backbone.feature_out = True
        # freeze backbone
        # for name, param in backbone.named_parameters():
        #     if param.requires_grad:
        #         param.requires_grad = False
        self.gr_num = gr_num
        self.num_features = num_features
        self.mode = mode
        self.backbone = backbone
        # self.linear = nn.Linear(num_features+1, gr_num, bias=True)
        self.linear = nn.Sequential(
                            nn.Linear(num_features+1, 128),
                            nn.ReLU(),
                            nn.Linear(128, gr_num)
                            )
        # torch.nn.init.uniform_(self.linear.weight, -1.0, 1.0)

    def forward(self, data, label=None):
        data = data.cuda()
        if label is None:
            label = torch.zeros(len(data), 1)
        feature = self.backbone(data)
        label = label.reshape([-1,1]).float().cuda()
        logits = nn.functional.softmax(self.linear(torch.cat([feature, label], 1)), dim=-1)
        return logits

class GrSpliter(object):
    def __init__(self, childnet, gr_num,
                 ent_w=0.001, eps=1e-3,
                 eps_clip=0.2, mode="ppo",
                 eval_step=100, g_lr=0.00035
                 ):
        self.mode = mode
        # self.childnet = childnet
        self.model = nn.DataParallel(ModelWrapper(copy.deepcopy(childnet), gr_num, mode=self.mode)).cuda()
        if self.mode == "supervised":
            self.g_optimizer = optim.Adam(self.model.parameters(), lr = 5e-4, weight_decay=1e-6)
        else:
            self.g_optimizer = optim.Adam(self.model.parameters(), lr = g_lr, betas=(0.,0.999), eps=0.001, weight_decay=1e-6)
        self.ent_w = ent_w
        self.eps = eps
        self.eps_clip = eps_clip
        self.eval_step = eval_step
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none').cuda()
        self.t_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
        self.transform = None


    def gr_assign(self, dataloader):
        # dataloader: just normaized data
        # TODO: DataParallel
        self.model.eval()
        all_gr_dist = []
        for data, label in dataloader:
            data, label = data.cuda(), label.cuda()
            gr_dist = self.model(data, label)
            all_gr_dist.append(gr_dist.cpu().detach())
        return torch.cat(all_gr_dist)

    def augmentation(self, data, gr_ids, policy):
        aug_imgs = []
        if "cifar" in C.get()["dataset"]:
            mean, std = _CIFAR_MEAN, _CIFAR_STD
        elif "svhn" in C.get()["dataset"]:
            mean, std = _SVHN_MEAN, _SVHN_STD
        # applied_policies = []
        for gr_id, img in zip(gr_ids, data):
            pil_img = transforms.ToPILImage()(UnNormalize()(img.cpu()))
            _aug = Augmentation(policy[int(gr_id)])
            aug_img = _aug(pil_img)
            aug_img = self.transform(aug_img)
            aug_imgs.append(aug_img)
            # applied_policy = _aug.policy # Todo
            # applied_policies.append(applied_policy)
        aug_imgs = torch.stack(aug_imgs)
        return aug_imgs.cuda()#, applied_policies

    def train(self, policy, config):
        # gr: group별 optimal policy가 주어질 때 평균 reward가 가장 높도록 나누는 assigner
        self.model.train()
        gr_num = self.model.module.gr_num
        cv_id = config['cv_id']
        load_path = config["load_path"]
        max_step = config["max_step"]
        childnet = get_model(C.get()['model'], num_class(C.get()['dataset'])).cuda()
        ckpt = torch.load(load_path)
        if 'model' in ckpt:
            childnet.load_state_dict(ckpt['model'])
        else:
            childnet.load_state_dict(ckpt)
        childnet = nn.DataParallel(childnet).cuda()
        childnet.eval()
        pol_losses = []
        ori_aug = C.get()["aug"]
        C.get()["aug"] = "clean"
        _, _, dataloader, _ = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], config['cv_ratio_test'], split_idx=cv_id, rand_val=True)
        loader_iter = iter(dataloader)
        reports = []
        for step in range(max_step):
            try:
                data, label = next(loader_iter)
            except:
                loader_iter = iter(dataloader)
                data, label = next(loader_iter)
            data = data.cuda()
            label = label.cuda()
            logits = self.model(data, label)
            if self.mode=="supervised":
                with torch.no_grad():
                    losses = torch.zeros(gr_num, data.size(0)).cuda()
                    for i in range(gr_num):
                        aug_data = self.augmentation(data, i*torch.ones(data.size(0)), policy)
                        losses[i] = self.loss_fn(childnet(aug_data), label)
                    optimal_gr_ids = losses.min(0)[1]
                loss = self.loss_fn(logits, optimal_gr_ids).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                report_number = loss
            else:
                m = Categorical(logits)
                gr_ids = m.sample()
                log_probs = m.log_prob(gr_ids)
                entropys = m.entropy()
                with torch.no_grad():
                    probs = m.log_prob(torch.tensor([[i] for i in range(gr_num)]).cuda()).exp()
                    rewards_list = torch.zeros(gr_num, data.size(0)).cuda()
                    for i in range(gr_num):
                        aug_data = self.augmentation(data, i*torch.ones_like(gr_ids), policy)
                        rewards_list[i] = 1. / (self.loss_fn(childnet(aug_data), label) + self.eps)
                    rewards = torch.tensor([ rewards_list[gr_id][idx] for idx, gr_id in enumerate(gr_ids)]).cuda().detach()
                    # value function as baseline
                    baselines = sum([ prob*reward for prob, reward in zip(probs, rewards_list) ])
                    advantages = rewards - baselines
                if self.mode=="reinforce":
                    loss = ( -log_probs * advantages ).mean()
                elif self.mode=="ppo":
                    old_log_probs = log_probs.detach()
                    gr_ids = m.sample()
                    log_probs = m.log_prob(gr_ids)
                    ratios = (log_probs - old_log_probs).exp()
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                    loss = -torch.min(surr1, surr2).mean()
                loss -= self.ent_w * entropys.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                report_number = advantages.mean()
            self.g_optimizer.step()
            self.g_optimizer.zero_grad()
            reports.append(float(report_number.cpu().detach()))
            if step % self.eval_step == 0 or step == max_step-1:
                if self.mode == "supervised":
                    entropy = 0.
                else:
                    entropy = (self.ent_w * entropys.mean()).cpu().detach().data
                print(f"[step{step}/{max_step}] objective {np.mean(reports):.4f}, entropy {entropy:.4f}")
        C.get()["aug"] = ori_aug
        return reports

    def dual_train(self, policy, config):
        # train G to maximize TargetNetwork loss
        g_net = self.model
        g_net.train()
        gr_num = g_net.module.gr_num
        cv_id = config['cv_id']
        g_step = config["g_step"]
        len_epoch = config["len_epoch"]
        save_path = config["save_path"]
        # TargetNetwork
        t_net = get_model(C.get()['model'], num_class(C.get()['dataset'])).cuda()
        if os.path.isfile(save_path):
            ckpt = torch.load(save_path)
            t_net.load_state_dict(ckpt['model'])
            start_epoch = ckpt['epoch']
            reports = ckpt['reports']
            policies = ckpt['policies']
        else:
            print("Initial step")
            start_epoch = 1
            reports = []
            policies = []
        policies.append(dict(policy))
        t_net = nn.DataParallel(t_net).cuda()
        t_optimizer, t_scheduler = get_optimizer(t_net)

        ori_aug = C.get()["aug"]
        end_epoch = min(start_epoch + len_epoch, C.get()['epoch']+1)
        total_t_train_time = 0
        total_g_train_time = 0
        for epoch in range(start_epoch, end_epoch):
            # train TargetNetwork
            t_net.train()
            C.get()["aug"] = policy
            ts = time.time()
            _, dataloader, _, _ = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], 0.0, gr_assign=self.gr_assign)
            metrics = run_epoch(t_net, dataloader, self.t_loss_fn, t_optimizer, desc_default='T-train', epoch=epoch, scheduler=t_scheduler, wd=C.get()['optimizer']['decay'], verbose=False)
            total_t_train_time += time.time() - ts
            print(f"[T-train] {epoch}/{end_epoch} (time {total_t_train_time:.1f}) {metrics}")
            # train G
            t_net.eval()
            C.get()["aug"] = "clean"
            gs = time.time()
            _, dataloader, _ , _ = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], 0.0)#, split_idx=cv_id, rand_val=True)
            for step, (data, label) in enumerate(dataloader):
                data, label = data.cuda(), label.cuda()
                # data split
                logits = g_net(data, label)
                if self.mode=="supervised":
                    # get label
                    with torch.no_grad():
                        losses = torch.zeros(gr_num, data.size(0)).cuda()
                        for i in range(gr_num):
                            aug_data = self.augmentation(data, i*torch.ones(data.size(0)), policy)
                            losses[i] = self.loss_fn(t_net(aug_data), label)
                        optimal_gr_ids = losses.max(0)[1]
                    g_loss = self.t_loss_fn(logits, optimal_gr_ids)
                    report_number = g_loss
                else:
                    m = Categorical(logits)
                    gr_ids = m.sample()
                    log_probs = m.log_prob(gr_ids)
                    entropys = m.entropy()
                    # Get Advantage - value function
                    with torch.no_grad():
                        probs = m.log_prob(torch.tensor([[i] for i in range(gr_num)]).cuda()).exp()
                        rewards_list = torch.zeros(gr_num, data.size(0)).cuda()
                        for i in range(gr_num):
                            aug_data = self.augmentation(data, i*torch.ones_like(gr_ids), policy)
                            rewards_list[i] = self.loss_fn(t_net(aug_data), label)
                        rewards = torch.tensor([ rewards_list[gr_id][idx] for idx, gr_id in enumerate(gr_ids)]).cuda().detach()
                        baselines = sum([ prob*reward for prob, reward in zip(probs, rewards_list) ]) # value function
                        advantages = rewards - baselines
                    # G_loss
                    if self.mode=="reinforce":
                        g_loss = ( -log_probs * advantages ).mean()
                    elif self.mode=="ppo":
                        old_log_probs = log_probs.detach()
                        gr_ids = m.sample()
                        log_probs = m.log_prob(gr_ids)
                        ratios = (log_probs - old_log_probs).exp()
                        surr1 = ratios * advantages
                        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                        g_loss = -torch.min(surr1, surr2).mean()
                    g_loss -= self.ent_w * entropys.mean()
                    report_number = advantages.mean()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(g_net.parameters(), 1.0)
                self.g_optimizer.step()
                self.g_optimizer.zero_grad()
                reports.append(float(report_number.cpu().detach()))
                if step % self.eval_step == 0 or step == len(dataloader)-1:
                    if self.mode == "supervised":
                        entropy = 0.
                    else:
                        entropy = (self.ent_w * entropys.mean()).cpu().detach().data
                    print(f"[epoch{epoch}/{end_epoch} {step}] objective {np.mean(reports[-self.eval_step:]):.4f}, entropy {entropy:.4f}")
                    # print(f"Max Advantage: {(rewards_list.max(0)[0] - baselines).mean()}")
            total_g_train_time += time.time() - gs
            print(f"(time {total_g_train_time:.1f})")
        C.get()["aug"] = ori_aug
        torch.save({
            'model': t_net.module.state_dict(),
            'epoch': end_epoch,
            'reports': reports,
            'policies': policies
        }, save_path)
        return reports

def get_optimizer(model):
    # optimizer & scheduler
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=C.get()['lr_schedule']['warmup']['epoch'],
            T_mult=C.get()['lr_schedule']['warmup']['multiplier']
        )
        # scheduler = GradualWarmupScheduler(
        #     optimizer,
        #     multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
        #     total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
        #     after_scheduler=scheduler
        # )
    return optimizer, scheduler

def gen_assign_group(version, num_group=5):
    if version == 1:
        return assign_group
    elif version == 2:
        return lambda data, label=None: assign_group2(data,label,num_group)
    elif version == 3:
        return lambda data, label=None: assign_group3(data,label,num_group)
    elif version == 4:
        return assign_group4
    elif version == 5:
        return assign_group5
def assign_group(data, label=None):
    """
    input: data(batch of images), label(optional, same length with data)
    return: assigned group numbers for data (same length with data)
    to be used before training B.O.
    """
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # exp1: group by manualy classified classes
    groups = {}
    groups[0] = ['plane', 'ship']
    groups[1] = ['car', 'truck']
    groups[2] = ['horse', 'deer']
    groups[3] = ['dog', 'cat']
    groups[4] = ['frog', 'bird']
    def _assign_group_id1(_label):
        gr_id = None
        for key in groups:
            if classes[_label] in groups[key]:
                gr_id = key
                return gr_id
        if gr_id is None:
            raise ValueError(f"label {_label} is not given properly. classes[label] = {classes[_label]}")
    if not isinstance(label, Iterable):
        return _assign_group_id1(label)
    return list(map(_assign_group_id1, label))

def assign_group2(data, label=None, num_group=5):
    """
    input: data(batch of images), label(optional, same length with data)
    return: randomly assigned group numbers for data (same length with data)
    to be used before training B.O.
    """
    _size = len(data)
    return np.random.randint(0, num_group, size=_size)

def assign_group3(data, label=None, num_group=5):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    _classes = list(copy.deepcopy(classes))
    # np.random.shuffle(_classes)
    groups = defaultdict(lambda: [])
    num_cls_per_gr = len(_classes) // num_group + 1
    for i in range(num_group):
        for _ in range(num_cls_per_gr):
            if len(_classes) == 0:
                break
            groups[i].append(_classes.pop())

    def _assign_group_id1(_label):
        gr_id = None
        for key in groups:
            if classes[_label] in groups[key]:
                gr_id = key
                return gr_id
        if gr_id is None:
            raise ValueError(f"label {_label} is not given properly. classes[label] = {classes[_label]}")
    if not isinstance(label, Iterable):
        return _assign_group_id1(label)
    return list(map(_assign_group_id1, label))

def assign_group4(data, label=None):
    return list(map(lambda x: 0 if x<10 else 1, label))

def assign_group5(data, label=None):
    return [0 for _ in label]

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
