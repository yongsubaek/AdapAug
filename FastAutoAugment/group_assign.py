import numpy as np
import copy
from collections import defaultdict
from collections.abc import Iterable
import torch
from torch import nn, optim
from torch.distributions import Categorical
from torchvision.transforms import transforms
from FastAutoAugment.data import get_dataloaders, CutoutDefault, Augmentation
from FastAutoAugment.networks import get_model, num_class
from theconf import Config as C
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
class ModelWrapper(nn.Module):
    def __init__(self, backbone, gr_num, mode="reinforce"):
        super(ModelWrapper, self).__init__()
        backbone = backbone.cuda()
        feature_extracter_list = list(backbone.children())[:-1]
        num_features = feature_extracter_list[-1].num_features # last: bn
        backbone.feature_out = True
        # freeze backbone
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                param.requires_grad = False
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
                 ent_w=0.00001, eps=1e-3,
                 eps_clip=0.2, mode="ppo",
                 eval_step=20
                 ):
        self.mode = mode
        self.childnet = childnet
        self.model = ModelWrapper(copy.deepcopy(self.childnet), gr_num, mode=self.mode).cuda()
        if self.mode == "supervised":
            self.optimizer = optim.Adam(self.model.parameters(), lr = 5e-4, betas=(0.,0.999), eps=0.001, weight_decay=1e-4)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr = 5e-5, weight_decay=1e-4)
        self.ent_w = ent_w
        self.eps = eps
        self.eps_clip = eps_clip
        self.eval_step = eval_step
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none').cuda()
        self.transform = None
        # self.transform = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        # ])
        # if C.get()['cutout'] > 0 and C.get()['aug'] != "nocut":
        #     self.transform.transforms.append(CutoutDefault(C.get()['cutout']))

    def gr_assign(self, dataloader):
        # dataloader: just normaized data
        self.model.eval()
        all_gr_dist = []
        for data, label in dataloader:
            data, label = data.cuda(), label.cuda()
            gr_dist = self.model(data, label)
            all_gr_dist.append(gr_dist.cpu().detach())
        return torch.cat(all_gr_dist)

    def augmentation(self, data, gr_ids, policy):
        aug_imgs = []
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
        cv_id = config['cv_id']
        load_path = config["load_path"]
        max_step = config["max_step"]
        childnet = get_model(C.get()['model'], num_class(C.get()['dataset'])).cuda()
        ckpt = torch.load(load_path)
        if 'model' in ckpt:
            childnet.load_state_dict(ckpt['model'])
        else:
            childnet.load_state_dict(ckpt)
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
                    losses = torch.zeros(self.model.gr_num, data.size(0)).cuda()
                    for i in range(self.model.gr_num):
                        aug_data = self.augmentation(data, i*torch.ones(data.size(0)), policy)
                        losses[i] = self.loss_fn(childnet(aug_data), label)
                    optimal_gr_ids = losses.min(0)[1]
                loss = self.loss_fn(logits, optimal_gr_ids).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.linear.parameters(), 5.0)
                report_number = loss
            else:
                m = Categorical(logits)
                gr_ids = m.sample()
                log_probs = m.log_prob(gr_ids)
                entropys = m.entropy()
                with torch.no_grad():
                    probs = m.log_prob(torch.tensor([[i] for i in range(self.model.gr_num)]).cuda()).exp()
                    rewards_list = torch.zeros(self.model.gr_num, data.size(0)).cuda()
                    for i in range(self.model.gr_num):
                        aug_data = self.augmentation(data, i*torch.ones_like(gr_ids), policy)
                        rewards_list[i] = 1. / (self.loss_fn(childnet(aug_data), label) + self.eps) + self.ent_w*entropys
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
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.linear.parameters(), 5.0)
                report_number = advantages.mean()
            self.optimizer.step()
            self.optimizer.zero_grad()
            reports.append(float(report_number.cpu().detach()))
            if step % self.eval_step == 0 or step == max_step-1:
                print(f"[step{step}/{max_step}] objective {report_number:.4f}")

        C.get()["aug"] = ori_aug
        return reports


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
