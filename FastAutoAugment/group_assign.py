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
    def __init__(self, backbone, gr_num, direct_prob=False):
        super(ModelWrapper, self).__init__()
        backbone = backbone.cuda()
        feature_extracter_list = list(backbone.children())[:-1]
        num_features = feature_extracter_list[-1].num_features # last: bn
        backbone.feature_out = True
        # backbone.linear = nn.Linear(num_features, gr_num)
        # backbone = nn.Sequential(*feature_extracter_list)
        # def hook(model, input, output):
        #     self.out = input[0].detach()
        # backbone.linear.register_forward_hook(hook)

        # freeze backbone
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                param.requires_grad = False
        self.num_features = num_features
        self.direct_prob = direct_prob
        self.backbone = backbone
        self.linear = nn.Linear(num_features+1, gr_num, bias=True)
        # torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, data, label=None):
        data = data.cuda()
        if label is None:
            label = torch.zeros(len(data), 1).cuda()
        else:
            label = label.cuda()
        feature = self.backbone(data)
        logits = nn.functional.softmax(self.linear(torch.cat([feature, label.reshape([-1,1])], 1)), dim=-1)
        gr_id = logits.max(1)[1]
        m = Categorical(logits)
        entropy = m.entropy()
        if self.direct_prob:
            log_prob = logits
        else:
            gr_id = m.sample()
            log_prob = m.log_prob(gr_id)
        return gr_id, log_prob, entropy


class GrSpliter(object):
    def __init__(self, childnet, gr_num):
        self.childnet = childnet
        self.model = ModelWrapper(copy.deepcopy(self.childnet), gr_num).cuda()
        self.optimizer = optim.Adam(self.model.linear.parameters(), lr = 0.00035, betas=(0.,0.999), eps=0.001)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none').cuda()
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        if C.get()['cutout'] > 0:
            self.transform.transforms.append(CutoutDefault(C.get()['cutout']))

    def gr_assign(self, dataloader):
        # dataloader: just normaized data
        self.model.eval()
        all_gr_ids = []
        for data, label in dataloader:
            data, label = data.cuda(), label.cuda()
            gr_ids = self.model(data, label)[0]
            all_gr_ids.append(gr_ids.cpu().detach())
        return torch.cat(all_gr_ids).numpy()

    def augmentation(self, data, gr_ids, policy):
        aug_imgs = []
        # applied_policies = []
        for gr_id, img in zip(gr_ids, data):
            pil_img = transforms.ToPILImage()(img.cpu())
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
        # linear part만 학습
        # torch.nn.DataParallel
        self.model.train()
        cv_id = config['cv_id']
        load_path = config["load_path"]
        rep = config["rep"]
        childnet = get_model(C.get()['model'], num_class(C.get()['dataset']))
        ckpt = torch.load(load_path)
        if 'model' in ckpt:
            childnet.load_state_dict(ckpt['model'])
        else:
            childnet.load_state_dict(ckpt)
        childnet.eval()
        baseline  = ExponentialMovingAverage(0.9)
        pol_losses = []
        ori_aug = C.get()["aug"]
        C.get()["aug"] = "nonorm"
        loaders = []
        for _ in range(rep):
            _, _, dataloader, _ = get_dataloaders(C.get()['dataset'], C.get()['batch'], config['dataroot'], config['cv_ratio_test'], split_idx=cv_id, rand_val=True)
            loaders.append(dataloader)
        for loader in loaders:
            for data, label in loader:
                data = data.cuda()
                label = label.cuda()
                gr_ids, log_probs, entropys = self.model(data, label)
                with torch.no_grad():
                    aug_data = self.augmentation(data, gr_ids, policy)
                    rewards = -self.loss_fn(childnet(aug_data), label)
                baseline.update(rewards.mean())
                policy_loss = ( -log_probs * (rewards - baseline.value()) ).sum()
                policy_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                pol_losses.append(float(policy_loss.cpu().detach()))
        C.get()["aug"] = ori_aug
        return pol_losses


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
