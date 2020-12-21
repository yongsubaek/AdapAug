import copy
import os
import sys
import time
from collections import OrderedDict, defaultdict, Counter

import torch
from torch.distributions import Categorical
import numpy as np
from hyperopt import hp
import ray
from tqdm import tqdm

from pathlib import Path
lib_dir = (Path("__file__").parent).resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from AdapAug.augmentations import augment_list
from AdapAug.common import get_logger, add_filehandler
from AdapAug.data import get_dataloaders
from AdapAug.metrics import Accumulator, accuracy
from AdapAug.networks import get_model, num_class
from AdapAug.train import train_and_eval
from theconf import Config as C, ConfigArgumentParser
from AdapAug.controller import Controller
from AdapAug.train_ctl import train_controller, train_controller2, train_controller3
import csv, random
import warnings
warnings.filterwarnings("ignore")


def save_res(iter, acc, best, term):
    base_path = f"models/{C.get()['exp_name']}"
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), base_path)
    os.makedirs(base_path, exist_ok=True)
    f = open(os.path.join(base_path, "iter_acc.csv"), "a", newline="")
    wr = csv.writer(f)
    wr.writerow([iter, acc, best, term])
    f.close()

logger = get_logger('Adap AutoAugment')

def gen_rand_policy(num_policy, num_op):
    op_list = augment_list(False)
    policies = []
    for i in range(num_policy):
        ops = []
        for j in range(num_op):
            op_idx = random.randint(0, len(op_list)-1)
            op_prob = random.random()
            op_level = random.random()
            ops.append((op_list[op_idx][0].__name__, op_prob, op_level))
        policies.append(ops)
    return policies

def get_affinity(aug, aff_bases, config, augment):
    C.get()
    C.get().conf = config
    # setup - provided augmentation rules
    C.get()['aug'] = aug
    load_paths = augment['load_paths']
    cv_num = augment["cv_num"]

    aug_loaders = []
    for cv_id in range(cv_num):
        _, tl, validloader, tl2 = get_dataloaders(C.get()['dataset'], C.get()['batch'], augment['dataroot'], augment['cv_ratio_test'], split_idx=cv_id, gr_ids=augment["gr_ids"])
        aug_loaders.append(validloader)
        del tl, tl2

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    aug_accs = []
    for cv_id, loader in enumerate(aug_loaders):
        # eval
        model = get_model(C.get()['model'], num_class(C.get()['dataset']))
        ckpt = torch.load(load_paths[cv_id])
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
        model.eval()

        metrics = Accumulator()
        for data, label in loader:
            data = data.cuda()
            label = label.cuda()

            pred = model(data)
            loss = loss_fn(pred, label) # (N)

            _, pred = pred.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(label.view(1, -1).expand_as(pred)).detach().cpu().numpy() # (1,N)

            # top1 = accuracy(pred, label, (1, 5))[0].detach().cpu().numpy()
            # correct = top1 * len(data)
            metrics.add_dict({
                'loss': np.sum(loss.detach().cpu().numpy()),
                'correct': np.sum(correct),
                'cnt': len(data)
            })
            del loss, correct, pred, data, label
        aug_accs.append(metrics['correct'] / metrics['cnt'])
    del model
    affs = []
    for aug_valid, clean_valid in zip(aug_accs, aff_bases):
        affs.append(aug_valid - clean_valid)
    return affs

def _get_path(dataset, model, tag, basemodel=True):
    base_path = "models" if basemodel else f"models/{C.get()['exp_name']}"
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), base_path)
    os.makedirs(base_path, exist_ok=True)
    return os.path.join(base_path, '%s_%s_%s.model' % (dataset, model, tag))     # TODO

@ray.remote(num_gpus=0.5, max_calls=1)
def train_model(config, dataloaders, dataroot, augment, cv_ratio_test, cv_id, save_path=None, skip_exist=False, evaluation_interval=5, gr_assign=None, gr_dist=None):
    C.get()
    C.get().conf = config
    C.get()['aug'] = augment
    result = train_and_eval(None, dataloaders, dataroot, cv_ratio_test, cv_id, save_path=save_path, only_eval=skip_exist, evaluation_interval=evaluation_interval, gr_assign=gr_assign, gr_dist=gr_dist)
    return C.get()['model']['type'], cv_id, result

if __name__ == '__main__':
    import json
    from pystopwatch2 import PyStopwatch
    w = PyStopwatch()

    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--dataroot', type=str, default='/mnt/hdd0/data/', help='torchvision data folder')
    parser.add_argument('--until', type=int, default=5)
    parser.add_argument('--num-op', type=int, default=2)
    parser.add_argument('--num-policy', type=int, default=5)
    parser.add_argument('--cv-num', type=int, default=5)
    parser.add_argument('--cv-ratio', type=float, default=0.4)
    parser.add_argument('--decay', type=float, default=-1)
    parser.add_argument('--redis', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--childaug', type=str, default="default")
    parser.add_argument('--mode', type=str, default="ppo")
    parser.add_argument('--load_search', action='store_true')
    parser.add_argument('--rand_search', action='store_true')
    parser.add_argument('--exp_name', type=str)

    parser.add_argument('--lstm_n', type=int, default=1)
    parser.add_argument('--lstm_size', type=int, default=100)
    parser.add_argument('--emb_size', type=int, default=32)
    parser.add_argument('--temp', type=float)
    parser.add_argument('--c_lr', type=float, default=0.00035)
    parser.add_argument('--cv_id', type=int)
    parser.add_argument('--c_step', type=int)
    parser.add_argument('--a_step', type=int)
    parser.add_argument('--d_step', type=int)
    parser.add_argument('--c_agg', type=int, default=1)
    parser.add_argument('--aw', type=float, default=0.)
    parser.add_argument('--dw', type=float, default=1.)
    parser.add_argument('--ew', type=float, default=1e-5)
    parser.add_argument('--M', type=int, default=1)
    parser.add_argument('--no_img', action='store_true')
    parser.add_argument('--r_type', type=int, default=1)
    parser.add_argument('--validation', action='store_true')


    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    if args.decay > 0:
        logger.info('decay=%.4f' % args.decay)
        C.get()['optimizer']['decay'] = args.decay
    C.get()['exp_name'] = args.exp_name
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', C.get()['exp_name'])
    os.makedirs(base_path, exist_ok=True)
    add_filehandler(logger, os.path.join(base_path, '%s_%s_cv%.1f.log' % (C.get()['dataset'], C.get()['model']['type'], args.cv_ratio)))
    logger.info('configuration...')
    logger.info(json.dumps(C.get().conf, sort_keys=True, indent=4))
    cv_num = args.cv_num
    C.get()["cv_num"] = cv_num
    bench_policy_set = C.get()["aug"]
    if 'test_dataset' not in C.get().conf:
        C.get()['test_dataset'] = C.get()['dataset']
    copied_c = copy.deepcopy(C.get().conf)
    paths = [_get_path(C.get()['dataset'], C.get()['model']['type'], '%s_ratio%.1f_fold%d' % (args.childaug, args.cv_ratio, i)) for i in range(cv_num)]
    logger.info('initialize ray...')
    ray.init(address=args.redis)
    logger.info('search augmentation policies, dataset=%s model=%s' % (C.get()['dataset'], C.get()['model']['type']))
    logger.info('----- Train without Augmentations cv=%d ratio(test)=%.1f -----' % (cv_num, args.cv_ratio))
    w.start(tag='train_no_aug')
    print(paths)
    reqs = [
        train_model.remote(copy.deepcopy(copied_c), None, args.dataroot, args.childaug, args.cv_ratio, i, save_path=paths[i], evaluation_interval=50)
        for i in range(cv_num)]

    tqdm_epoch = tqdm(range(C.get()['epoch']))
    is_done = False
    for epoch in tqdm_epoch:
        while True:
            epochs_per_cv = OrderedDict()
            for cv_idx in range(cv_num):
                try:
                    latest_ckpt = torch.load(paths[cv_idx])
                    if 'epoch' not in latest_ckpt:
                        epochs_per_cv['cv%d' % (cv_idx + 1)] = C.get()['epoch']
                        continue
                    epochs_per_cv['cv%d' % (cv_idx+1)] = latest_ckpt['epoch']
                except Exception as e:
                    continue
            tqdm_epoch.set_postfix(epochs_per_cv)
            if len(epochs_per_cv) == cv_num and min(epochs_per_cv.values()) >= C.get()['epoch']:
                is_done = True
            if len(epochs_per_cv) == cv_num and min(epochs_per_cv.values()) >= epoch:
                break
            time.sleep(10)
        if is_done:
            break
    logger.info('getting results...')
    pretrain_results = ray.get(reqs)
    aff_bases = []
    for r_model, r_cv, r_dict in pretrain_results:
        logger.info('model=%s cv=%d top1_train=%.4f top1_valid=%.4f' % (r_model, r_cv+1, r_dict['top1_train'], r_dict['top1_valid']))
        # for Affinity calculation
        aff_bases.append(r_dict['top1_valid'])
        del r_model, r_cv, r_dict
    logger.info('processed in %.4f secs' % w.pause('train_no_aug'))
    if args.until == 1:
        sys.exit(0)
    del latest_ckpt, pretrain_results, reqs
    ray.shutdown()
    logger.info('----- Search Test-Time Augmentation Policies -----')
    w.start(tag='search-g_train')
    ops = augment_list(False)
    target_path = base_path + "/target_network.pt"
    ctl_save_path = base_path + "/ctl_network.pt"
    controller = Controller(n_subpolicy=args.num_policy, lstm_size=args.lstm_size, emb_size=args.emb_size, lstm_num_layers = args.lstm_n,
                            operation_prob=0, img_input=not args.no_img, temperature=args.temp).cuda()
    ctl_config = {
            'dataroot': args.dataroot, 'split_ratio': args.cv_ratio, 'load_search': args.load_search,
            'target_path': target_path, 'ctl_save_path': ctl_save_path, 'childnet_paths': paths,
            'childaug': args.childaug, 'mode': args.mode, 'c_lr': args.c_lr,
            'cv_num': cv_num, 'cv_id': args.cv_id,
            'ctl_train_steps': args.c_step, 'aff_step': args.a_step, 'div_step': args.d_step, # version 2
            'aff_w': args.aw, 'div_w': args.dw, 'ctl_entropy_w': args.ew, 'reward_type': args.r_type, # version 3
            'ctl_num_aggre': args.c_agg, "M": args.M, 'validation': args.validation,
    }
    if args.version == 2:
        # epoch-wise alternating training
        train_ctl = train_controller2
    elif args.version == 3:
        # training by weighted sum
        train_ctl = train_controller3
    else:
        # Adversarial AutoAugment
        train_ctl = train_controller

    trace, train_metrics, test_metrics = train_ctl(controller, ctl_config)
    aff_metrics = train_metrics["affinity"][-1]
    div_metrics = train_metrics["diversity"][-1]
    metrics = test_metrics[-1]
    # test t_net
    logger.info('getting results...')
    logger.info(f'train_acc={div_metrics["top1"]:.4f}' \
              + f' affinity={aff_metrics["top1"]:.4f} diversity={div_metrics["loss"]:.4f}' \
              + f' test_loss={metrics["loss"]:.4f} test_acc={metrics["top1"]:.4f}')
    logger.info(w)
