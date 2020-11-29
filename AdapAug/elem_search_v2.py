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
import gorilla
from ray import tune
from ray.tune.trial import Trial
from ray.tune.trial_runner import TrialRunner
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune import register_trainable, run_experiments, run, Experiment
from tqdm import tqdm

from pathlib import Path
lib_dir = (Path("__file__").parent).resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from AdapAug.archive import remove_deplicates, policy_decoder, fa_reduced_svhn, fa_reduced_cifar10
from AdapAug.augmentations import augment_list
from AdapAug.common import get_logger, add_filehandler
from AdapAug.data import get_dataloaders, get_gr_dist, get_post_dataloader
from AdapAug.metrics import Accumulator, accuracy
from AdapAug.networks import get_model, num_class
from AdapAug.train import train_and_eval
from theconf import Config as C, ConfigArgumentParser
from AdapAug.controller import Controller
from AdapAug.train_ctl import train_controller
import csv, random

top1_valid_by_cv = defaultdict(lambda: list)

def save_res(iter, acc, best, term):
    base_path = f"models/{C.get()['exp_name']}"
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), base_path)
    os.makedirs(base_path, exist_ok=True)
    f = open(os.path.join(base_path, "iter_acc.csv"), "a", newline="")
    wr = csv.writer(f)
    wr.writerow([iter, acc, best, term])
    f.close()

def step_w_log(self):
    original = gorilla.get_original_attribute(ray.tune.trial_runner.TrialRunner, 'step')

    # log
    cnts = OrderedDict()
    for status in [Trial.RUNNING, Trial.TERMINATED, Trial.PENDING, Trial.PAUSED, Trial.ERROR]:
        cnt = len(list(filter(lambda x: x.status == status, self._trials)))
        cnts[status] = cnt
    best_top1_acc = 0.
    # last_acc = 0.
    for trial in filter(lambda x: x.status == Trial.TERMINATED, self._trials):
        if not trial.last_result:
            continue
        best_top1_acc = max(best_top1_acc, trial.last_result['top1_valid'])
        # last_acc = trial.last_result['top1_valid']
    # save_res(self._iteration, last_acc, best_top1_acc, cnts[Trial.TERMINATED])
    print('iter', self._iteration, 'top1_acc=%.3f' % best_top1_acc, cnts, end='\r')
    return original(self)


patch = gorilla.Patch(ray.tune.trial_runner.TrialRunner, 'step', step_w_log, settings=gorilla.Settings(allow_hit=True))
gorilla.apply(patch)


logger = get_logger('Fast AutoAugment')

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
    parser.add_argument('--num-search', type=int, default=200)
    parser.add_argument('--cv-num', type=int, default=5)
    parser.add_argument('--cv-ratio', type=float, default=0.4)
    parser.add_argument('--decay', type=float, default=-1)
    parser.add_argument('--redis', type=str)
    parser.add_argument('--per-class', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--rpc', type=int, default=10)
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--iter', type=int, default=5)
    parser.add_argument('--childaug', type=str, default="clean")
    parser.add_argument('--mode', type=str, default="ppo")
    parser.add_argument('--load_search', type=str)
    parser.add_argument('--rand_search', action='store_true')
    parser.add_argument('--exp_name', type=str)

    parser.add_argument('--lstm_size', type=int, default=100)
    parser.add_argument('--emb_size', type=int, default=32)
    parser.add_argument('--c_lr', type=float, default=0.00035)

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    C.get()['exp_name'] = args.exp_name
    if args.decay > 0:
        logger.info('decay=%.4f' % args.decay)
        C.get()['optimizer']['decay'] = args.decay
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', C.get()['exp_name'])
    os.makedirs(base_path, exist_ok=True)
    add_filehandler(logger, os.path.join(base_path, '%s_%s_cv%.1f.log' % (C.get()['dataset'], C.get()['model']['type'], args.cv_ratio)))
    logger.info('configuration...')
    logger.info(json.dumps(C.get().conf, sort_keys=True, indent=4))
    logger.info('initialize ray...')
    ray.init(address=args.redis)

    num_result_per_cv = args.rpc
    gr_num = args.gr_num
    cv_num = args.cv_num
    C.get()["cv_num"] = cv_num
    bench_policy_set = C.get()["aug"]
    if 'test_dataset' not in C.get().conf:
        C.get()['test_dataset'] = C.get()['dataset']
    copied_c = copy.deepcopy(C.get().conf)

    logger.info('search augmentation policies, dataset=%s model=%s' % (C.get()['dataset'], C.get()['model']['type']))
    logger.info('----- Train without Augmentations cv=%d ratio(test)=%.1f -----' % (cv_num, args.cv_ratio))
    w.start(tag='train_no_aug')
    paths = [_get_path(C.get()['dataset'], C.get()['model']['type'], '%s_ratio%.1f_fold%d' % (args.childaug, args.cv_ratio, i)) for i in range(cv_num)]
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
    logger.info('----- Search Test-Time Augmentation Policies -----')
    w.start(tag='search-g_train')
    ops = augment_list(False)
    target_path = base_path + "/target_network.pt"
    ctl_save_path = base_path + "/ctl_network.pt"
    controller = Controller(n_subpolicy=args.num_policy, lstm_size=args.lstm_size, emb_size=args.emb_size,
                            operation_prob=0).cuda()
    ctl_config = {
            'dataroot': args.dataroot, 'split_ratio': args.cv_ratio,
            'target_path': target_path, 'ctl_save_path': ctl_save_path, 'childnet_paths': paths,
            'childaug': args.childaug, 'cv_num': args.cv_num
            'mode': args.mode, 'c_lr': args.c_lr
    }
    metrics = train_controller(controller, ctl_config, args.load_search)

    logger.info('getting results...')
    # Affinity Calculation
    augment = {
        'dataroot': args.dataroot, 'load_paths': paths,
        'cv_ratio_test': args.cv_ratio, "cv_num": args.cv_num,
    }
    for k in metrics:
        logger.info(json.dumps(metrics.metrics))
    # bench_affs = get_affinity(bench_policy_set, aff_bases, copy.deepcopy(copied_c), augment)
    # aug_affs = get_affinity(final_policy_group, aff_bases, copy.deepcopy(copied_c), augment)
    # # Diversity calculation
    # bench_divs = []
    # aug_divs = []
    # logger.info('processed in %.4f secs' % w.pause('train_aug'))
    # logger.info("bench_aff_avg={:.2f}".format(np.mean(bench_affs)))
    # logger.info("aug_aff_avg={:.2f}".format(np.mean(aug_affs)))
    # logger.info("bench_div_avg={:.2f}".format(np.mean(bench_divs)))
    # logger.info("aug_div_avg={:.2f}".format(np.mean(aug_divs)))
    logger.info(w)
