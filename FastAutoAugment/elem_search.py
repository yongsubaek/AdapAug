import copy
import os
import sys
import time
from collections import OrderedDict, defaultdict

import torch

import numpy as np
from hyperopt import hp
import ray
import gorilla
from ray.tune.trial import Trial
from ray.tune.trial_runner import TrialRunner
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import register_trainable, run_experiments
from tqdm import tqdm

from pathlib import Path
lib_dir = (Path("__file__").parent).resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from FastAutoAugment.archive import remove_deplicates, policy_decoder
from FastAutoAugment.augmentations import augment_list
from FastAutoAugment.common import get_logger, add_filehandler
from FastAutoAugment.data import get_dataloaders, get_custom_dataloaders
from FastAutoAugment.metrics import Accumulator
from FastAutoAugment.networks import get_model, num_class
from FastAutoAugment.train import train_and_eval, train_controller, batch_policy_decoder, train_and_eval_ctl
from theconf import Config as C, ConfigArgumentParser
from FastAutoAugment.controller import Controller, RandAug

top1_valid_by_cv = defaultdict(lambda: list)

@ray.remote(num_gpus=1)
def eval_controller(config, controller, dataroot, cv_ratio=0., cv_fold=0, save_path=None, skip_exist=False):
    """
    training with augmented data and test with pure data
    """
    C.get()
    C.get().conf = config
    result = train_and_eval_ctl(None, controller, dataroot, test_ratio=cv_ratio, cv_fold=cv_fold, save_path=save_path, only_eval=skip_exist)
    return C.get()['model']['type'], cv_fold, result

def step_w_log(self):
    original = gorilla.get_original_attribute(ray.tune.trial_runner.TrialRunner, 'step')

    # log
    cnts = OrderedDict()
    for status in [Trial.RUNNING, Trial.TERMINATED, Trial.PENDING, Trial.PAUSED, Trial.ERROR]:
        cnt = len(list(filter(lambda x: x.status == status, self._trials)))
        cnts[status] = cnt
    best_top1_acc = 0.
    for trial in filter(lambda x: x.status == Trial.TERMINATED, self._trials):
        if not trial.last_result:
            continue
        best_top1_acc = max(best_top1_acc, trial.last_result['top1_valid'])
    print('iter', self._iteration, 'top1_acc=%.3f' % best_top1_acc, cnts, end='\r')
    return original(self)


patch = gorilla.Patch(ray.tune.trial_runner.TrialRunner, 'step', step_w_log, settings=gorilla.Settings(allow_hit=True))
gorilla.apply(patch)


logger = get_logger('Fast AutoAugment')


def _get_path(dataset, model, tag, basemodel=True):
    base_path = "models" if basemodel else f"models/{C.get()['exp_name']}"
    os.makedirs(base_path, exist_ok=True)
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '%s/%s_%s_%s.model' % (base_path, dataset, model, tag))     # TODO


@ray.remote(num_gpus=1)
def train_model(config, dataloaders, dataroot, augment, cv_ratio_test, cv_fold, save_path=None, skip_exist=False, reduced=False):
    C.get()
    C.get().conf = config
    C.get()['aug'] = augment

    result = train_and_eval(None, dataloaders, dataroot, cv_ratio_test, cv_fold, save_path=save_path, only_eval=skip_exist, reduced=reduced)
    return C.get()['model']['type'], cv_fold, result


if __name__ == '__main__':
    import json
    from pystopwatch2 import PyStopwatch
    w = PyStopwatch()

    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--dataroot', type=str, default='/mnt/ssd/data/', help='torchvision data folder')
    parser.add_argument('--until', type=int, default=5)
    parser.add_argument('--num-op', type=int, default=2)
    parser.add_argument('--num-policy', type=int, default=5)
    parser.add_argument('--num-search', type=int, default=200)
    parser.add_argument('--cv-ratio', type=float, default=0.40)
    parser.add_argument('--decay', type=float, default=-1)
    parser.add_argument('--redis', type=str)
    parser.add_argument('--per-class', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--cv-num', type=int, default=1)
    parser.add_argument('--exp_name', type=str)

    parser.add_argument('--lstm-size', type=int, default=100)
    parser.add_argument('--num-group', type=int, default=0)
    parser.add_argument('--gr-prob-weight', type=float, default=1e-3)
    parser.add_argument('--random-group', action='store_true', default=False)
    parser.add_argument('--random-aug', action='store_true')

    args = parser.parse_args()
    C.get()['exp_name'] = args.exp_name
    if args.decay > 0:
        logger.info('decay=%.4f' % args.decay)
        C.get()['optimizer']['decay'] = args.decay

    add_filehandler(logger, os.path.join('models', '%s_%s_cv%.1f.log' % (C.get()['dataset'], C.get()['model']['type'], args.cv_ratio)))
    logger.info('configuration...')
    logger.info(json.dumps(C.get().conf, sort_keys=True, indent=4))
    logger.info('initialize ray...')
    ray.init(address=args.redis, num_gpus=4)

    num_result_per_cv = 10
    cv_num = args.cv_num
    copied_c = copy.deepcopy(C.get().conf)

    dataloaders = get_custom_dataloaders(C.get()['dataset'], C.get()['batch'], C.get()['dataroot'], args.cv_ratio)
    logger.info('search augmentation policies, dataset=%s model=%s' % (C.get()['dataset'], C.get()['model']['type']))
    logger.info('----- Train without Augmentations cv=%d ratio(test)=%.1f -----' % (cv_num, args.cv_ratio))
    w.start(tag='train_no_aug')
    paths = [_get_path(C.get()['dataset'], C.get()['model']['type'], 'ratio%.1f_fold%d' % (args.cv_ratio, i)) for i in range(cv_num)]
    print(paths)
    reqs = [ # model training
        train_model.remote(copy.deepcopy(copied_c), dataloaders, args.dataroot, C.get()['aug'], args.cv_ratio, i, save_path=paths[i], skip_exist=True)
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
    for r_model, r_cv, r_dict in pretrain_results:
        logger.info('model=%s cv=%d top1_train=%.4f top1_valid=%.4f' % (r_model, r_cv+1, r_dict['top1_train'], r_dict['top1_valid']))
    logger.info('processed in %.4f secs' % w.pause('train_no_aug'))

    if args.until == 1:
        sys.exit(0)
    if not args.random_aug:
        logger.info('----- Search Test-Time Augmentation Policies -----')
        w.start(tag='search')
        ctl_save_path = f"./models/{args.exp_name}"
        os.makedirs(ctl_save_path, exist_ok=True)
        ctl_save_path += "/controller.pt"
        # for _ in range(0):  # run multiple times.
        #     for cv_fold in range(cv_num):
                # TODO: training controller -> search policy using lstm controller
        controller = Controller(n_subpolicy=args.num_policy, lstm_size=args.lstm_size, n_group=args.num_group, gr_prob_weight=args.gr_prob_weight,\
                                img_input=not args.random_group).cuda()
        metrics, baseline = train_controller(controller, dataloaders, paths[0], ctl_save_path)
    else:
        controller = RandAug(n_subpolicy=args.num_policy)

    logger.info('----- Train with Augmentations model=%s dataset=%s aug=%s ratio(test)=%.1f -----' % (C.get()['model']['type'], C.get()['dataset'], C.get()['aug'], args.cv_ratio))
    w.start(tag='train_aug')

    num_experiments = 4
    # default_path = [_get_path(C.get()['dataset'], C.get()['model']['type'], 'ratio%.1f_default%d' % (args.cv_ratio, _), False) for _ in range(num_experiments)]
    augment_path = [_get_path(C.get()['dataset'], C.get()['model']['type'], 'ratio%.1f_augment%d' % (args.cv_ratio, _), False) for _ in range(num_experiments)]
    # reqs = [train_model.remote(copy.deepcopy(copied_c), dataloaders, args.dataroot, C.get()['aug'], 0.0, 0, save_path=default_path[_], skip_exist=True) for _ in range(num_experiments)] + \
    reqs = [eval_controller.remote(copy.deepcopy(copied_c), controller, args.dataroot, 0.0, 0, save_path=augment_path[_]) for _ in range(num_experiments)]
            # [train_model.remote(copy.deepcopy(copied_c), dataloaders, args.dataroot, "fa_reduced_cifar10", 0.0, 0, save_path=augment_path[_]) for _ in range(num_experiments)] + \

    tqdm_epoch = tqdm(range(C.get()['epoch']))
    is_done = False
    for epoch in tqdm_epoch:
        while True:
            epochs = OrderedDict()
            for exp_idx in range(num_experiments):
                # try:
                #     if os.path.exists(default_path[exp_idx]):
                #         latest_ckpt = torch.load(default_path[exp_idx])
                #         epochs['default_exp%d' % (exp_idx + 1)] = latest_ckpt['epoch']
                # except:
                #     pass
                try:
                    if os.path.exists(augment_path[exp_idx]):
                        latest_ckpt = torch.load(augment_path[exp_idx])
                        epochs['augment_exp%d' % (exp_idx + 1)] = latest_ckpt['epoch']
                except:
                    pass

            tqdm_epoch.set_postfix(epochs)
            if len(epochs) == num_experiments and min(epochs.values()) >= C.get()['epoch']:
                is_done = True
            if len(epochs) == num_experiments and min(epochs.values()) >= epoch:
                break
            time.sleep(10)
        if is_done:
            break

    logger.info('getting results...')
    final_results = ray.get(reqs)
    print(final_results)
    print(final_results[0])
    print(final_results[0][-1]['top1_test'])
    for train_mode in ['augment']: # ['default', 'augment']
        avg = 0.
        for _ in range(num_experiments):
            r_model, r_cv, r_dict = final_results.pop(0)
            logger.info('[%s] top1_train=%.4f top1_test=%.4f' % (train_mode, r_dict['top1_train'], r_dict['top1_test']))
            avg += r_dict['top1_test']
        avg /= num_experiments
        logger.info('[%s] top1_test average=%.4f (#experiments=%d)' % (train_mode, avg, num_experiments))
    logger.info('processed in %.4f secs' % w.pause('train_aug'))

    logger.info(w)
