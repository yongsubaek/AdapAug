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
from AdapAug.archive import remove_deplicates, policy_decoder
from AdapAug.augmentations import augment_list
from AdapAug.common import get_logger, add_filehandler
from AdapAug.data import get_dataloaders, get_custom_dataloaders
from AdapAug.metrics import Accumulator
from AdapAug.networks import get_model, num_class
from AdapAug.train import train_and_eval, train_controller, batch_policy_decoder, train_and_eval_ctl
from theconf import Config as C, ConfigArgumentParser
from AdapAug.controller import Controller, RandAug

top1_valid_by_cv = defaultdict(lambda: list)


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
