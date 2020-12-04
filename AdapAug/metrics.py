import copy

import torch
import numpy as np
from collections import defaultdict
from collections.abc import Iterable
from torch import nn

class Tracker:
    def __init__(self):
        self.trace = defaultdict(lambda: [])
        self.accum = Accumulator()

    def add(self, key, value):
        self.trace[key].append(value)
        if not isinstance(value, Iterable):
            self.accum.add(key, value)

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.trace[item]

    def __setitem__(self, key, value):
        self.trace[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.trace))

    def reset_accum(self):
        self.accum = Accumulator()
        return self.accum

    def items(self):
        return self.accum.items()

    def last(self):
        return dict( [ (k,v[-1]) for k,v in self.get_dict().items()])

    def __str__(self):
        repr = ""
        for k, v in self.items():
            if type(v) in [str, int]:
                repr += f"{k}: {v:d}\t"
            else:
                repr += f"{k}: {v:.4f}\t"
        return repr

    def __truediv__(self, other):
        return self.accum / other


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1. / batch_size))
    return res


class CrossEntropyLabelSmooth(torch.nn.Module):
    def __init__(self, num_classes, epsilon, reduction='mean'):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, target):  # pylint: disable=redefined-builtin
        log_probs = self.logsoftmax(input)
        targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        if self.epsilon > 0.0:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        targets = targets.detach()
        loss = (-targets * log_probs)

        if self.reduction in ['avg', 'mean']:
            loss = torch.mean(torch.sum(loss, dim=1))
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == "batched_sum":
            loss = torch.sum(loss, dim=1)
        return loss


class Accumulator:
    def __init__(self):
        self.metrics = defaultdict(lambda: 0)

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        # return str(dict(self.metrics))
        repr = ""
        for k, v in dict(self.metrics).items():
            if type(v) in [str, int]:
                repr += f"{k}: {v:d} "
            else:
                repr += f"{k}: {v:.4f} "
        return repr

    def __truediv__(self, other):
        newone = Accumulator()
        for key, value in self.items():
            if isinstance(other, str):
                if other == key or key in ['time']:
                    newone[key] = value
                else:
                    newone[key] = value / self[other]
            else:
                newone[key] = value / other
        return newone


class SummaryWriterDummy:
    def __init__(self, log_dir):
        pass

    def add_scalar(self, *args, **kwargs):
        pass
