import numpy as np
import copy
from collections import defaultdict
from collections.abc import Iterable

def gen_assign_group(version, num_group=5):
    if version == 1:
        return assign_group
    elif version == 2:
        return lambda data, label=None: assign_group2(data,label,num_group)
    elif version == 3:
        return lambda data, label=None: assign_group3(data,label,num_group)
    elif version == 4:
        return assign_group4

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
    np.random.shuffle(_classes)
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
