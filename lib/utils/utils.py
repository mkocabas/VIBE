# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import yaml
import operator
from tqdm import tqdm
from functools import reduce
from typing import List, Union

def get_from_dict(dict, keys):
    return reduce(operator.getitem, keys, dict)

def tqdm_enumerate(iter):
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1

def iterdict(d):
    for k,v in d.items():
        if isinstance(v, dict):
            d[k] = dict(v)
            iterdict(v)
    return d

def accuracy(output, target):
    _, pred = output.topk(1)
    pred = pred.view(-1)

    correct = pred.eq(target).sum()

    return correct.item(), target.size(0) - correct.item()

def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step/decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def step_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def read_yaml(filename):
    return yaml.load(open(filename, 'r'))

def write_yaml(filename, object):
    with open(filename, 'w') as f:
        yaml.dump(object, f)

def bool_to_string(x: Union[List[bool],bool]) ->  Union[List[str],str]:
    """
    boolean to string conversion
    :param x: list or bool to be converted
    :return: string converted thing
    """
    if isinstance(x, bool):
        return [str(x)]
    for i, j in enumerate(x):
        x[i]=str(j)
    return x
