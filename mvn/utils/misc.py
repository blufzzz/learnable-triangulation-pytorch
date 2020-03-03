import os
import yaml
import json
import re

import torch


def config_to_str(config):
    return yaml.dump(yaml.safe_load(json.dumps(config)))  # fuck yeah


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_gradient_norm(named_parameters, silence=False):
    total_norm = 0.0
    for name, p in named_parameters:
        # print(name)
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        elif not silence:
            print (name, 'grad is None')

    total_norm = total_norm ** (1. / 2)

    return total_norm


def calc_gradient_magnitude(named_parameters, silence=False):
    total_amplitude = []
    for name, p in named_parameters:
        # print(name)
        if p.grad is not None:
            param_amplitude = p.grad.data.abs().max()
            total_amplitude += [param_amplitude.item()]
        elif not silence:
            print (name, 'grad is None')    

    total_amplitude = max(total_amplitude)

    return total_amplitude


def get_capacity(model):
    s_total = 0
    for param in model.parameters():
        s_total+=param.numel()
    return round(s_total / (10**6),2)

def description(model):
    for k, m in model._modules.items():
        print ('{}:  {}M params'.format(k,get_capacity(m)))