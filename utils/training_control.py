# Adjustment during training

import os
import json
import torch


def adjust_learning_rate(optimizer, lr_decay):
    """
    Adjust learning rate with a factor.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay


def save_checkpoint(state, info_json, dirname):
    """
    Save a training checkpoint. 
    """
    try:
        os.makedirs(dirname)
    except OSError:
        pass
    with open(os.path.join(dirname, 'info.json'), 'w', encoding='utf-8') as f:
        json.dump(info_json, f, indent=4)
    torch.save(state, os.path.join(dirname, 'csn.ckpt'))
