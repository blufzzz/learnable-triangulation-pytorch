import os
import yaml
import json
import re
import numpy as np
from mvn.utils.img import image_batch_to_numpy, denormalize_image,to_numpy
import torch

retval = {
    'subject_names': ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11'],
    'camera_names': ['54138969', '55011271', '58860488', '60457274'],
    'action_names': [
        'Directions-1', 'Directions-2',
        'Discussion-1', 'Discussion-2',
        'Eating-1', 'Eating-2',
        'Greeting-1', 'Greeting-2',
        'Phoning-1', 'Phoning-2',
        'Posing-1', 'Posing-2',
        'Purchases-1', 'Purchases-2',
        'Sitting-1', 'Sitting-2',
        'SittingDown-1', 'SittingDown-2',
        'Smoking-1', 'Smoking-2',
        'TakingPhoto-1', 'TakingPhoto-2',
        'Waiting-1', 'Waiting-2',
        'Walking-1', 'Walking-2',
        'WalkingDog-1', 'WalkingDog-2',
        'WalkingTogether-1', 'WalkingTogether-2']
}

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
        print ('{}:  {}M'.format(k,get_capacity(m)))


def get_start_stop_frame_indxs(labels):
    
    frame_idx = labels['table']['frame_idx']
    change_mask = np.concatenate((frame_idx[:-1] > frame_idx[1:], [True]))
    stop_frame_indxs = frame_idx[change_mask]
    
    videos_lengths = stop_frame_indxs+1
    stop_indexes = np.cumsum(videos_lengths) - 1
    
    start_indexes = np.array([0] + list(stop_indexes + 1)[:-1])
    
    return start_indexes, stop_indexes

def index_to_name(i_stop, stop_indexes, val=True):
    sublects = ['S9', 'S11']
    actions = retval['action_names']
    
    number = np.arange(len(stop_indexes))[i_stop <= stop_indexes][0]
    subject_number = number//len(actions)
    action_number = number % len(actions)
    return sublects[subject_number], actions[action_number]

def get_error_diffs(keypoints_gt, keypoints_pred, kind='mpii'):
    
    rmse = lambda x,y: np.sqrt((x-y)**2).mean()
    assert len(keypoints_gt) == len(keypoints_pred) 
    n_frames = len(keypoints_gt)
    
    gt_diffs, pred_diffs = [], []
    
    for i in range(1,n_frames):

        if kind == "mpii":
            root_index = 6 
        else:
            raise RuntimeError

        keypoints_gt_relative = keypoints_gt - keypoints_gt[:, root_index:root_index + 1, :]
        keypoints_pred_relative = keypoints_pred - keypoints_pred[:, root_index:root_index + 1, :]

        per_pose_error_relative = np.sqrt(((keypoints_gt_relative - keypoints_pred_relative) ** 2).sum(2)).mean(1)
        
        gt_diffs.append(rmse(keypoints_gt[i], keypoints_gt[i-1]))
        pred_diffs.append(rmse(keypoints_pred, keypoints_pred[i-1]))

    pred_diffs = np.array(pred_diffs)
    gt_diffs = np.array(gt_diffs)
    
    return per_pose_error_relative, gt_diffs, pred_diffs


def normalize_temporal_images_batch(images_batch, pivot_position):
    images_batch = images_batch[:,pivot_position]
    # normalize images batch
    images_batch = image_batch_to_numpy(images_batch.detach().cpu())
    images_batch = denormalize_image(images_batch).astype(np.uint8)
    images_batch = images_batch[..., ::-1]  # bgr -> rgb
    return images_batch            