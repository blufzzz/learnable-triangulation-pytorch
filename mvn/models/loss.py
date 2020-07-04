import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


from IPython.core.debugger import set_trace

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

##################

class KeypointsMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss

class KeypointsMSESmoothLoss(nn.Module):
    def __init__(self, threshold=400):
        super().__init__()

        self.threshold = threshold

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        diff = (keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity
        diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * (self.threshold ** 0.9)
        loss = torch.sum(diff) / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss


class KeypointsMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, diff, keypoints_binary_validity):
        dimension = diff.shape[-1]
        loss = torch.sum(torch.abs(diff) * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss

#######################

class KeypointsL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        loss = torch.sum(torch.sqrt(torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity, dim=2)))
        loss = loss / max(1, torch.sum(keypoints_binary_validity).item())
        return loss


class VolumetricCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coord_volumes_batch, volumes_batch_pred, keypoints_gt, keypoints_binary_validity):
        loss = 0.0
        n_losses = 0

        batch_size = volumes_batch_pred.shape[0]
        for batch_i in range(batch_size):
            coord_volume = coord_volumes_batch[batch_i]
            keypoints_gt_i = keypoints_gt[batch_i]

            coord_volume_unsq = coord_volume.unsqueeze(0)
            keypoints_gt_i_unsq = keypoints_gt_i.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            dists = torch.sqrt(((coord_volume_unsq - keypoints_gt_i_unsq) ** 2).sum(-1))
            dists = dists.view(dists.shape[0], -1)

            min_indexes = torch.argmin(dists, dim=-1).detach().cpu().numpy()
            min_indexes = np.stack(np.unravel_index(min_indexes, volumes_batch_pred.shape[-3:]), axis=1)

            for joint_i, index in enumerate(min_indexes):
                validity = keypoints_binary_validity[batch_i, joint_i]
                loss += validity[0] * (-torch.log(volumes_batch_pred[batch_i, joint_i, index[0], index[1], index[2]] + 1e-5))
                n_losses += 1

        return loss / n_losses



class GAN_loss(nn.Module):
    """docstring for GAN_loss"""
    def __init__(self):
        super().__init__()
    def forward(self, discriminator, keypoints_3d_pred, keypoints_3d_gt, discriminator_loss):

        if discriminator_loss:
            disc_on_real_data = discriminator(keypoints_3d_gt) 
            disc_on_fake_data = discriminator(keypoints_3d_pred) 
            logp_real_is_real = F.logsigmoid(disc_on_real_data)
            logp_gen_is_fake = torch.log(1. - disc_on_fake_data.sigmoid() + 1e-15)
            discriminator_loss =  - (logp_real_is_real + logp_gen_is_fake).mean()
            return discriminator_loss

        else:
            generator_loss =  -F.logsigmoid(discriminator(keypoints_3d_pred)).mean()    
            return generator_loss




class LSGAN_loss(nn.Module):
    """docstring for LSGAN_loss"""
    def __init__(self):
        super().__init__()
    def forward(self, discriminator, keypoints_3d_pred, keypoints_3d_gt):
        pass    
