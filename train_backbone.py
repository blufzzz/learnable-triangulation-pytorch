import os
import shutil
import argparse
import time
import json
from datetime import datetime
from collections import defaultdict
from itertools import islice
import pickle
import copy

import numpy as np
import cv2

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

from mvn.models import pose_hrnet 
from mvn.models import pose_resnet 

from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss, JointsMSELoss
from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.models import pose_resnet
from mvn.datasets.human36m import Human36MTemporalDataset, Human36MMultiViewDataset
from mvn.datasets import utils as dataset_utils


from mvn.utils.multiview import project_3d_points_to_image_plane_without_distortion
from mvn.utils.vis import draw_2d_pose, fig_to_array
from mvn.utils.img import image_batch_to_numpy, denormalize_image,to_numpy

from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt


def generate_target(joints, 
                    joints_vis=None, 
                    num_joints=17, 
                    target_type='gaussian', 
                    sigma=3, 
                    heatmap_size=[96,96],
                    image_size=[384,384]):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        heatmap_size = np.array(heatmap_size)
        image_size = np.array(image_size)
        if joints_vis is not None:
            target_weight[:, 0] = joints_vis[:, 0]

        if target_type == 'gaussian':
            target = np.zeros((num_joints,
                               heatmap_size[1],
                               heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = sigma * 3

            for joint_id in range(num_joints):
                feat_stride = image_size / heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

#         if use_different_joints_weight:
#             target_weight = np.multiply(target_weight, joints_weight)

        return target, target_weight
    

def setup_human36m_dataloaders(config, is_train, distributed_train):
    train_dataloader = None
    train_sampler = None

    # parameters for both val\train
    singleview_dataset = config.dataset.singleview if hasattr(config.dataset, 'singleview') else False    
    pivot_type = config.dataset.pivot_type if hasattr(config.dataset, "pivot_type") else 'first'
    dt = config.dataset.dt if hasattr(config.dataset, "dt") else 1
    dataset_type = Human36MTemporalDataset if singleview_dataset else Human36MMultiViewDataset
    dilation = config.dataset.dilation if hasattr(config.dataset, 'dilation') else 0
    keypoints_per_frame=config.dataset.keypoints_per_frame if hasattr(config.dataset, 'keypoints_per_frame') else False
    dilation_type = config.dataset.dilation_type if hasattr(config.dataset, 'dilation_type') else 'constant'
    if is_train:
        # train
        train_dataset = dataset_type(
            h36m_root=config.dataset.train.h36m_root,
            pred_results_path=config.dataset.train.pred_results_path if hasattr(config.dataset.train, "pred_results_path") else None,
            train=True,
            test=False,
            image_shape=config.image_shape if hasattr(config, "image_shape") else (384, 384),
            labels_path=config.dataset.train.labels_path,
            with_damaged_actions=config.dataset.train.with_damaged_actions,
            scale_bbox=config.dataset.train.scale_bbox,
            kind=config.kind,
            undistort_images=config.dataset.train.undistort_images,
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [],
            crop=config.dataset.train.crop if hasattr(config.dataset.train, "crop") else True,
            dt = dt,
            dilation = dilation,
            evaluate_cameras = [0,1,2,3],
            keypoints_per_frame=keypoints_per_frame,
            pivot_type = pivot_type,
            dilation_type = dilation_type
            )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.train.shuffle and (train_sampler is None), # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.train.randomize_n_views,
                                                     min_n_views=config.dataset.train.min_n_views,
                                                     max_n_views=config.dataset.train.max_n_views),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn
            )

    val_dataset = dataset_type(
        h36m_root=config.dataset.val.h36m_root,
        pred_results_path=config.dataset.val.pred_results_path if hasattr(config.dataset.val, "pred_results_path") else None,
        train=False,
        test=True,
        image_shape=config.image_shape if hasattr(config, "image_shape") else (384, 384),
        labels_path=config.dataset.val.labels_path,
        with_damaged_actions=config.dataset.val.with_damaged_actions,
        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
        scale_bbox=config.dataset.val.scale_bbox,
        kind=config.kind,
        undistort_images=config.dataset.val.undistort_images,
        ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
        crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
        dt = dt,
        dilation = dilation,
        evaluate_cameras = [0,1,2,3],
        keypoints_per_frame=keypoints_per_frame,
        pivot_type = pivot_type,
        dilation_type = dilation_type
        )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                 min_n_views=config.dataset.val.min_n_views,
                                                 max_n_views=config.dataset.val.max_n_views),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn
        )

    return train_dataloader, val_dataloader, train_sampler


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
    parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")
    parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--logdir", type=str, default="/Vol1/dbstore/datasets/k.iskakov/logs/multi-view-net-repr", help="Path, where logs will be stored")
    parser.add_argument('--experiment_comment', default='', type=str)
    parser.add_argument('--experiment_dir', type=str)

    args = parser.parse_args()
    return args


def setup_dataloaders(config, is_train=True, distributed_train=False):
    if config.dataset.kind == 'human36m':
        train_dataloader, val_dataloader, train_sampler = setup_human36m_dataloaders(config, is_train, distributed_train)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))

    return train_dataloader, val_dataloader, train_sampler


def setup_experiment(config, model_name, is_train=True, existed_path=None):

    if existed_path is None:
        prefix = "" if is_train else "eval_"

        if config.title:
            experiment_title = config.title + "_" + model_name
        else:
            experiment_title = model_name

        experiment_title = config.experiment_comment if config.experiment_comment else prefix + experiment_title

        experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
        print("Experiment name: {}".format(experiment_name))

        experiment_dir = os.path.join(args.logdir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

        # tensorboard
        writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

        # dump config to tensorboard
        writer.add_text(misc.config_to_str(config), "config", 0)

    else:
        experiment_dir = existed_path
        # tensorboard
        writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    return experiment_dir, writer


def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        return False

    torch.cuda.set_device(args.local_rank)

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"

    torch.manual_seed(args.seed)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return True


def one_epoch(backbone,
              opt, 
              config, 
              dataloader, 
              device, 
              epoch, 
              n_iters_total=0, 
              is_train=True, 
              master=False, 
              experiment_dir=None, 
              writer=None):

    name = "train" if is_train else "val"

    if is_train:
        backbone.train()
    else:
        backbone.eval()

    batch_time = misc.AverageMeter()
    data_time = misc.AverageMeter()
    metric_dict = defaultdict(list)

    criterion = JointsMSELoss().to(device)

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        end = time.time()

        iterator = enumerate(dataloader)
        if is_train and config.opt.n_iters_per_epoch is not None:
            iterator = islice(iterator, config.opt.n_iters_per_epoch)

        for iter_i, batch in iterator:
            with autograd.detect_anomaly():
                # measure data loading time
                data_time.update(time.time() - end)

                if batch is None:
                    print("Found None batch")
                    continue

                (images_batch, 
                keypoints_3d_batch_gt, 
                keypoints_3d_validity_batch_gt, 
                proj_matricies_batch) = dataset_utils.prepare_batch(batch, device)
                
                batch_size, n_views = images_batch.shape[:2]
                image_shape = images_batch.shape[-2:]
                images_batch = images_batch.view(-1, 3, *image_shape)
                
                (heatmaps, 
                features, 
                alg_confidences, 
                vol_confidences, 
                bottleneck) = backbone(images_batch)

                # make target heatmaps
                targets = []
                for batch_index in range(batch_size):
                    for view_index in range(n_views):
                        proj_matrix = proj_matricies_batch[batch_index, view_index]
                        keypoints_3d = keypoints_3d_batch_gt[batch_index]

                        keypoints_2d_wrt_new = project_3d_points_to_image_plane_without_distortion(proj_matrix, keypoints_3d)
                        keypoints_2d_wrt_new = keypoints_2d_wrt_new.detach().cpu().numpy()

                        target, _ = generate_target(keypoints_2d_wrt_new,
                                                     sigma = config.backbone.SIGMA,
                                                     heatmap_size=config.backbone.HEATMAP_SIZE,
                                                     image_size=config.backbone.IMAGE_SIZE)
                        
                        targets.append(torch.tensor(target, dtype=torch.float32, requires_grad=True))

                target = torch.stack(targets, 0).to(device)    
                # calculate loss
                total_loss = criterion(heatmaps, target)
                writer.add_scalar(f"{name}/total_loss", total_loss.item(), n_iters_total)

                if is_train:
                    opt.zero_grad()
                    total_loss.backward()

                    if hasattr(config.opt, "grad_clip"):
                        torch.nn.utils.clip_grad_norm_(features_regressor.parameters(), config.opt.grad_clip / config.opt.lr)

                    metric_dict['grad_norm_times_lr'].append(config.opt.lr * misc.calc_gradient_norm(filter(lambda x: x[1].requires_grad,
                                                                                                     backbone.named_parameters())))
                    metric_dict['grad_amplitude_times_lr'].append(config.opt.lr * misc.calc_gradient_magnitude(filter(lambda x: x[1].requires_grad,
                                                                                                                backbone.named_parameters())))
                    opt.step()

                if n_iters_total % config.vis_freq == 0:
                    vis_kind = config.kind

                    images_batch = images_batch.view(batch_size, n_views, 3, *image_shape)
                    heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[-3:])

                    for batch_index in range(min(batch_size, config.vis_n_elements)):

                        fig, axes = plt.subplots(nrows=2 , ncols=n_views, figsize = (n_views*10, 20))
                        images = image_batch_to_numpy(images_batch[batch_index])
                        images = denormalize_image(images).astype(np.uint8)
                        images = images[..., ::-1]  # bgr -> rgb

                        for view_index in range(n_views):
                            proj_matrix = proj_matricies_batch[batch_index, view_index]
                            keypoints_3d = keypoints_3d_batch_gt[batch_index]

                            image = images[view_index]
                            heatmap = heatmaps[batch_index, view_index].detach().cpu().numpy()

                            keypoints_2d_wrt_new = project_3d_points_to_image_plane_without_distortion(proj_matrix,
                                                                                                       keypoints_3d)
                            keypoints_2d_wrt_new = keypoints_2d_wrt_new.detach().cpu().numpy()
                            
                            axes[0,view_index].imshow(image)
                            axes[0,view_index].scatter(keypoints_2d_wrt_new[:, 0], keypoints_2d_wrt_new[:, 1], s=150, c='red')
                            axes[1,view_index].imshow(heatmap.sum(0)) # sum over keypoints

                            fig.tight_layout()
                            fig_image = fig_to_array(fig)
                            plt.close('all')

                        writer.add_image(f"{name}/images/{batch_index}", 
                                        fig_image.transpose(2, 0, 1), 
                                        global_step=n_iters_total)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # dump to tensorboard per-iter loss/metric stats
                if is_train:
                    for title, value in metric_dict.items():
                        writer.add_scalar(f"{name}/{title}", value[-1], n_iters_total)

                # dump to tensorboard per-iter time stats
                writer.add_scalar(f"{name}/batch_time", batch_time.avg, n_iters_total)
                writer.add_scalar(f"{name}/data_time", data_time.avg, n_iters_total)

                # dump to tensorboard per-iter stats about sizes
                writer.add_scalar(f"{name}/batch_size", batch_size, n_iters_total)
                writer.add_scalar(f"{name}/views", n_views, n_iters_total)

                n_iters_total += 1
                batch_start_time = time.time()

    # calculate evaluation metrics
    if master:
        # dump to tensorboard per-epoch stats
        for title, value in metric_dict.items():
            writer.add_scalar(f"{name}/{title}_epoch", np.mean(value), epoch)

    return n_iters_total


def main(args):
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    is_distributed = init_distributed(args)
    master = True
    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0

    if is_distributed:
        device = torch.device(args.local_rank)
    else:
        device = torch.device(0)

    # config
    config = cfg.load_config(args.config)
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size
    config.experiment_comment = args.experiment_comment
    backbone_name = config.backbone.name

    if backbone_name in ['hrnet32']:
        backbone = pose_hrnet.get_pose_net(config.backbone, device).to(device)
    elif backbone_name in ['resnet152', 'resnet50']:
        backbone = pose_resnet.get_pose_net(config.backbone, device).to(device)
    else:
        raise RuntimeError('Wrong backbone_name')  

    if config.backbone.init_weights:
        state_dict = torch.load(config.backbone.checkpoint)
        backbone.load_state_dict(state_dict, strict=True)
        print("Loaded weights for backbone from {} ...".format(config.backbone.checkpoint))
    else:
        print("Training from scratch...")

    # optimizer
    opt = optim.Adam(backbone.parameters(),
                     lr=config.opt.lr)

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=is_distributed)

    # experiment
    experiment_dir, writer = None, None
    if master:
        experiment_dir, writer = setup_experiment(config, type(backbone).__name__, is_train=not args.eval)

    # multi-gpu
    if is_distributed:
        backbone = DistributedDataParallel(backbone, device_ids=[device])

    # train loop
    n_iters_total_train, n_iters_total_val = 0, 0
    for epoch in range(config.opt.n_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        print ('EPOCH', epoch)

        if master:
            checkpoint_path = os.path.join(experiment_dir, "latest_checkpoint.pth")
            dict_to_save = {'model_state': backbone.state_dict(),
                            'opt_state' : opt.state_dict(),
                            "epoch": epoch}
            torch.save(dict_to_save, checkpoint_path)


        n_iters_total_train = one_epoch(backbone,
                                        opt, 
                                        config, 
                                        train_dataloader, 
                                        device, 
                                        epoch, 
                                        n_iters_total=n_iters_total_train, 
                                        is_train=True, 
                                        master=master, 
                                        experiment_dir=experiment_dir, 
                                        writer=writer)

        n_iters_total_val = one_epoch(backbone,
                                      opt, 
                                      config, 
                                      val_dataloader, 
                                      device, 
                                      epoch, 
                                      n_iters_total=n_iters_total_val, 
                                      is_train=False, 
                                      master=master, 
                                      experiment_dir=experiment_dir, 
                                      writer=writer)



        print(f"{n_iters_total_train} iters done, Epoch: {epoch}.")


if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)
