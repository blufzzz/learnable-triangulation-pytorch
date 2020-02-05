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

from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from mvn.models.volumetric_temporal import VolumetricTemporalNet,\
                                           VolumetricLSTMAdaINNet,\
                                           VolumetricFRAdaINNet

from mvn.models.temporal import Seq2VecRNN,\
                                FeaturesAR_RNN,\
                                FeaturesAR_CNN1D,\
                                FeaturesAR_CNN2D_UNet,\
                                FeaturesAR_CNN2D_ResNet,\
                                FeaturesEncoder_Bottleneck,\
                                FeaturesEncoder_DenseNet                                           
                                           
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss
from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.models import pose_resnet
from mvn.datasets.human36m import Human36MTemporalDataset, Human36MMultiViewDataset
from mvn.datasets import utils as dataset_utils

from IPython.core.debugger import set_trace


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
            image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
            labels_path=config.dataset.train.labels_path,
            with_damaged_actions=config.dataset.train.with_damaged_actions,
            scale_bbox=config.dataset.train.scale_bbox,
            kind=config.kind,
            undistort_images=config.dataset.train.undistort_images,
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [],
            crop=config.dataset.train.crop if hasattr(config.dataset.train, "crop") else True,
            dt = dt,
            dilation = dilation,
            evaluate_cameras = config.dataset.train.evaluate_cameras if hasattr(config.dataset.train, "evaluate_cameras") else [0,1,2,3],
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
        image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
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
        evaluate_cameras = config.dataset.train.evaluate_cameras if hasattr(config.dataset.train, "evaluate_cameras") else [0,1,2,3],
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


def setup_experiment(config, model_name, is_train=True):
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

    return experiment_dir, writer


def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        return False

    torch.cuda.set_device(args.local_rank)

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"

    torch.manual_seed(args.seed)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")


def one_epoch(features_regressor,
              process_features, 
              backbone, 
              opt, 
              config, 
              dataloader, 
              device, 
              epoch, 
              n_iters_total=0, 
              is_train=True, 
              caption='', 
              master=False, 
              experiment_dir=None, 
              writer=None, 
              discriminator=None, 
              opt_discr=None):

    name = "train" if is_train else "val"
    model_type = config.features_regressor

    singleview_dataset = dataloader.dataset.singleview if hasattr(dataloader.dataset, 'singleview') else False
    dump_weights = config.dump_weights if hasattr(config, 'dump_weights') else False

    if is_train:
        features_regressor.train()
    else:
        features_regressor.eval()

    batch_time = misc.AverageMeter()
    data_time = misc.AverageMeter()

    metric_dict = defaultdict(list)

    results = defaultdict(list)

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

                (images_batch, _, _, _) = dataset_utils.prepare_batch(batch, device, is_train=is_train)
                
                batch_size, dt = images_batch.shape[:2]
                image_shape = images_batch.shape[-2:]
                images_batch = images_batch.view(-1, 3, *image_shape)
                
                heatmaps, features, alg_confidences, vol_confidences, bottleneck = backbone(images_batch)
                features = features.detach()
                
                features = process_features(features)
                features_shape = features.shape[-2:]
                features_channels = features.shape[1] # 32

                # process features before lifting to volume
                features = features.view(batch_size, dt, features_channels, *features_shape)
                features_gt = features[:,-1,...]
                features_aux = features[:,:-1,...]
		          
                features_pred = features_regressor(features_aux.view(batch_size, -1, *features_shape))

                # set_trace()

                # calculate loss
                total_loss = 0.0
                fr_loss = torch.sum((features_pred - features_gt)**2) / batch_size
                total_loss += fr_loss
                metric_dict['fr_loss'].append(total_loss.item())

                if is_train:
                    opt.zero_grad()
                    total_loss.backward()

                    if hasattr(config.opt, "grad_clip"):
                        torch.nn.utils.clip_grad_norm_(features_regressor.parameters(), config.opt.grad_clip / config.opt.lr)

                    metric_dict['grad_norm_times_lr'].append(config.opt.lr * misc.calc_gradient_norm(filter(lambda x: x[1].requires_grad,
                                                                                                     features_regressor.named_parameters())))
                    metric_dict['grad_amplitude_times_lr'].append(config.opt.lr * misc.calc_gradient_magnitude(filter(lambda x: x[1].requires_grad,
                                                                                                                features_regressor.named_parameters())))

                    opt.step()

                if n_iters_total % config.vis_freq == 0:
                    vis_kind = config.kind

                    for batch_i in range(min(batch_size, config.vis_n_elements)):

                        heatmaps_vis = vis.visualize_features(
                            images_batch.view(batch_size, dt, 3, *image_shape), 
                            torch.cat([features_aux,
                                       features_pred.unsqueeze(1), 
                                       features_gt.unsqueeze(1)], 1).detach(),
                            kind=vis_kind,
                            batch_index=batch_i, 
                            size=5,
                            max_n_rows=10, 
                            max_n_cols=10
                        )
                        writer.add_image(f"{name}/heatmaps/{batch_i}", 
                                        heatmaps_vis.transpose(2, 0, 1), 
                                        global_step=n_iters_total)

                    # dump weights to tensoboard
                    if dump_weights:
                        for p_name, p in features_regressor.named_parameters():
                            try:
                                writer.add_histogram(p_name, p.clone().cpu().data.numpy(), n_iters_total)
                            except ValueError as e:
                                print(e)
                                print(p_name, p)
                                exit()

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
                writer.add_scalar(f"{name}/dt", dt, n_iters_total)

                n_iters_total += 1
                batch_start_time = time.time()

    # calculate evaluation metrics
    if master:
        if not is_train:
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)

        # dump to tensorboard per-epoch stats
        for title, value in metric_dict.items():
            writer.add_scalar(f"{name}/{title}_epoch", np.mean(value), epoch)

    return n_iters_total


def main(args):
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    is_distributed = init_distributed(args) and config.distributed_train
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
    save_model = config.opt.save_model if hasattr(config.opt, 'save_model') else False

    process_features = nn.Sequential(nn.Conv2d(256, config.intermediate_features_dim, 1)).to(device)

    backbone = pose_resnet.get_pose_net(config.backbone).to(device)
    features_regressor = FeaturesAR_CNN2D_UNet(config.intermediate_features_dim*(config.dataset.dt-1),
                                                 config.intermediate_features_dim,
                                                 C = config.features_regressor_base_channels).to(device)

    if config.init_weights:
        state_dict = torch.load(config.checkpoint)
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        features_regressor.load_state_dict(state_dict, strict=True)
        print("Successfully loaded pretrained weights for whole features_regressor")

    # optimizer
    opt = None
    if not args.eval:
        opt = optim.Adam([{'params':process_features.parameters(), 'lr':config.opt.process_features_lr},
                            {'params':features_regressor.parameters(), 'lr':config.opt.lr}])

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=is_distributed)

    # experiment
    experiment_dir, writer = None, None
    if master:
        experiment_dir, writer = setup_experiment(config, type(features_regressor).__name__, is_train=not args.eval)

    # multi-gpu
    if is_distributed:
        features_regressor = DistributedDataParallel(features_regressor, device_ids=[device])

    if not args.eval:
        # train loop
        n_iters_total_train, n_iters_total_val = 0, 0
        for epoch in range(config.opt.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            print ('EPOCH', epoch)    
            n_iters_total_train = one_epoch(features_regressor,
                                            process_features,
                                            backbone,  
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

            n_iters_total_val = one_epoch(features_regressor,
                                          process_features,
                                          backbone,   
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

            if master and save_model:

                checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
                os.makedirs(checkpoint_dir, exist_ok=True)

                dict_to_save = {'model_state': features_regressor.state_dict(),
                                'opt_state' : opt.state_dict(),
                                'process_features_state':process_features.state_dict()}
                                
                torch.save(dict_to_save, os.path.join(checkpoint_dir, "weights.pth"))

            print(f"{n_iters_total_train} iters done.")
    else:
        if args.eval_dataset == 'train':
            one_epoch(features_regressor,
                      process_features,
                      backbone,   
                      opt, 
                      config, 
                      train_dataloader, 
                      device, 
                      0, 
                      n_iters_total=0, 
                      is_train=False, 
                      master=master, 
                      experiment_dir=experiment_dir, 
                      writer=writer)

        else:
            one_epoch(features_regressor,
                        process_features,
                        backbone,   
                        opt, 
                        config, 
                        val_dataloader, 
                        device, 
                        0, 
                        n_iters_total=0, 
                        is_train=False, 
                        master=master, 
                        experiment_dir=experiment_dir, 
                        writer=writer)

    print("Done.")

if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)
