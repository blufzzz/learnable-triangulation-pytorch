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
from mvn.datasets.human36m import Human36MSingleViewDataset, Human36MMultiViewDataset
from mvn.datasets import utils as dataset_utils

from IPython.core.debugger import set_trace

from train import init_distributed, setup_dataloaders, parse_args, setup_human36m_dataloaders, setup_experiment


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
        if is_train and config.n_iters_per_epoch is not None:
            iterator = islice(iterator, config.n_iters_per_epoch)

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
                features_aux = features[:,:-1,...].view(batch_size, dt, features_channels, *features_shape)

                features_pred = features_regressor(features_aux).unsqueeze(1)

                # calculate loss
                fr_loss = 0.0
                fr_loss = torch.sum(torch.abs(features_pred - features_gt)**2) / (features_channels * features_shape.numel() * batch_size)
                total_loss += fr_loss
                metric_dict['fr_loss'].append(fr_loss.item())

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

                        heatmaps_vis = vis.visualize_heatmaps(
                            images_batch, 
                            features_pred,
                            kind=vis_kind,
                            batch_index=batch_i, size=5,
                            max_n_rows=10, max_n_cols=10
                        )
                        writer.add_image(f"{name}/heatmaps/{batch_i}", heatmaps_vis.transpose(2, 0, 1), global_step=n_iters_total)

                    # dump weights to tensoboard
                    if n_iters_total % config.vis_freq == 0 and dump_weights:
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
                    writer.add_scalar(f"{name}/n_views", n_views, n_iters_total)

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

    process_features = nn.Sequential(
            nn.Conv2d(256, config.intermediate_features_dim, 1)
        )
    backbone = pose_resnet.get_pose_net(config.backbone)
    features_regressor = {
        "conv2d_unet": FeaturesAR_CNN2D_UNet(config.intermediate_features_dim*(config.dataset.dt-1),
                                                 config.intermediate_features_dim,
                                                 C = config.features_regressor_base_channels)
    }[config.features_regressor](config, device=device).to(device)

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
        opt = optim.Adam([{'params':process_features.parameters(), lr=config.process_features_lr},
                            'params':features_regressor.parameters(), lr=config.lr])

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
        for epoch in range(config.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

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
            # saving    
            if master:

                checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
                os.makedirs(checkpoint_dir, exist_ok=True)

                dict_to_save = {'model_state': features_regressor.state_dict(),'opt_state' : opt.state_dict()}
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
