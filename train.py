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
import sys

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
                                           VolumetricTemporalAdaINNet,\
                                           VolumetricFRAdaINNet
                                           
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss

from mvn.utils import img, multiview, op, vis, misc, cfg

from mvn.datasets.human36m import Human36MTemporalDataset, Human36MMultiViewDataset
from mvn.datasets import utils as dataset_utils

from IPython.core.debugger import set_trace

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


def setup_human36m_dataloaders(config, is_train, distributed_train):
    train_dataloader = None
    train_sampler = None

    # parameters for both val\train
    singleview_dataset = config.dataset.singleview if hasattr(config.dataset, 'singleview') else False    
    pivot_type = config.dataset.pivot_type if hasattr(config.dataset, "pivot_type") else 'first'
    dt = config.dataset.dt if hasattr(config.dataset, "dt") else 1
    dataset_type = Human36MTemporalDataset if singleview_dataset else Human36MMultiViewDataset
    dilation = config.dataset.dilation if hasattr(config.dataset, 'dilation') else 0
    dilation_type = config.dataset.dilation_type if hasattr(config.dataset, 'dilation_type') else 'constant'
    keypoints_per_frame=config.dataset.keypoints_per_frame if hasattr(config.dataset, 'keypoints_per_frame') else False


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
        evaluate_cameras = [0,1,2,3],
        keypoints_per_frame=keypoints_per_frame,
        pivot_type = pivot_type,
        dilation_type = dilation_type,
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


def save(experiment_dir, model, opt, epoch, discriminator, opt_discr, use_temporal_discriminator_loss):

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
    os.makedirs(checkpoint_dir, exist_ok=True)

    dict_to_save = {'model_state': model.state_dict(),'opt_state' : opt.state_dict()}
    if use_temporal_discriminator_loss:
        dict_to_save['discr_state'] = discriminator.state_dict()
        dict_to_save['discr_opt_state'] = opt_discr.state_dict()
    torch.save(dict_to_save, os.path.join(checkpoint_dir, "weights.pth"))


def one_epoch(model,
              criterion, 
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
    model_type = config.model.name

    singleview_dataset = dataloader.dataset.singleview if hasattr(dataloader.dataset, 'singleview') else False
    visualize_volumes = config.visualize_volumes if hasattr(config, 'visualize_volumes') else False
    visualize_heatmaps = config.visualize_heatmaps if hasattr(config, 'visualize_heatmaps') else False
    use_heatmaps = config.model.backbone.return_heatmaps if hasattr(config.model.backbone, 'return_heatmaps') else False
    use_temporal_discriminator_loss  = config.opt.use_temporal_discriminator_loss if hasattr(config.opt, "use_temporal_discriminator_loss") else False  
    dump_weights = config.opt.dump_weights if hasattr(config.opt, 'dump_weights') else False
    transfer_cmu_to_human36m = config.transfer_cmu_to_human36m if hasattr(config, "transfer_cmu_to_human36m") else False
    use_intermediate_fr_loss = config.opt.use_intermediate_fr_loss if hasattr(config.opt, "use_intermediate_fr_loss") else False

    if is_train:
        model.train()
    else:
        model.eval()

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

                images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = dataset_utils.prepare_batch(batch, device)

                heatmaps_pred, keypoints_2d_pred, cuboids_pred, base_points_pred = None, None, None, None
                
                if model_type == "alg" or model_type == "ransac":
                    (keypoints_3d_pred, 
                    keypoints_2d_pred, 
                    heatmaps_pred, 
                    confidences_pred) = model(images_batch, proj_matricies_batch, batch)

                elif model_type == "vol_temporal_fr_adain":
                    (keypoints_3d_pred, 
                     keypoints_3d_pred_fr, 
                     features_pred, 
                     features_pred_fr, 
                     volumes_pred, 
                     volumes_pred_fr, 
                     confidences_pred, 
                     cuboids_pred, 
                     coord_volumes_pred, 
                     base_points_pred) = model(images_batch, batch)

                elif model_type == "vol_temporal_adain":
                    (keypoints_3d_pred, 
                     keypoints_3d_pred_fr, 
                     features_pred, 
                     features_pred_fr, 
                     volumes_pred, 
                     volumes_pred_fr, 
                     confidences_pred, 
                     cuboids_pred, 
                     coord_volumes_pred, 
                     base_points_pred,
                     style_vector) = model(images_batch, batch)    

                else:
                    (keypoints_3d_pred, 
                    features_pred, 
                    volumes_pred, 
                    confidences_pred, 
                    cuboids_pred, 
                    coord_volumes_pred, 
                    base_points_pred) = model(images_batch, batch)    
                
                batch_size, n_views, image_shape = images_batch.shape[0], images_batch.shape[1], tuple(images_batch.shape[3:])
                n_joints = keypoints_3d_pred[0].shape[1]
                keypoints_3d_binary_validity_gt = (keypoints_3d_validity_gt > 0.0).type(torch.float32)
                scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0

                # root-relative coordinates for    
                # singleview dataset of multiview dataset with singleview setup
                if singleview_dataset:
                    coord_volumes_pred = coord_volumes_pred - base_points_pred.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    keypoints_3d_gt = op.root_centering(keypoints_3d_gt, config.kind)
                    keypoints_3d_pred = op.root_centering(keypoints_3d_pred, config.kind)
                    if model_type == "vol_temporal_fr_adain" and keypoints_3d_pred_fr is not None:
                        keypoints_3d_pred_fr = op.root_centering(keypoints_3d_pred_fr, config.kind)

                # calculate loss
                total_loss = 0.0
                if model_type == "vol_temporal_fr_adain" and use_intermediate_fr_loss:
                    intermediate_fr_loss_weight = config.opt.intermediate_fr_loss_weight if hasattr(config.opt, "intermediate_fr_loss_weight") else 1.
                    features_shape = features_pred.shape[-2:]
                    fr_loss = torch.sum(torch.abs(features_pred - features_pred_fr)**2) / (features_shape.numel() * batch_size)
                    total_loss += fr_loss*intermediate_fr_loss_weight
                    metric_dict['fr_loss'].append(fr_loss.item())

                if use_temporal_discriminator_loss and is_train:
                    pred_keypoints_features = keypoints_to_features(keypoints_3d_batch_pred_list[1]).transpose(1,0)
                    gt_keypoints_features = keypoints_to_features(keypoints_3d_batch_gt).transpose(1,0)
                    all_keypoints = torch.stack([pred_keypoints_features, gt_keypoints_features],0)
                    target = torch.cat([torch.zeros(1), torch.ones(1)]).long().cuda()
                    predicted = discriminator(all_keypoints)
                    keypoints_ce_criterion = nn.CrossEntropyLoss()

                    discriminator_loss = keypoints_ce_criterion(predicted,target)
                    generator_loss = -torch.log(predicted[:batch_size]).sum()

                    metric_dict['discriminator_loss'].append(discriminator_loss.item())
                    metric_dict['generator_loss'].append(generator_loss.item())

                    weight = config.opt.temporal_discriminator_loss_weight if hasattr(config.opt, "temporal_discriminator_loss_weight") else 1.0
                    total_loss += weight * generator_loss

                    discr_freq = config.opt.train_discriminator_freq if hasattr(config.opt, "train_discriminator_freq") else 1
                    if iter_i%discr_freq==0:

                        opt_discr.zero_grad()
                        discriminator_loss.backward()
                        opt_discr.step()
                        continue

                loss = criterion(keypoints_3d_pred * scale_keypoints_3d, keypoints_3d_gt * scale_keypoints_3d, keypoints_3d_binary_validity_gt)
                total_loss += loss
                metric_dict[f'{config.opt.criterion}'].append(loss.item())

                # volumetric ce loss
                use_volumetric_ce_loss = config.opt.use_volumetric_ce_loss if hasattr(config.opt, "use_volumetric_ce_loss") else False
                if use_volumetric_ce_loss:
                    volumetric_ce_criterion = VolumetricCELoss()

                    loss = volumetric_ce_criterion(coord_volumes_pred, volumes_pred, keypoints_3d_gt, keypoints_3d_binary_validity_gt)
                    metric_dict['volumetric_ce_loss'].append(loss.item())

                    weight = config.opt.volumetric_ce_loss_weight if hasattr(config.opt, "volumetric_ce_loss_weight") else 1.0
                    total_loss += weight * loss

                metric_dict['total_loss'].append(total_loss.item())

                if is_train:
                    opt.zero_grad()
                    total_loss.backward()

                    if hasattr(config.opt, "grad_clip"):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.opt.grad_clip / config.opt.lr)

                    metric_dict['grad_norm_times_lr'].append(config.opt.lr * misc.calc_gradient_norm(filter(lambda x: x[1].requires_grad, model.named_parameters())))
                    metric_dict['grad_amplitude_times_lr'].append(config.opt.lr * misc.calc_gradient_magnitude(filter(lambda x: x[1].requires_grad, model.named_parameters())))

                    opt.step()

                # calculate metrics
                l2 = KeypointsL2Loss()(keypoints_3d_pred * scale_keypoints_3d, keypoints_3d_gt * scale_keypoints_3d,\
                                                                                    keypoints_3d_binary_validity_gt)
                metric_dict['l2'].append(l2.item())
                if model_type == "vol_temporal_fr_adain" and keypoints_3d_pred_fr is not None:
                    l2_fr = KeypointsL2Loss()(keypoints_3d_pred_fr * scale_keypoints_3d, keypoints_3d_gt * scale_keypoints_3d,\
                                                                                            keypoints_3d_binary_validity_gt)
                    metric_dict['l2_fr'].append(l2_fr.item())

                # base point l2
                if base_points_pred is not None and not config.model.use_gt_pelvis:
                    base_point_l2_list = []
                    for batch_i in range(batch_size):
                        base_point_pred = base_points_pred[batch_i]

                        if config.model.kind == "coco":
                            base_point_gt = (keypoints_3d_gt[batch_i, 11, :3] + keypoints_3d[batch_i, 12, :3]) / 2
                        elif config.model.kind == "mpii":
                            base_point_gt = keypoints_3d_gt[batch_i, 6, :3]

                        base_point_l2_list.append(torch.sqrt(torch.sum((base_point_pred * scale_keypoints_3d - base_point_gt * scale_keypoints_3d) ** 2)).item())

                    base_point_l2 = 0.0 if len(base_point_l2_list) == 0 else np.mean(base_point_l2_list)
                    metric_dict['base_point_l2'].append(base_point_l2)

                # plot visualization
                if singleview_dataset:
                    keypoints_3d_gt = op.root_centering(keypoints_3d_gt, config.kind, inverse=True)
                    keypoints_3d_pred = op.root_centering(keypoints_3d_pred, config.kind, inverse=True)
                    
                    if model_type == "vol_temporal_fr_adain" and keypoints_3d_pred_fr is not None:
                        keypoints_3d_pred_fr = op.root_centering(keypoints_3d_pred_fr, config.kind, inverse=True)    

                 # save answers for evalulation
                if not is_train:
                    results['keypoints_3d'].append(keypoints_3d_pred.detach().cpu().numpy())
                    results['indexes'].append(batch['indexes'])
                        
                if master:
                    if n_iters_total % config.vis_freq == 0:
                        vis_kind = config.kind
                        if (transfer_cmu_to_human36m):
                            vis_kind = "coco"

                        for batch_i in range(min(batch_size, config.vis_n_elements)):
                            keypoints_vis = vis.visualize_batch(
                                images_batch,
                                proj_matricies_batch, 
                                keypoints_3d_gt, 
                                keypoints_3d_pred,
                                kind=vis_kind,
                                cuboids_batch=cuboids_pred,
                                confidences_batch=confidences_pred,
                                batch_index=batch_i, size=5,
                                keypoints_3d_batch_pred_fr = keypoints_3d_pred_fr if model_type == "vol_temporal_fr_adain" else None,
                                max_n_cols=10
                            )
                            writer.add_image(f"{name}/keypoints_vis/{batch_i}", keypoints_vis.transpose(2, 0, 1), global_step=n_iters_total)

                            if visualize_heatmaps and use_heatmaps:    
                                heatmaps_vis = vis.visualize_heatmaps(
                                    images_batch, heatmaps_pred,
                                    kind=vis_kind,
                                    batch_index=batch_i, size=5,
                                    max_n_rows=10, max_n_cols=10
                                )
                                writer.add_image(f"{name}/heatmaps/{batch_i}", heatmaps_vis.transpose(2, 0, 1), global_step=n_iters_total)

                            if model_type == "vol" and visualize_volumes:
                                volumes_vis = vis.visualize_volumes(
                                    images_batch, volumes_pred, proj_matricies_batch,
                                    kind=vis_kind,
                                    cuboids_batch=cuboids_pred,
                                    batch_index=batch_i, size=5,
                                    max_n_rows=1, max_n_cols=16
                                )
                                writer.add_image(f"{name}/volumes/{batch_i}", volumes_vis.transpose(2, 0, 1), global_step=n_iters_total)

                    # dump weights to tensoboard
                    if n_iters_total % config.vis_freq == 0 and dump_weights:
                        for p_name, p in model.named_parameters():
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
            print ('Validation...')
            results['keypoints_3d'] = np.concatenate(results['keypoints_3d'], axis=0)
            results['indexes'] = np.concatenate(results['indexes'])
            provide_dataset_metric_for_all_cameras = config.dataset.val.provide_dataset_metric_for_all_cameras if hasattr(config.dataset.val,\
                                                                                        "provide_dataset_metric_for_all_cameras") else False
            try:
                if singleview_dataset:
                    # we need to consider all cameras
                    cameras_results = dataloader.dataset.evaluate(results['keypoints_3d'], results['indexes'])
                    # `dataset_metric` averaged by cameras 
                    scalar_metric = []

                    for camera_index in cameras_results.keys():
                        camera_scalar = cameras_results[camera_index][0]
                        scalar_metric.append(camera_scalar)
                        if provide_dataset_metric_for_all_cameras:
                            metric_dict['dataset_metric_{}_camera'.format(camera_index)].append(camera_scalar)

                    scalar_metric = np.mean(scalar_metric)
                    full_metric = cameras_results

                else:
                    scalar_metric, full_metric = dataloader.dataset.evaluate(results['keypoints_3d'])
            except Exception as e:
                print("Failed to evaluate. Reason: ", e)
                scalar_metric, full_metric = 0.0, {}

            metric_dict['dataset_metric'].append(scalar_metric)
            print ('Dataset metric', scalar_metric)

            checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)

            # dump results
            with open(os.path.join(checkpoint_dir, "results.pkl"), 'wb') as fout:
                pickle.dump(results, fout)

            # dump full metric
            with open(os.path.join(checkpoint_dir, "metric.json".format(epoch)), 'w') as fout:
                json.dump(full_metric, fout, indent=4, sort_keys=True)

        # dump to tensorboard per-epoch stats
        for title, value in metric_dict.items():
            writer.add_scalar(f"{name}/{title}_epoch", np.mean(value), epoch)

    return n_iters_total


def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        return False

    torch.cuda.set_device(args.local_rank)

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"

    torch.manual_seed(args.seed)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return True


def main(args):
    
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))
    
    config = cfg.load_config(args.config)
    is_distributed = init_distributed(args) and config.distributed_train
    print ('Distributed training' if is_distributed else 'No Distributed training')
    master = True

    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0

    device = torch.device(args.local_rank) if is_distributed else torch.device(0)

    # options
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size
    config.experiment_comment = args.experiment_comment
    use_temporal_discriminator_loss  = config.opt.use_temporal_discriminator_loss if hasattr(config.opt, "use_temporal_discriminator_loss") else False  
    save_model = config.opt.save_model if hasattr(config.opt, "save_model") else True

    model = {
        "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        "vol": VolumetricTriangulationNet,
        "vol_temporal": VolumetricTemporalNet,
        "vol_temporal_adain":VolumetricTemporalAdaINNet,
        "vol_temporal_fr_adain":VolumetricFRAdaINNet,
        "vol_temporal_lstm_v2v":VolumetricTemporalNet
    }[config.model.name](config, device=device).to(device)

    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint)
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded pretrained weights for whole model")

    # criterion
    criterion_class = {
        "MSE": KeypointsMSELoss,
        "MSESmooth": KeypointsMSESmoothLoss,
        "MAE": KeypointsMAELoss
    }[config.opt.criterion]

    if config.opt.criterion == "MSESmooth":
        criterion = criterion_class(config.opt.mse_smooth_threshold)
    else:
        criterion = criterion_class()

    # optimizer
    opt = None
    if not args.eval:
        if config.model.name == "vol":
            opt = torch.optim.Adam(
                [{'params': model.backbone.parameters()},
                 {'params': model.process_features.parameters(), 'lr': config.opt.process_features_lr if hasattr(config.opt, "process_features_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(), 'lr': config.opt.volume_net_lr if hasattr(config.opt, "volume_net_lr") else config.opt.lr}
                ],
                lr=config.opt.lr
            )
        elif config.model.name == "vol_temporal_adain":
            opt = torch.optim.Adam(
                [{'params': model.backbone.parameters()},
                 {'params': model.process_features.parameters(), 'lr': config.opt.process_features_lr if hasattr(config.opt, "process_features_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(), 'lr': config.opt.volume_net_lr if hasattr(config.opt, "volume_net_lr") else config.opt.lr},
                 {'params': model.encoder.parameters(), 'lr': config.opt.encoder_lr if hasattr(config.opt, "encoder_lr") else config.opt.lr},
                 {'params': model.features_sequence_to_vector.parameters(), 'lr': config.opt.features_sequence_to_vector_lr if hasattr(config.opt, "features_sequence_to_vector_lr") else config.opt.lr},
                 {'params': model.affine_mappings.parameters(), 'lr': config.opt.affine_mappings_lr if hasattr(config.opt, "affine_mappings_lr") else config.opt.lr},
                ] + [{'params':model.auxilary_backbone.parameters(), 'lr': config.opt.auxilary_backbone_lr if hasattr(config.opt, "auxilary_backbone_lr") else config.opt.lr}] if model.use_auxilary_backbone else [],
                lr=config.opt.lr
            )
        elif config.model.name == "vol_temporal_fr_adain":
            opt = torch.optim.Adam(
                [{'params': model.backbone.parameters()},
                 {'params': model.process_features.parameters(), 'lr': config.opt.process_features_lr if hasattr(config.opt, "process_features_lr") else config.opt.lr},
                 {'params': model.features_regressor.parameters(), 'lr': config.opt.features_regressor_lr if hasattr(config.opt, "features_regressor_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(), 'lr': config.opt.volume_net_lr if hasattr(config.opt, "volume_net_lr") else config.opt.lr}
                ],
                lr=config.opt.lr
            )
        elif config.model.name == "vol_temporal_lstm_v2v":
            opt = torch.optim.Adam(
                [{'params': model.backbone.parameters()},
                 {'params': model.process_features.parameters(), 'lr': config.opt.process_features_lr if hasattr(config.opt, "process_features_lr") else config.opt.lr},
                 {'params': model.lstm_v2v.parameters(), 'lr': config.opt.lstm_v2v_lr if hasattr(config.opt, "lstm_v2v_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(), 'lr': config.opt.volume_net_lr if hasattr(config.opt, "volume_net_lr") else config.opt.lr}
                ],
                lr=config.opt.lr
            )
        elif config.model.name == "vol_temporal":
            opt = torch.optim.Adam(
                [{'params': model.backbone.parameters()},
                 {'params': model.process_features.parameters(), 'lr': config.opt.process_features_lr if hasattr(config.opt, "process_features_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(), 'lr': config.opt.volume_net_lr if hasattr(config.opt, "volume_net_lr") else config.opt.lr}
                ],
                lr=config.opt.lr
            )    
        else:
            opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.opt.lr)

    # use_temporal_discriminator_loss
    discriminator = None
    opt_discr = None    
    if use_temporal_discriminator_loss:
        discriminator = TemporalDiscriminator().to(device)
        opt_discr = optim.Adam(discriminator.parameters(), lr=config.opt.discr_lr)

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=is_distributed)

    # experiment
    experiment_dir, writer = None, None
    if master:
        # pass
        experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)

    # multi-gpu
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device])

    
    if not args.eval:
        # try:
        # train loop
        n_iters_total_train, n_iters_total_val = 0, 0
        for epoch in range(config.opt.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            n_iters_total_train = one_epoch(model,
                                            criterion, 
                                            opt, 
                                            config, 
                                            train_dataloader, 
                                            device, 
                                            epoch, 
                                            n_iters_total=n_iters_total_train, 
                                            is_train=True, 
                                            master=master, 
                                            experiment_dir=experiment_dir, 
                                            writer=writer,
                                            discriminator=discriminator,
                                            opt_discr=opt_discr)

            n_iters_total_val = one_epoch(model, 
                                          criterion, 
                                          opt, 
                                          config, 
                                          val_dataloader, 
                                          device, 
                                          epoch, 
                                          n_iters_total=n_iters_total_val, 
                                          is_train=False, 
                                          master=master, 
                                          experiment_dir=experiment_dir, 
                                          writer=writer,
                                          discriminator=discriminator,
                                          opt_discr=opt_discr)
            # saving    
            if master and save_model:
                print ('Saving model...')
                save(experiment_dir, model, opt, epoch, discriminator, opt_discr, use_temporal_discriminator_loss)
            print(f"{n_iters_total_train} iters done.")
        # except BaseException:
            # print('Exception:',sys.exc_info()[0])
            # print ('Saving model...')
            # save(experiment_dir, model, opt, epoch, discriminator, opt_discr, use_temporal_discriminator_loss)
                    
    else:
        if args.eval_dataset == 'train':
            one_epoch(model, 
                      criterion, 
                      opt, 
                      config, 
                      train_dataloader, 
                      device, 
                      0, 
                      n_iters_total=0, 
                      is_train=False, 
                      master=master, 
                      experiment_dir=experiment_dir, 
                      writer=writer,
                      discriminator=discriminator,
                      opt_discr=opt_discr)
        else:
            one_epoch(model, 
                        criterion, 
                        opt, 
                        config, 
                        val_dataloader, 
                        device, 
                        0, 
                        n_iters_total=0, 
                        is_train=False, 
                        master=master, 
                        experiment_dir=experiment_dir, 
                        writer=writer,
                        discriminator=discriminator,
                        opt_discr=opt_discr)

    print("Done.")

if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)
