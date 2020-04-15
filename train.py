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

from mvn.models.triangulation import VolumetricTriangulationNet
from mvn.models.volumetric_adain import VolumetricTemporalAdaINNet
from mvn.models.volumetric_lstm import VolumetricTemporalLSTM
from mvn.models.volumetric_grid import VolumetricTemporalGridDeformation
from mvn.models.temporal import Seq2VecCNN
from mvn.models.loss import KeypointsMSELoss, \
                            KeypointsMSESmoothLoss, \
                            KeypointsMAELoss, \
                            KeypointsL2Loss, \
                            VolumetricCELoss,\
                            GAN_loss,\
                            LSGAN_loss

from mvn.utils import img, multiview, op, vis, misc, cfg

from mvn.datasets.human36m import Human36MTemporalDataset, Human36MMultiViewDataset
from mvn.datasets import utils as dataset_utils

from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt

MAKE_EXPERIMENT_DIR = False

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
    keypoints_per_frame = config.dataset.keypoints_per_frame if hasattr(config.dataset, 'keypoints_per_frame') else False

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
            keypoints_per_frame=keypoints_per_frame,
            pivot_type = pivot_type,
            dilation_type = dilation_type,
            norm_image=config.dataset.train.norm_image if hasattr(config.dataset.train, "norm_image") else True,
            custom_iterator=config.dataset.custom_iterator if hasattr(config.dataset, "custom_iterator") else None
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
        keypoints_per_frame=keypoints_per_frame,
        pivot_type = pivot_type,
        dilation_type = dilation_type,
        norm_image=config.dataset.val.norm_image if hasattr(config.dataset.val, "norm_image") else True,
        custom_iterator=config.dataset.custom_iterator if hasattr(config.dataset, "custom_iterator") else None
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


def save(experiment_dir, model, opt, epoch, discriminator, opt_discr, use_temporal_discriminator):

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints") # , "{:04}".format(epoch)
    os.makedirs(checkpoint_dir, exist_ok=True)

    dict_to_save = {'model_state': model.state_dict(),'opt_state' : opt.state_dict()}
    if use_temporal_discriminator:
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
    silence = config.opt.silence if hasattr(config.opt, 'silence') else False

    singleview_dataset = config.dataset.singleview if hasattr(config.dataset, 'singleview') else False
    pivot_index =  {'first':config.dataset.dt-1,
                    'intermediate':config.dataset.dt//2}[config.dataset.pivot_type]
    
    visualize_volumes = config.visualize_volumes if hasattr(config, 'visualize_volumes') else False
    visualize_heatmaps = config.visualize_heatmaps if hasattr(config, 'visualize_heatmaps') else False
    visualize = config.visualize if hasattr(config, "visualize") else True

    scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0    
    transfer_cmu_to_human36m = config.transfer_cmu_to_human36m if hasattr(config, "transfer_cmu_to_human36m") else False
    use_temporal_discriminator = config.opt.use_temporal_discriminator if hasattr(config.opt, "use_temporal_discriminator") else False
    use_bone_length_term = config.opt.use_bone_length_term if hasattr(config.opt, 'use_bone_length_term') else False
    bone_length_weight = config.opt.bone_length_weight if hasattr(config.opt, 'bone_length_weight') else None
    keypoints_per_frame = config.dataset.keypoints_per_frame if hasattr(config.dataset, 'keypoints_per_frame') else False

    use_style_decoder = config.model.use_style_decoder if hasattr(config.model, 'use_style_decoder') else False
    use_time_weighted_loss = config.opt.use_time_weighted_loss if hasattr(config.opt, 'use_time_weighted_loss') else False

    use_style_pose_lstm_loss = config.model.use_style_pose_lstm_loss if hasattr(config.model, 'use_style_pose_lstm_loss') else False
    if use_style_pose_lstm_loss:
        style_pose_lstm_loss_weight = config.opt.style_pose_lstm_loss_weight if hasattr(config.opt, 'style_pose_lstm_loss_weight') else 0.1

    if use_temporal_discriminator:
        assert (discriminator is not None) and (opt_discr is not None) and (not model.evaluate_only_last_volume)
        adversarial_temporal_criterion = {'vanilla':GAN_loss(),
                                          'lsgan':LSGAN_loss()}[config.opt.adversarial_temporal_criterion]
        adversarial_temporal_loss_weight = config.opt.adversarial_temporal_loss_weight
        adversarial_generator_iters = config.opt.adversarial_generator_iters 
        train_generator_during_critic_iters = config.opt.train_generator_during_critic_iters                               

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
        if is_train and config.opt.n_objects_per_epoch is not None:
            iterator = islice(iterator, config.opt.n_objects_per_epoch)

        for iter_i, batch in iterator:
            with autograd.detect_anomaly():
                # measure data loading time
                data_time.update(time.time() - end)

                if batch is None:
                    print("Found None batch at iter {}, continue...".format(iter_i))
                    continue

                # set_trace()
                torch.cuda.empty_cache()
                (images_batch, 
                keypoints_3d_gt, 
                keypoints_3d_validity_gt, 
                proj_matricies_batch) = dataset_utils.prepare_batch(batch, device)

                heatmaps_pred, keypoints_2d_pred, cuboids_pred, base_points_pred = None, None, None, None
                
                if config.model.name == 'vol_temporal_grid':
                    (keypoints_3d_pred, 
                    features_pred, 
                    volumes_pred, 
                    confidences_pred, 
                    cuboids_pred, 
                    coord_volumes_pred, 
                    base_points_pred,
                    coord_offsets) = model(images_batch, batch)

                elif config.model.name == 'vol_temporal_adain':    
                    (keypoints_3d_pred, 
                    features_pred, 
                    volumes_pred, 
                    confidences_pred, 
                    cuboids_pred, 
                    coord_volumes_pred, 
                    base_points_pred,
                    style_vector,
                    unproj_features,
                    decoded_features) = model(images_batch, batch) 

                else:    
                    (keypoints_3d_pred, 
                    features_pred, 
                    volumes_pred, 
                    confidences_pred, 
                    cuboids_pred, 
                    coord_volumes_pred,
                    base_points_pred) = model(images_batch, batch)            
                    
                batch_size, dt = images_batch.shape[:2]
                keypoints_3d_binary_validity_gt = (keypoints_3d_validity_gt > 0.0).type(torch.float32)                

                ################
                # MODEL OUTPUT #   
                ################
                if use_style_pose_lstm_loss:
                    assert keypoints_per_frame and isinstance(keypoints_3d_pred, list)
                    
                    auxilary_keypoints_3d_gt = keypoints_3d_gt[:,pivot_index+1:]
                    auxilary_keypoints_3d_pred = keypoints_3d_pred[1]
                    auxilary_keypoints_3d_binary_validity_gt = keypoints_3d_binary_validity_gt[:,pivot_index+1:]

                    keypoints_3d_gt = keypoints_3d_gt[:,pivot_index]
                    keypoints_3d_pred = keypoints_3d_pred[0]
                    keypoints_3d_binary_validity_gt = keypoints_3d_binary_validity_gt[:,pivot_index]

                # root-relative coordinates for    
                # singleview dataset of multiview dataset with singleview setup
                if singleview_dataset:
                    coord_volumes_pred = coord_volumes_pred - base_points_pred.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    keypoints_3d_gt = op.root_centering(keypoints_3d_gt, config.kind)
                    keypoints_3d_pred = op.root_centering(keypoints_3d_pred, config.kind)

                ##################
                # CALCULATE LOSS #   
                ##################
                # MSE\MAE loss
                total_loss = 0.0
                loss = criterion((keypoints_3d_pred  - keypoints_3d_gt)*scale_keypoints_3d,
                                 keypoints_3d_binary_validity_gt)

                total_loss += loss
                metric_dict[f'{config.opt.criterion}'].append(loss.item())

                # lstm temporal pose-style loss
                if use_style_pose_lstm_loss:
                    if use_time_weighted_loss:
                        future_keypoints_loss_weight = torch.stack([torch.exp(torch.arange(0,(-dt//2)+1, -1, dtype=torch.float)) \
                                                                    for i in range(batch_size)]).view(batch_size, -1,1,1).to(device)
                    else:
                        future_keypoints_loss_weight = 1.    
                    # check auxilary_keypoints_3d_pred grad_fn
                    pose_lstm_diff = (auxilary_keypoints_3d_gt - auxilary_keypoints_3d_pred)*scale_keypoints_3d*future_keypoints_loss_weight
                    validity = auxilary_keypoints_3d_binary_validity_gt.view(-1, *auxilary_keypoints_3d_binary_validity_gt.shape[-2:])
                    pose_lstm_loss = criterion(pose_lstm_diff.view(-1, *pose_lstm_diff.shape[-2:]), validity)
                    weighted_style_pose_lstm_loss = style_pose_lstm_loss_weight * pose_lstm_loss
                    total_loss += weighted_style_pose_lstm_loss

                    metric_dict['style_pose_lstm_loss_weighted'].append(weighted_style_pose_lstm_loss.item())


                # Bone length loss
                if use_bone_length_term:
                    connectivity = vis.CONNECTIVITY_DICT[dataloader.dataset.kind]
                    bone_length_loss = 0.
                    for (index_from, index_to) in connectivity:
                        bone_length_loss += torch.norm(keypoints_3d_pred[:,index_from] - keypoints_3d_gt[:,index_to], dim=-1)
                    bone_length_loss = bone_length_loss.mean()
                    total_loss += bone_length_loss * bone_length_weight
                    metric_dict['bone_length_loss_weighted'].append(bone_length_loss.item()*bone_length_weight)

                # volumetric loss
                use_volumetric_ce_loss = config.opt.use_volumetric_ce_loss
                if use_volumetric_ce_loss:
                    volumetric_ce_criterion = VolumetricCELoss()

                    loss = volumetric_ce_criterion(coord_volumes_pred, 
                                                    volumes_pred, 
                                                    keypoints_3d_gt, 
                                                    keypoints_3d_binary_validity_gt)
                    metric_dict['volumetric_ce_loss'].append(loss.item())

                    weight = config.opt.volumetric_ce_loss_weight
                    total_loss += weight * loss

                # temporal adversarial loss        
                if use_temporal_discriminator: 
                    keypoints_3d_pred_seq = keypoints_3d_pred.view(batch_size, dt, -1)
                    keypoints_3d_gt_seq = keypoints_3d_gt.view(batch_size, dt, -1)

                    # discriminator step, no need gradient flow to generator
                    discriminator_loss = adversarial_temporal_criterion(discriminator, 
                                                                        keypoints_3d_pred_seq.clone().detach(), 
                                                                        keypoints_3d_gt_seq.clone().detach(),
                                                                        discriminator_loss=True)

                    if is_train:
                        opt_discr.zero_grad()
                        discriminator_loss.backward()
                        opt_discr.step()

                    # generator step
                    if is_train: 
                        discriminator.zero_grad()
                    generator_loss = adversarial_temporal_criterion(discriminator, 
                                                                    keypoints_3d_pred_seq,
                                                                    keypoints_3d_gt_seq,
                                                                    discriminator_loss=False)
                    
                    # add loss                    
                    if iter_i%adversarial_generator_iters == 0:
                        total_loss += generator_loss * adversarial_temporal_loss_weight

                    metric_dict[f'{config.opt.adversarial_temporal_criterion}_generator_loss'].append(generator_loss.item())
                    metric_dict[f'{config.opt.adversarial_temporal_criterion}_discriminator_loss'].append(discriminator_loss.item())


                if use_style_decoder:
                    if use_time_weighted_loss:
                        time_weights = torch.stack([torch.exp(torch.arange(0,(-dt//2)+1, -1, dtype=torch.float)) \
                                            for i in range(batch_size)]).view(batch_size, -1,1,1,1).to(device) # [bs,(dt//2)-1,1,1,1]
                    else:
                        time_weights = 1.    
                    style_decoder_loss = torch.norm(decoded_features - features_pred)*time_weights
                    style_decoder_loss = style_decoder_loss.mean()
                    style_decoder_loss_weight = config.opt.style_decoder_loss_weight
                    total_loss += style_decoder_loss * style_decoder_loss_weight
                    metric_dict['style_decoder_loss_weighted'].append(style_decoder_loss.item()*style_decoder_loss_weight)

                ############
                # BACKWARD #   
                ############
                if use_temporal_discriminator:
                    adversarial_condition =  (train_generator_during_critic_iters or iter_i%adversarial_generator_iters == 0)
                else:
                    adversarial_condition=True    
                if is_train and adversarial_condition:
                    opt.zero_grad()
                    total_loss.backward()        

                    if hasattr(config.opt, "grad_clip"):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.opt.grad_clip / config.opt.lr)

                    metric_dict['grad_norm_times_lr'].append(config.opt.lr * \
                                                             misc.calc_gradient_norm(filter(lambda x: x[1].requires_grad, \
                                                             model.named_parameters()), silence=silence))
                    metric_dict['grad_amplitude_times_lr'].append(config.opt.lr * \
                                                                  misc.calc_gradient_magnitude(filter(lambda x: x[1].requires_grad, \
                                                                  model.named_parameters()), silence=silence))

                    opt.step()


                ###########
                # METRICS #   
                ###########
                metric_dict['total_loss'].append(total_loss.item())

                l2 = KeypointsL2Loss()(keypoints_3d_pred * scale_keypoints_3d, \
                                       keypoints_3d_gt * scale_keypoints_3d, \
                                       keypoints_3d_binary_validity_gt)

                metric_dict['l2'].append(l2.item())

                # base point l2
                if base_points_pred is not None and not config.model.use_gt_pelvis:
                    base_point_l2_list = []
                    for batch_i in range(base_points_pred.shape[0]):
                        base_point_pred = base_points_pred[batch_i]

                        if config.model.kind == "coco":
                            base_point_gt = (keypoints_3d_gt[batch_i, 11, :3] + keypoints_3d_gt[batch_i, 12, :3]) / 2
                        elif config.model.kind == "mpii":
                            base_point_gt = keypoints_3d_gt[batch_i, 6, :3]

                        base_point_l2_list.append(torch.sqrt(torch.sum((base_point_pred * scale_keypoints_3d - \
                                                                        base_point_gt * scale_keypoints_3d) ** 2)).item())

                    base_point_l2 = 0.0 if len(base_point_l2_list) == 0 else np.mean(base_point_l2_list)
                    metric_dict['base_point_l2'].append(base_point_l2)

                # plot visualization
                if singleview_dataset:
                    keypoints_3d_gt = op.root_centering(keypoints_3d_gt, config.kind, inverse=True)
                    keypoints_3d_pred = op.root_centering(keypoints_3d_pred, config.kind, inverse=True)
                    

                # save answers for evalulation
                if not is_train:
                    if keypoints_per_frame and keypoints_3d_pred.dim() == 4:
                        keypoints_3d_pred = keypoints_3d_pred.view(batch_size, -1, *keypoints_3d_pred.shape[-2:])[:,pivot_index]
                    results['keypoints_3d'].append(keypoints_3d_pred.detach().cpu().numpy())
                    results['indexes'].append(batch['indexes'])
                
                #################
                # VISUALIZATION #   
                #################        
                if master and MAKE_EXPERIMENT_DIR:
                    if n_iters_total % config.vis_freq == 0 and visualize:
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
                                batch_index=batch_i, 
                                size=5,
                                keypoints_3d_batch_pred_fr = None,
                                max_n_cols=10,
                                keypoints_per_frame = keypoints_per_frame
                            )
                            writer.add_image(f"{name}/keypoints_vis/{batch_i}", keypoints_vis.transpose(2, 0, 1), global_step=n_iters_total)

                            if visualize_heatmaps:    
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

                            if model_type == 'vol_temporal_grid':
                                fig=plt.figure()
                                diffs = coord_offsets[batch_i].abs().view(-1,3).mean(-1).detach().cpu().numpy()
                                plt.hist(diffs, bins=50)
                                fig.tight_layout()
                                hist_image = vis.fig_to_array(fig)
                                plt.close('all')    
                                writer.add_image(f"{name}/diff_hist/{batch_i}", hist_image.transpose(2, 0, 1), global_step=n_iters_total)

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
                    writer.add_scalar(f"{name}/n_views", dt, n_iters_total)

                    n_iters_total += 1
                    batch_start_time = time.time()

    # calculate evaluation metrics
    if master:
        # validation
        if not is_train:
            results['keypoints_3d'] = np.concatenate(results['keypoints_3d'], axis=0)
            results['indexes'] = np.concatenate(results['indexes'])

            try:
                scalar_metric = dataloader.dataset.evaluate(results['keypoints_3d'], result_indexes=results['indexes'])
                metric_dict['dataset_metric'].append(scalar_metric)
                print ('Dataset metric', scalar_metric)    

            except Exception as e:
                print("Failed to evaluate. Reason: ", e)
            
            if MAKE_EXPERIMENT_DIR:
                checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
                os.makedirs(checkpoint_dir, exist_ok=True)

                # dump results
                with open(os.path.join(checkpoint_dir, "results.pkl"), 'wb') as fout:
                    pickle.dump(results, fout)

        # dump to tensorboard per-epoch stats
        if MAKE_EXPERIMENT_DIR:
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
    print ('DISTRIBUTER TRAINING' if is_distributed else 'No Distributed training')
    master = True

    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0

    device = torch.device(args.local_rank) if is_distributed else torch.device(0)

    # options
    # config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size
    config.experiment_comment = args.experiment_comment
    use_temporal_discriminator  = config.opt.use_temporal_discriminator if \
                                     hasattr(config.opt, "use_temporal_discriminator") else False  
    save_model = config.opt.save_model if hasattr(config.opt, "save_model") else True
    
    model = {
        "vol": VolumetricTriangulationNet,
        "vol_temporal_adain":VolumetricTemporalAdaINNet,
        "vol_temporal_grid": VolumetricTemporalGridDeformation,
        "vol_temporal_lstm": VolumetricTemporalLSTM
    }[config.model.name](config, device=device).to(device)




    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint)['model_state']
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
                 {'params': model.process_features.parameters(), \
                            'lr': config.opt.process_features_lr if \
                            hasattr(config.opt, "process_features_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(), \
                            'lr': config.opt.volume_net_lr if \
                            hasattr(config.opt, "volume_net_lr") else config.opt.lr}
                ],
                lr=config.opt.lr
            )

        elif config.model.name == "vol_temporal_adain":
            opt = torch.optim.Adam(
                [{'params': model.backbone.parameters()},
                 {'params': model.process_features.parameters(), 
                            'lr': config.opt.process_features_lr if\
                            hasattr(config.opt, "process_features_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(), \
                            'lr': config.opt.volume_net_lr if \
                            hasattr(config.opt, "volume_net_lr") else config.opt.lr}
                ] + \

                ([{'params': model.features_sequence_to_vector.parameters(), \
                            'lr': config.opt.features_sequence_to_vector_lr if \
                            hasattr(config.opt, "features_sequence_to_vector_lr") else config.opt.lr}] if \
                            hasattr(model, "features_sequence_to_vector") else []) + \

                ([{'params':model.auxilary_backbone.parameters(), \
                            'lr': config.opt.auxilary_backbone_lr if \
                            hasattr(config.opt, "auxilary_backbone_lr") else config.opt.lr}] if \
                            model.use_auxilary_backbone else []) + \

                ([{'params':model.motion_extractor.parameters(), \
                            'lr': config.opt.motion_extractor_lr if \
                            hasattr(config.opt, "motion_extractor_lr") else config.opt.lr}] if \
                            model.use_motion_extractor else []) + \

                ([{'params':model.style_decoder.parameters(), \
                            'lr': config.opt.style_decoder_lr if \
                            hasattr(config.opt, "style_decoder_lr") else config.opt.lr}] if \
                            model.use_style_decoder else []) + \

                ([{'params':model.style_pose_lstm_loss_decoder.parameters(), \
                            'lr': config.opt.style_pose_lstm_decoder_lr if \
                            hasattr(config.opt, "style_pose_lstm_decoder_lr") else config.opt.lr}] if \
                            model.use_style_pose_lstm_loss else []),
                lr=config.opt.lr) 

        elif config.model.name == "vol_temporal_grid":
            opt = torch.optim.Adam(
                [{'params': model.backbone.parameters()},
                 {'params': model.process_features.parameters(), \
                            'lr': config.opt.process_features_lr if \
                            hasattr(config.opt, "process_features_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(), \
                            'lr': config.opt.volume_net_lr if \
                            hasattr(config.opt, "volume_net_lr") else config.opt.lr},
                 {'params': model.features_sequence_to_vector.parameters(), \
                            'lr': config.opt.features_sequence_to_vector_lr if \
                            hasattr(config.opt, "features_sequence_to_vector_lr") else config.opt.lr},
                 {'params': model.grid_deformator.parameters(), \
                            'lr': config.opt.grid_deformator_lr if \
                            hasattr(config.opt, "grid_deformator_lr") else config.opt.lr},
                ] + \
                ([{'params':model.auxilary_backbone.parameters(), \
                            'lr': config.opt.auxilary_backbone_lr if \
                            hasattr(config.opt, "auxilary_backbone_lr") else config.opt.lr}] if \
                            model.use_auxilary_backbone else []),
                lr=config.opt.lr)
            
        elif config.model.name == "vol_temporal_lstm":
             opt = torch.optim.Adam(
                [{'params': model.backbone.parameters()},
                 {'params': model.process_features.parameters(), \
                            'lr': config.opt.process_features_lr if \
                            hasattr(config.opt, "process_features_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(), \
                            'lr': config.opt.volume_net_lr if \
                            hasattr(config.opt, "volume_net_lr") else config.opt.lr},
                 {'params': model.lstm3d.parameters(), \
                            'lr': config.opt.lstm3d_lr if \
                            hasattr(config.opt, "lstm3d_lr") else config.opt.lr},
                ]+ \
                ([{'params':model.entangle_processing_net.parameters(), \
                            'lr': config.opt.entangle_processing_net_lr if \
                            hasattr(config.opt, "entangle_processing_net_lr") else config.opt.lr}] if \
                            model.disentangle else []) + \
                ([{'params':model.final_processing.parameters(), \
                            'lr': config.opt.final_processing_lr if \
                            hasattr(config.opt, "final_processing_lr") else config.opt.lr}] if \
                            model.use_final_processing else []),
                lr=config.opt.lr)                         
        else:
            raise RuntimeError('Unknown config.model.name')

    # use_temporal_discriminator
    opt_discr, discriminator = None, None
    if use_temporal_discriminator:
        discriminator = Seq2VecCNN(config.model.backbone.num_joints*3,
                                   output_features_dim=1, 
                                   intermediate_channels=config.discriminator.intermediate_channels, 
                                   normalization_type=config.discriminator.normalization_type,
                                   dt = config.dataset.dt,
                                   kernel_size = 3,
                                   n_groups = 32).to(device)

        opt_discr = optim.Adam(discriminator.parameters(), lr=config.opt.discr_lr)

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=is_distributed)

    # experiment
    experiment_dir, writer = None, None
    if master and MAKE_EXPERIMENT_DIR: 
        experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)
        print ('EXPERIMENT IN LOGDIR:', args.logdir)    
        
    # multi-gpu
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device])

    
    if not args.eval:
        # train loop
        n_iters_total_train, n_iters_total_val = 0, 0
        for epoch in range(config.opt.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            print ('Training...')    
            # set_trace()
            # time.sleep(15) 
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

            print ('Validation...')
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
                print (f'Saving model at {experiment_dir}...')
                save(experiment_dir, model, opt, epoch, discriminator, opt_discr, use_temporal_discriminator)
            print(f"epoch: {epoch}, iters: {n_iters_total_train}, done.")
   
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
