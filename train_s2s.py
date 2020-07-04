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

from mvn.models.s2s import S2SModel
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
from mvn.utils.op import get_coord_volumes, unproject_heatmaps, integrate_tensor_3d_with_coordinates, softmax_volumes

from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt

MAKE_EXPERIMENT_DIR = True

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


def save(experiment_dir, model, opt):

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    dict_to_save = {'model_state': model.state_dict(),'opt_state' : opt.state_dict()}

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
    pivot_type = config.dataset.pivot_type
    pivot_index =  {'first':config.dataset.dt-1,
                    'intermediate':config.dataset.dt//2}[pivot_type]
    
    scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0    
    transfer_cmu_to_human36m = config.transfer_cmu_to_human36m if hasattr(config, "transfer_cmu_to_human36m") else False
    use_bone_length_term = config.opt.use_bone_length_term if hasattr(config.opt, 'use_bone_length_term') else False
    bone_length_weight = config.opt.bone_length_weight if hasattr(config.opt, 'bone_length_weight') else None
    keypoints_per_frame = config.dataset.keypoints_per_frame if hasattr(config.dataset, 'keypoints_per_frame') else False

    use_style_pose_vce_loss = config.opt.use_style_pose_vce_loss if hasattr(config.opt, 'use_style_pose_vce_loss') else False
    use_style_pose_criterion_loss = config.opt.use_style_pose_criterion_loss if hasattr(config.opt, 'use_style_pose_criterion_loss') else False
    
    style_vce_weight = config.opt.style_vce_weight if use_style_pose_vce_loss else None
    style_criterion_weight = config.opt.style_criterion_weight if use_style_pose_criterion_loss else None

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

                debug = False    
                (images_batch, 
                keypoints_3d_gt, 
                keypoints_3d_validity_gt, 
                proj_matricies_batch) = dataset_utils.prepare_batch(batch, device)

                heatmaps_pred, keypoints_2d_pred, cuboids_pred, base_points_pred = None, None, None, None
                torch.cuda.empty_cache()

                (keypoints_3d_pred, 
                volumes_pred, 
                coord_volumes_pred,
                base_points_pred) = model(images_batch, batch)            
                    
                batch_size, dt = images_batch.shape[:2]
                keypoints_3d_binary_validity_gt = (keypoints_3d_validity_gt > 0.0).type(torch.float32)
                keypoints_shape = keypoints_3d_gt.shape[-2:]

                # print ('ITER:', iter_i,'MASTER:', master, 'KEYPOINTS ID:', keypoints_3d_gt.mean())
                if hasattr(model, 'style_vector_parameter'):
                    metric_dict['style_vector_parameter_variance'].append(model.style_vector_parameter.data.var().item())

                ################
                # MODEL OUTPUT #   
                ################
                if singleview_dataset:
                    if keypoints_3d_gt.dim() == 4: # FIX IT
                        keypoints_3d_gt = keypoints_3d_gt[:,-1,...]
                        keypoints_3d_binary_validity_gt = keypoints_3d_binary_validity_gt[:,-1,...]
                    coord_volumes_pred = coord_volumes_pred - base_points_pred.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    keypoints_3d_gt = op.root_centering(keypoints_3d_gt, config.kind)
                    keypoints_3d_pred = op.root_centering(keypoints_3d_pred, config.kind)

                ##################
                # CALCULATE LOSS #   
                ##################
                # MSE\MAE loss
                total_loss = 0.0
                if use_style_pose_criterion_loss:
                    loss = style_criterion_weight*criterion((keypoints_3d_pred  - keypoints_3d_gt)*scale_keypoints_3d,
                                                            keypoints_3d_binary_validity_gt)

                    total_loss += loss
                    metric_dict[f'{config.opt.criterion}'].append(loss.item())

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
                if use_style_pose_vce_loss:
                    volumetric_ce_criterion = VolumetricCELoss()
                    loss = volumetric_ce_criterion(coord_volumes_pred, 
                                                    volumes_pred, 
                                                    keypoints_3d_gt, 
                                                    keypoints_3d_binary_validity_gt)
                    metric_dict['volumetric_ce_loss'].append(loss.item())

                    total_loss += style_vce_weight * loss

                if config.model.pelvis_type !='gt':
                    base_joint = 6
                    pelvis_loss_weight = config.opt.pelvis_loss_weight
                    pelvis_loss = criterion((base_points_pred.unsqueeze(1) - keypoints_3d_gt[:,base_joint:base_joint+1])*scale_keypoints_3d,
                                                keypoints_3d_binary_validity_gt[:,base_joint:base_joint+1])
                    pelvis_loss = pelvis_loss_weight*pelvis_loss
                    total_loss += pelvis_loss
                    metric_dict['pelvis_loss_weighted'].append(pelvis_loss.item())

                ############
                # BACKWARD #   
                ############
                if is_train:
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

                # plot visualization
                if singleview_dataset:
                    keypoints_3d_gt = op.root_centering(keypoints_3d_gt, config.kind, inverse=True)
                    keypoints_3d_pred = op.root_centering(keypoints_3d_pred, config.kind, inverse=True)

                # save answers for evalulation
                if not is_train:
                    results['keypoints_3d'].append(keypoints_3d_pred.detach().cpu().numpy())
                    results['indexes'].append(batch['indexes'])
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # dump to tensorboard per-iter loss/metric stats
                if is_train and MAKE_EXPERIMENT_DIR:
                    try:
                        for title, value in metric_dict.items():
                            writer.add_scalar(f"{name}/{title}", value[-1], n_iters_total)

                        # dump to tensorboard per-iter time stats
                        writer.add_scalar(f"{name}/batch_time", batch_time.avg, n_iters_total)
                        writer.add_scalar(f"{name}/data_time", data_time.avg, n_iters_total)

                        # dump to tensorboard per-iter stats about sizes
                        writer.add_scalar(f"{name}/batch_size", batch_size, n_iters_total)
                        writer.add_scalar(f"{name}/n_views", dt, n_iters_total)

                    except Exception as e:
                        print ('Exception:', str(e), 'Failed to save writer')

                n_iters_total += 1
                            

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
                try:
                    checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    # dump results
                    with open(os.path.join(checkpoint_dir, "results.pkl"), 'wb') as fout:
                        pickle.dump(results, fout)
                except Exception as e:
                    print ('Exception:', str(e), 'Failed to save writer')        
                            

        # dump to tensorboard per-epoch stats
        if MAKE_EXPERIMENT_DIR:
            try:
                for title, value in metric_dict.items():
                    writer.add_scalar(f"{name}/{title}_epoch", np.mean(value), epoch)
            except Exception as e:
                print ('Exception:', str(e), 'Failed to save writer')             
                       

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

    ##########################
    # BACKWARD COMPATIBILITY #
    ##########################

    if not hasattr(config.model, 'pelvis_type'):
        if config.model.use_gt_pelvis:
            config.model.pelvis_type = 'gt'
        else:
            raise RuntimeError()  

    if not hasattr(config.model, 'pelvis_space_type'):
        config.model.pelvis_space_type = 'global'          

    ##########################
    is_distributed = config.distributed_train and init_distributed(args)
    print ('DISTRIBUTER TRAINING' if is_distributed else 'No Distributed training')
    master = True
    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0
    device = torch.device(args.local_rank) if is_distributed else torch.device(0)

    # options
    config.experiment_comment = args.experiment_comment
    save_model = config.opt.save_model if hasattr(config.opt, "save_model") else True
    
    model = {
        "s2s": S2SModel,
    }[config.model.name](config, device=device).to(device)

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

    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint)['model_state']
        model.load_state_dict(state_dict, strict=True)
        print("LOADED PRE-TRAINED MODEL!!!")        

    # optimizer
    opt = torch.optim.Adam(
        [{'params': model.backbone.parameters()},
         {'params': model.features_sequence_to_vector.parameters(),'lr': config.opt.features_sequence_to_vector_lr},
         {'params': model.encoder.parameters(),'lr': config.opt.encoder_lr},
         {'params': model.style_to_volumes.parameters(),'lr': config.opt.style_to_volumes_lr}],
        lr=config.opt.lr
    )

    load_optimizer = config.model.load_optimizer if hasattr(config.model, 'load_optimizer') else False
    if config.model.init_weights and load_optimizer:
        state_dict = torch.load(config.model.checkpoint)['opt_state']
        opt.load_state_dict(state_dict, strict=True)
        print("LOADED PRE-TRAINED OPTIMIZER!!!")

    use_scheduler = config.opt.use_scheduler if hasattr(config.opt, 'use_scheduler') else False    
    if use_scheduler:   
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=[lambda1, lambda2])

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=False)

    # experiment
    experiment_dir, writer = None, None
    if master and MAKE_EXPERIMENT_DIR: 
        experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)
        print ('EXPERIMENT IN LOGDIR:', args.logdir)    
        
    torch.cuda.empty_cache()
    # train loop
    n_iters_total_train, n_iters_total_val = 0, 0
    for epoch in range(config.opt.n_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        print ('Training...')    
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
                                        writer=writer)

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
                                      writer=writer)
        if use_scheduler:
            scheduler.step()
        # saving    
        if master and save_model:
            print (f'Saving model at {experiment_dir}...')
            save(experiment_dir, model, opt)
        print(f"epoch: {epoch}, iters: {n_iters_total_train}, done.")

    print("Done.")

if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)
