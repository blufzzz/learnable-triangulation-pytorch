import os
import shutil
import argparse
import time
import json
from datetime import datetime
from collections import defaultdict
from itertools import islice
import pickle
import numpy as np

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

from mvn.models.triangulation import VolumetricTriangulationNet
from mvn.models.decomposition import VolumetricDecompositionNet
from mvn.models.image2lixel import I2LModel

from mvn.models.loss import KeypointsMSELoss, \
                            KeypointsMSESmoothLoss, \
                            KeypointsMAELoss, \
                            KeypointsL2Loss, \
                            VolumetricCELoss

from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.datasets.human36m import Human36MMultiViewDataset, Human36MTemporalDataset
from mvn.datasets import utils as dataset_utils
from mvn.utils.op import get_coord_volumes, unproject_heatmaps, integrate_tensor_3d_with_coordinates, softmax_volumes

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
    only_keypoints=config.dataset.only_keypoints if hasattr(config.dataset, 'only_keypoints') else False

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
            only_keypoints = only_keypoints,
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
                                                     only_keypoints=only_keypoints,
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
        only_keypoints = only_keypoints,
        custom_iterator=config.dataset.custom_iterator if hasattr(config.dataset, "custom_iterator") else None
        )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                 min_n_views=config.dataset.val.min_n_views,
                                                 only_keypoints=only_keypoints,
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


def setup_experiment(config, model_name, args, is_train=True, existed_path=None):
    
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


def save(experiment_dir, model, opt, epoch):

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints") # , "{:04}".format(epoch)
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
              writer=None):
    
    name = "train" if is_train else "val"
    model_type = config.model.name
    silence = config.opt.silence if hasattr(config.opt, 'silence') else False

    singleview_dataset = config.dataset.singleview if hasattr(config.dataset, 'singleview') else False
    scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0    
    transfer_cmu_to_human36m = config.transfer_cmu_to_human36m if hasattr(config, "transfer_cmu_to_human36m") else False
    is_baseline = config.model.name == 'vol' 
    use_precalculated_basis = config.model.use_precalculated_basis if hasattr(config, "transfer_cmu_to_human36m") else True
    decomposition_type = config.model.decomposition_type if hasattr(config.model, "decomposition_type") else None
    
    ##################
    # LOSSES WEIGTHS #
    ##################
    if (not use_precalculated_basis) and (not is_baseline):
        coefficients_weight = config.opt.coefficients_weight
        basis_weight = config.opt.basis_weight

    use_coefs_norm_weight = config.opt.use_coefs_norm_weight if hasattr(config.opt, 'use_coefs_norm_weight') else False
    coefs_norm_weight = config.opt.coefs_norm_weight if hasattr(config.opt, 'coefs_norm_weight') else None
    
    use_volumetric_ce_loss = config.opt.use_volumetric_ce_loss
    vce_weight = config.opt.volumetric_ce_loss_weight

    criterion_weight = config.opt.criterion_weight

    use_bone_length_term = config.opt.use_bone_length_term if hasattr(config.opt, 'use_bone_length_term') else False
    bone_length_weight = config.opt.bone_length_weight if hasattr(config.opt, 'bone_length_weight') else None
    
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

                torch.cuda.empty_cache()
    
                (keypoints_3d_pred, 
                volumes_pred, 
                coefficients, # T
                basis, # [U1,...,Un]
                coord_volumes_pred,
                base_points_pred) = model(images_batch, batch)       

                batch_size, dt = images_batch.shape[:2] # dt == 1
                assert dt == 1
                keypoints_3d_binary_validity_gt = (keypoints_3d_validity_gt > 0.0).type(torch.float32)
                keypoints_shape = keypoints_3d_gt.shape[-2:]

                ################
                # MODEL OUTPUT #   
                ################
                if coord_volumes_pred is not None:
                    coord_volumes_pred = coord_volumes_pred - base_points_pred.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                keypoints_3d_gt = op.root_centering(keypoints_3d_gt, config.kind)
                keypoints_3d_pred = op.root_centering(keypoints_3d_pred, config.kind)

                ##################
                # CALCULATE LOSS #   
                ##################
                # MSE\MAE loss
                total_loss = 0.0
                loss = criterion((keypoints_3d_pred  - keypoints_3d_gt)*scale_keypoints_3d, keypoints_3d_binary_validity_gt)*criterion_weight
                total_loss += loss
                metric_dict[f'{config.opt.criterion}'].append(loss.item())

                # decomposition loss
                if (not use_precalculated_basis) and (not is_baseline):
                    heatmaps_3d = make_3d_heatmap(coord_volumes_pred, keypoints_3d_gt)

                    coefficients_gt, basis_gt = op.decompose(heatmaps_3d, decomposition_type)
                    coefficients_loss = coefficients_weight*torch.norm(coefficients_gt - coefficients)/coefficients_gt.numel()
                    set_trace()
                    
                    basis_loss = 0
                    for i in range(len(basis_gt)):
                        basis_loss += basis_weight*torch.norm(basis_gt[i] - basis[i])/basis[i].numel()
                    set_trace()

                    total_loss += basis_loss
                    total_loss += coefficients_loss

                    metric_dict['coefficients_loss'].append(coefficients_loss.item())
                    metric_dict['basis_loss'].append(basis_loss.item())

                # norm weight 
                if use_coefs_norm_weight:
                    # 7.4747086 - norm of the core tensor from Tucker decomposition of the PCA 3d-heatmaps
                    norm_loss = coefs_norm_weight*torch.abs(torch.norm(coefficients) - 7.4747086)
                    total_loss += norm_loss
                    metric_dict['norm_loss'].append(norm_loss.item())


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
                if use_volumetric_ce_loss:
                    volumetric_ce_criterion = VolumetricCELoss()
                    loss = volumetric_ce_criterion(coord_volumes_pred, 
                                                    volumes_pred, 
                                                    keypoints_3d_gt, 
                                                    keypoints_3d_binary_validity_gt)
                    metric_dict['volumetric_ce_loss'].append(loss.item())
                    total_loss += vce_weight * loss




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


                ###########
                # METRICS #   
                ###########
                metric_dict['total_loss'].append(total_loss.item())

                l2 = KeypointsL2Loss()(keypoints_3d_pred * scale_keypoints_3d, \
                                       keypoints_3d_gt * scale_keypoints_3d, \
                                       keypoints_3d_binary_validity_gt)

                metric_dict['l2'].append(l2.item())

                # base point l2
                if base_points_pred is not None and not config.model.pelvis_type =='gt':
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
                    results['keypoints_3d'].append(keypoints_3d_pred.detach().cpu().numpy())
                    results['indexes'].append(batch['indexes'])
                
                ###################
                # SAVE STATISTICS #   
                ###################      
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
                            
    ##############
    # EVALUATION #
    ##############
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

    is_distributed = config.distributed_train and init_distributed(args)
    print ('DISTRIBUTER TRAINING' if is_distributed else 'No Distributed training')
    master = True

    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0

    device = torch.device(args.local_rank) if is_distributed else torch.device(0)

    # options
    # config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size
    config.experiment_comment = args.experiment_comment
    save_model = config.opt.save_model if hasattr(config.opt, "save_model") else True
    
    model = {
        "vol": VolumetricTriangulationNet,
        "vol_decomposition": VolumetricDecompositionNet,
        'i2l': I2LModel,
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
        rebuild_dict = config.model.rebuild_dict if hasattr(config.model, 'rebuild_dict') else False
        weights_path  = os.path.join(config.model.checkpoint, 'checkpoints/weights.pth')
        state_dict = torch.load(weights_path)['model_state']
        if rebuild_dict:
            model_state_dict = model.state_dict() 
            for k,v in model_state_dict.items():
                if k in state_dict.keys():
                    model_state_dict[k] = v
                else:
                    assert 'adaptive_norm' or 'group_norm' in k
            state_dict = model_state_dict
            del model_state_dict 
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        torch.cuda.empty_cache()
        print("LOADED PRE-TRAINED MODEL!!!")        

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
                ], lr=config.opt.lr
            )  

        elif config.model.name == 'i2l':
            opt = torch.optim.Adam(
                [{'params': model.backbone.parameters()},
                 {'params': model.pose_net.parameters() ,'lr':config.opt.pose_net_lr}
                ] +\
                ([{'params': model.process_features.parameters(), \
                            'lr': config.opt.process_features_lr if \
                            hasattr(config.opt, "process_features_lr") else config.opt.lr}] \
                            if hasattr(model, 'process_features') else []) + \
                ([{'params': model.volume_net.parameters(), \
                            'lr': config.opt.volume_net_lr if \
                            hasattr(config.opt, "volume_net_lr") else config.opt.lr}] if
                            hasattr(model, 'volume_net') else []),
                lr=config.opt.lr
            )                        
        elif config.model.name == "vol_decomposition":
            opt = torch.optim.Adam(
                [{'params': model.backbone.parameters()},
                 {'params': model.process_features.parameters(), \
                            'lr': config.opt.process_features_lr if \
                            hasattr(config.opt, "process_features_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(), \
                            'lr': config.opt.volume_net_lr if \
                            hasattr(config.opt, "volume_net_lr") else config.opt.lr}
                ] +\
                ([{'params': model.basis.parameters(), \
                 'lr': config.opt.basis_lr if \
                 hasattr(config.opt, "basis_lr") else config.opt.lr}] if model.basis_source == 'optimized' else [])
                +\
                ([{'params': model.basis_net.parameters(), \
                 'lr': config.opt.basis_net_lr if \
                 hasattr(config.opt, "basis_net_lr") else config.opt.lr}] if hasattr(model, 'basis_net') else []),
                lr=config.opt.lr
            )                        
        else:
            raise RuntimeError('Unknown config.model.name')

        print('N_OPT_GROUPS:', len(opt.param_groups))

    CONTINUE_TRAINING = config.model.continue_training if hasattr(config.model, 'continue_training') else False
    if config.model.init_weights and CONTINUE_TRAINING:
        state_dict = torch.load(weights_path)['opt_state']
        opt.load_state_dict(state_dict)
        del state_dict
        print("LOADED PRE-TRAINED OPTIMIZER!!!")

    use_scheduler = config.opt.use_scheduler if hasattr(config.opt, 'use_scheduler') else False    
    if use_scheduler:   
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=[lambda1, lambda2])

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=is_distributed)

    # experiment
    experiment_dir, writer = None, None
    if master and MAKE_EXPERIMENT_DIR: 
        experiment_dir, writer = setup_experiment(config, 
                                                    type(model).__name__, 
                                                    args=args, 
                                                    is_train=not args.eval,
                                                    existed_path = config.model.checkpoint if CONTINUE_TRAINING else None)
        print ('EXPERIMENT IN LOGDIR:', args.logdir)    
        
    # multi-gpu
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device])

    torch.cuda.empty_cache()

    start_epoch, start_iter_train = 0, 0
    if CONTINUE_TRAINING:
        start_epoch, start_iter_train = misc.get_epoch_iter(config.model.checkpoint)

    print('STARTING EPOCH: {0}, STARTING ITER: {1}'.format(start_epoch, start_iter_train))

    if not args.eval:
        # train loop
        n_iters_total_train, n_iters_total_val = start_iter_train, 0
        for epoch in range(start_epoch, config.opt.n_epochs):
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
                save(experiment_dir, model, opt, epoch)
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
