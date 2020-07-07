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
from mvn.models.volumetric_spade_debug import VolumetricSpadeDebug
from mvn.models.loss import KeypointsMSELoss, \
                            KeypointsMAELoss, \
                            KeypointsL2Loss, \
                            VolumetricCELoss 

from mvn.utils import img, multiview, op, vis, misc, cfg

from mvn.datasets.human36m import Human36MTemporalDataset, Human36MMultiViewDataset
from mvn.datasets import utils as dataset_utils
from mvn.utils.op import get_coord_volumes, unproject_heatmaps, integrate_tensor_3d_with_coordinates, softmax_volumes
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
from train import parse_args, setup_human36m_dataloaders, setup_dataloaders, setup_experiment, save, init_distributed

MAKE_EXPERIMENT_DIR = True

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

                (keypoints_3d_pred, 
                volumes_pred, 
                confidences_pred, 
                cuboids_pred, 
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
                    coord_volumes_pred = coord_volumes_pred - base_points_pred.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    keypoints_3d_gt = op.root_centering(keypoints_3d_gt, config.kind)
                    keypoints_3d_pred = op.root_centering(keypoints_3d_pred, config.kind)

                ##################
                # CALCULATE LOSS #   
                ##################
                # MSE\MAE loss
                total_loss = 0.0
                loss = criterion((keypoints_3d_pred  - keypoints_3d_gt)*scale_keypoints_3d,keypoints_3d_binary_validity_gt)
                total_loss += loss
                metric_dict[f'{config.opt.criterion}'].append(loss.item())

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
    # config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size
    config.experiment_comment = args.experiment_comment
    use_temporal_discriminator  = config.opt.use_temporal_discriminator if \
                                     hasattr(config.opt, "use_temporal_discriminator") else False  
    save_model = config.opt.save_model if hasattr(config.opt, "save_model") else True
    
    model = {
        "vol_spade_debug": VolumetricSpadeDebug
    }[config.model.name](config, device=device).to(device)

    # criterion
    criterion_class = {
        "MSE": KeypointsMSELoss,
        "MAE": KeypointsMAELoss
    }[config.opt.criterion]

    if config.opt.criterion == "MSESmooth": 
        criterion = criterion_class(config.opt.mse_smooth_threshold)
    else:
        criterion = criterion_class() 
 
    # optimizer
    opt = None
    if not args.eval:
        opt = torch.optim.Adam(
            [
            # {'params': model.backbone.parameters()},
            #  {'params': model.process_features.parameters(), 
            #             'lr': config.opt.process_features_lr if\
            #             hasattr(config.opt, "process_features_lr") else config.opt.lr},
             {'params': model.volume_net.parameters(), \
                        'lr': config.opt.volume_net_lr if \
                            hasattr(config.opt, "volume_net_lr") else config.opt.lr,
                        'betas': config.opt.volume_net_betas if \
                            hasattr(config.opt, "volume_net_betas") else (0.9, 0.999)}
            ], #+ \
            # ([{'params': model.features_sequence_to_vector.parameters(), \
            #             'lr': config.opt.features_sequence_to_vector_lr if \
            #             hasattr(config.opt, "features_sequence_to_vector_lr") else config.opt.lr}] if \
            #             hasattr(model, "features_sequence_to_vector") else []) 
            lr=config.opt.lr)

    # use_temporal_discriminator
    opt_discr, discriminator = None, None

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=is_distributed)

    # experiment
    experiment_dir, writer = None, None
    if master and MAKE_EXPERIMENT_DIR: 
        experiment_dir, writer = setup_experiment(config, type(model).__name__, args=args, is_train=not args.eval)
        print ('EXPERIMENT IN LOGDIR:', args.logdir)    
        
    # multi-gpu
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device])

    torch.cuda.empty_cache()
    if not args.eval:
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
