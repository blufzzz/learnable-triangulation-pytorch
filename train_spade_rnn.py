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

from mvn.models.temporal import Seq2VecCNN
# from mvn.models.volumetric_rnn_spade_CUDA import VolumetricRNNSpade
from mvn.models.volumetric_rnn_spade import VolumetricRNNSpade
from mvn.utils.multiview import update_camera

from mvn.models.loss import KeypointsMSELoss, \
                            KeypointsMAELoss, \
                            KeypointsL2Loss, \
                            VolumetricCELoss,\
                            GAN_loss,\
                            LSGAN_loss

from mvn.utils import img, multiview, op, vis, misc, cfg

from mvn.datasets.human36m import Human36MTemporalDataset, Human36MMultiViewDataset
from mvn.datasets import utils as dataset_utils
from mvn.utils.op import get_coord_volumes, unproject_heatmaps, integrate_tensor_3d_with_coordinates, softmax_volumes
from train import parse_args, setup_human36m_dataloaders, setup_dataloaders, setup_experiment, save, init_distributed

from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt

MAKE_EXPERIMENT_DIR = False

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
    
    scale_keypoints_3d = config.opt.scale_keypoints_3d 
    use_temporal_discriminator = config.opt.use_temporal_discriminator if hasattr(config.opt, "use_temporal_discriminator") else False
    use_smoothness_criterion = config.opt.use_smoothness_criterion
    keypoints_per_frame = config.dataset.keypoints_per_frame
    assert keypoints_per_frame

    auxilary_criterion_weight = config.opt.auxilary_criterion_weight
    use_aux_VCE_loss = config.opt.use_aux_VCE_loss
    if use_aux_VCE_loss:
        auxilary_VCE_weight = config.opt.auxilary_VCE_weight

    use_time_weighted_loss = config.opt.use_time_weighted_loss
    use_rnn_loss = config.opt.use_rnn_loss
    if use_rnn_loss:
        rnn_loss_weight = config.opt.rnn_loss_weight

    if use_temporal_discriminator:
        assert (discriminator is not None) and (opt_discr is not None)
        adversarial_temporal_criterion = {'vanilla':GAN_loss(),
                                          'lsgan':LSGAN_loss()}[config.opt.adversarial_temporal_criterion]
        adversarial_temporal_loss_weight = config.opt.adversarial_temporal_loss_weight
        adversarial_generator_iters = config.opt.adversarial_generator_iters 
        train_generator_during_critic_iters = config.opt.train_generator_during_critic_iters                               

    use_jit = config.opt.use_jit if hasattr(config.opt, 'use_jit') else False

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
                batch_size, dt = images_batch.shape[:2]
                keypoints_3d_binary_validity_gt = (keypoints_3d_validity_gt > 0.0).type(torch.float32)
                keypoints_shape = keypoints_3d_gt.shape[-2:]

                if use_jit:
                    proj_matricies_batch = update_camera(batch, batch_size, (384,384), (96,96), dt, device)
                    (keypoints_3d_pred, 
                    rnn_keypoints_3d, 
                    volumes_pred, 
                    coord_volumes_pred,
                    base_points_pred) = model(images_batch, keypoints_3d_gt, proj_matricies_batch) 
                else:

                    (keypoints_3d_pred, 
                    rnn_keypoints_3d, 
                    volumes_pred, 
                    _, 
                    _, 
                    coord_volumes_pred,
                    base_points_pred) = model(images_batch, batch) 


                ################
                # MODEL OUTPUT #   
                ################
                # flattening
                keypoints_3d_gt = keypoints_3d_gt.view(-1, *keypoints_shape)
                keypoints_3d_pred = keypoints_3d_pred.view(-1, *keypoints_shape)

                coord_volumes_pred = coord_volumes_pred.view(batch_size*dt, *coord_volumes_pred.shape[-4:]) 
                base_points_pred = base_points_pred.view(batch_size*dt, *base_points_pred.shape[-1:]) 
                
                # centering
                coord_volumes_pred = coord_volumes_pred - base_points_pred.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                keypoints_3d_gt = op.root_centering(keypoints_3d_gt, config.kind)
                keypoints_3d_pred = op.root_centering(keypoints_3d_pred, config.kind)

                # unflattening
                coord_volumes_pred = coord_volumes_pred.view(batch_size, dt, *coord_volumes_pred.shape[-4:])
                volumes_pred = volumes_pred.view(batch_size, dt, *volumes_pred.shape[-4:])
                base_points_pred = base_points_pred.view(batch_size, dt, *base_points_pred.shape[-1:]) 
                
                keypoints_3d_gt = keypoints_3d_gt.view(batch_size, dt, *keypoints_shape)
                keypoints_3d_pred = keypoints_3d_pred.view(batch_size, dt, *keypoints_shape)

                # extracting auxilary
                auxilary_keypoints_3d_gt = keypoints_3d_gt[:,:pivot_index].contiguous()
                if use_rnn_loss:
                    keypoints_3d_gt_rnn = keypoints_3d_gt[:,1:].contiguous().view(-1, *keypoints_shape)
                    rnn_keypoints_3d = rnn_keypoints_3d.view(-1, *keypoints_shape)
                    rnn_keypoints_binary_validity = keypoints_3d_binary_validity_gt[:,1:].contiguous().view(-1, *keypoints_3d_binary_validity_gt.shape[-2:])
                    rnn_keypoints_3d = op.root_centering(rnn_keypoints_3d, config.kind)

                auxilary_keypoints_3d_pred =  keypoints_3d_pred[:,:pivot_index].contiguous()
                auxilary_keypoints_3d_binary_validity_gt = keypoints_3d_binary_validity_gt[:,:pivot_index].contiguous()

                aux_coord_volumes_pred = coord_volumes_pred[:,:pivot_index].contiguous()
                aux_volumes_pred = volumes_pred[:,:pivot_index].contiguous()

                # extracting pivot
                keypoints_3d_gt = keypoints_3d_gt[:,pivot_index]
                keypoints_3d_pred = keypoints_3d_pred[:,pivot_index]
                keypoints_3d_binary_validity_gt = keypoints_3d_binary_validity_gt[:,pivot_index]
                
                coord_volumes_pred = coord_volumes_pred[:,pivot_index]
                volumes_pred = volumes_pred[:,pivot_index]

                ##################
                # CALCULATE LOSS #   
                ##################
                # MSE\MAE loss
                total_loss = 0.0
                loss = criterion((keypoints_3d_pred  - keypoints_3d_gt)*scale_keypoints_3d,keypoints_3d_binary_validity_gt)
                total_loss += loss
                metric_dict[f'{config.opt.criterion}'].append(loss.item())

                # lstm temporal pose-style loss
                if use_time_weighted_loss:
                    future_keypoints_loss_weight = torch.stack([torch.exp(torch.arange((-dt)+1,0, 1, dtype=torch.float)) \
                                                                for i in range(batch_size)]).view(batch_size, -1,1,1).to(device)
                else:
                    future_keypoints_loss_weight = 1.

                aux_criterion_diff = (auxilary_keypoints_3d_gt - auxilary_keypoints_3d_pred) * scale_keypoints_3d * future_keypoints_loss_weight
                aux_criterion_diff = aux_criterion_diff.view(-1, *aux_criterion_diff.shape[-2:])
                validity = auxilary_keypoints_3d_binary_validity_gt.view(-1, *auxilary_keypoints_3d_binary_validity_gt.shape[-2:])
                auxilary_criterion = criterion(aux_criterion_diff, validity)
                
                auxilary_criterion_weighted = auxilary_criterion_weight * auxilary_criterion
                total_loss += auxilary_criterion_weighted
                metric_dict['style_pose_lstm_loss_weighted'].append(auxilary_criterion_weighted.item()) # name is legacy

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

                    # pass squeezed tensors
                    if use_aux_VCE_loss:
                        loss = volumetric_ce_criterion(aux_coord_volumes_pred.view(-1, *aux_coord_volumes_pred.shape[2:]), 
                                                       aux_volumes_pred.view(-1, *aux_volumes_pred.shape[2:]), 
                                                       auxilary_keypoints_3d_gt.view(-1, *auxilary_keypoints_3d_gt.shape[2:]), 
                                                       auxilary_keypoints_3d_binary_validity_gt.view(-1, *auxilary_keypoints_3d_binary_validity_gt.shape[2:]))

                        total_loss += weight * auxilary_VCE_weight * loss
                        metric_dict['aux_lstm_VCE_loss'].append(loss.item())    

                ############
                # RNN LOSS #
                ############
                if use_rnn_loss:
                    rnn_loss = criterion((rnn_keypoints_3d -keypoints_3d_gt_rnn)*scale_keypoints_3d,
                                          rnn_keypoints_binary_validity)
                    loss = rnn_loss * rnn_loss_weight
                    total_loss += loss
                    metric_dict['rnn_loss_weighted'].append(loss.item()) 

                #############################        
                # TEMPORAL ADVERSARIAL LOSS #
                #############################  
                # if use_temporal_discriminator: 
                #     # FIX SLICING!
                #     keypoints_3d_pred_seq = keypoints_3d_pred.view(batch_size, dt, -1)
                #     keypoints_3d_gt_seq = keypoints_3d_gt.view(batch_size, dt, -1)

                #     # discriminator step, no need gradient flow to generator
                #     discriminator_loss = adversarial_temporal_criterion(discriminator, 
                #                                                         keypoints_3d_pred_seq.clone().detach(), 
                #                                                         keypoints_3d_gt_seq.clone().detach(),
                #                                                         discriminator_loss=True)

                #     if is_train:
                #         opt_discr.zero_grad()
                #         discriminator_loss.backward()
                #         opt_discr.step()

                #     # generator step
                #     if is_train: 
                #         discriminator.zero_grad()
                #     generator_loss = adversarial_temporal_criterion(discriminator, 
                #                                                     keypoints_3d_pred_seq,
                #                                                     keypoints_3d_gt_seq,
                #                                                     discriminator_loss=False)
                    
                #     # add loss                    
                #     if iter_i%adversarial_generator_iters == 0:
                #         total_loss += generator_loss * adversarial_temporal_loss_weight

                #     metric_dict[f'{config.opt.adversarial_temporal_criterion}_generator_loss'].append(generator_loss.item())
                #     metric_dict[f'{config.opt.adversarial_temporal_criterion}_discriminator_loss'].append(discriminator_loss.item())

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
    assert args.eval is False
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
    use_temporal_discriminator  = config.opt.use_temporal_discriminator if \
                                     hasattr(config.opt, "use_temporal_discriminator") else False  
    save_model = config.opt.save_model if hasattr(config.opt, "save_model") else True
    use_rnn = config.model.use_rnn if hasattr(config.model, 'use_rnn') else False

    model = {
        "vol_rnn_spade": VolumetricRNNSpade
    }[config.model.name](config, device=device).to(device)

    # criterion 
    criterion = {
        "MSE": KeypointsMSELoss,
        "MAE": KeypointsMAELoss
    }[config.opt.criterion]()

    # load v2v\backbone weights
    if config.model.init_weights:
        rebuild_dict = config.model.rebuild_dict if hasattr(config.model, 'rebuild_dict') else False
        state_dict = torch.load(config.model.checkpoint)['model_state']
        if rebuild_dict:
            model_state_dict = model.state_dict() 
            for k,v in model_state_dict.items():
                if k in state_dict.keys():
                    model_state_dict[k] = v
                else:
                    print (f'{k} missed')
                    assert ('adaptive_norm' in k.split('.')) or ('group_norm' in k.split('.'))
            state_dict = model_state_dict
            del model_state_dict 
        model.load_state_dict(state_dict, strict=True)
        del state_dict 
        torch.cuda.empty_cache()
        print("LOADED PRE-TRAINED V2V MODEL!!!")

        # load rnn weights
        if use_rnn and hasattr(config.model.rnn, 'experiment_dir'):
            rnn_state_dict = torch.load(os.path.join(config.model.rnn.experiment_dir, 'checkpoints/weights.pth'))['model_state']
            model.rnn_model.load_state_dict(rnn_state_dict, strict=True)   
            del rnn_state_dict
            torch.cuda.empty_cache()
            print("LOADED PRE-TRAINED RNN MODEL!!!")

    # optimizer
    opt = None
    assert config.model.name == "vol_rnn_spade"
    opt = torch.optim.Adam(
        [{'params': model.backbone.parameters()},
         {'params': model.process_features.parameters(), \
                    'lr': config.opt.process_features_lr if \
                    hasattr(config.opt, "process_features_lr") else config.opt.lr},
         {'params': model.volume_net.parameters(), \
                    'lr': config.opt.volume_net_lr if \
                    hasattr(config.opt, "volume_net_lr") else config.opt.lr}] +\

        ([{'params':model.rnn_model.parameters(), \
                    'lr': config.opt.rnn_model_lr}] if \
                    hasattr(model,'rnn_model') else []),
        lr=config.opt.lr)
            
    load_optimizer = config.model.load_optimizer if hasattr(config.model, 'load_optimizer') else False
    if config.model.init_weights and load_optimizer:
        state_dict = torch.load(config.model.checkpoint)['opt_state']
        opt.load_state_dict(state_dict, strict=True)
        print("LOADED PRE-TRAINED OPTIMIZER!!!")

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

    use_scheduler = config.opt.use_scheduler if hasattr(config.opt, 'use_scheduler') else False    
    if use_scheduler:   
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=[lambda1, lambda2])

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=is_distributed)

    # experiment
    experiment_dir, writer = None, None
    if master and MAKE_EXPERIMENT_DIR: 
        experiment_dir, writer = setup_experiment(config, type(model).__name__, args=args, is_train=True)
        print ('EXPERIMENT IN LOGDIR:', args.logdir)    
        
    # multi-gpu
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device])

    torch.cuda.empty_cache()
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
        if use_scheduler:
            scheduler.step()
        # saving    
        if master and save_model:
            print (f'Saving model at {experiment_dir}...')
            save(experiment_dir, model, opt, epoch, discriminator, opt_discr, use_temporal_discriminator)
        print(f"epoch: {epoch}, iters: {n_iters_total_train}, done.")

    print("Done.")

if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)
