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
from mvn.models.rnn import KeypointsRNNCell

from mvn.models.rnn import LSTMState
from mvn.models.temporal import Seq2VecCNN
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
    silence = config.opt.silence if hasattr(config.opt, 'silence') else False

    singleview_dataset = config.dataset.singleview if hasattr(config.dataset, 'singleview') else False
    pivot_type = config.dataset.pivot_type
    pivot_index =  {'first':config.dataset.dt-1,
                    'intermediate':config.dataset.dt//2}[pivot_type]
    
    scale_keypoints_3d = config.opt.scale_keypoints_3d 
    use_temporal_discriminator = config.opt.use_temporal_discriminator
    use_smoothness_criterion = config.opt.use_smoothness_criterion
    keypoints_per_frame = config.dataset.keypoints_per_frame
    only_keypoints=config.dataset.only_keypoints
    assert keypoints_per_frame

    use_time_weighted_loss = config.opt.use_time_weighted_loss

    if use_temporal_discriminator:
        assert (discriminator is not None) and (opt_discr is not None)
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

        iterator = enumerate(dataloader) # breaks here!
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
                keypoints_3d_gt, keypoints_3d_validity_gt = dataset_utils.prepare_batch(batch, device, only_keypoints=only_keypoints)

                heatmaps_pred, keypoints_2d_pred, cuboids_pred, base_points_pred = None, None, None, None
                torch.cuda.empty_cache()

                batch_size, dt = keypoints_3d_gt.shape[:2]
                keypoints_3d_binary_validity_gt = (keypoints_3d_validity_gt > 0.0).type(torch.float32)
                keypoints_shape = keypoints_3d_gt.shape[-2:]
                validity_shape = keypoints_3d_binary_validity_gt.shape[-2:]

                if model.layer_norm:
                    hidden = [[LSTMState(torch.randn(batch_size, model.hidden_dim).to(device),
                                        torch.randn(batch_size, model.hidden_dim).to(device))
                             for _ in range(2 if model.bidirectional else 1)] for rnn_layer in range(model.num_layers)]
                else:
                    hx = torch.randn(model.num_layers * (2 if model.bidirectional else 1), batch_size, model.hidden_dim).to(device)
                    cx = torch.randn(model.num_layers * (2 if model.bidirectional else 1), batch_size, model.hidden_dim).to(device)
                    hidden = (hx,cx)

                keypoints_3d_pred, _ = model(keypoints_3d_gt, hidden)       

                ################
                # MODEL OUTPUT #   
                ################
                # flattening
                keypoints_3d_binary_validity_gt = keypoints_3d_binary_validity_gt[:,1:].contiguous().view(-1, *validity_shape)
                keypoints_3d_gt = keypoints_3d_gt[:,1:].contiguous().view(-1, *keypoints_shape)
                keypoints_3d_pred = keypoints_3d_pred[:,:-1].contiguous().view(-1, *keypoints_shape)

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

                ########################
                # SMOOTHNESS CRITERION #
                ########################
                keypoints_3d_pred_seq = keypoints_3d_pred.view(batch_size, -1, *keypoints_shape)
                smoothness_loss = torch.abs(keypoints_3d_pred_seq[:-1] - keypoints_3d_pred_seq[1:]).sum(-1).mean()
                metric_dict['smoothness_MAE'].append(smoothness_loss.item())
                if use_smoothness_criterion:
                    weight = config.opt.smoothness_criterion_weight
                    total_loss += weight * smoothness_loss

                #############################        
                # TEMPORAL ADVERSARIAL LOSS #
                #############################  
                if use_temporal_discriminator: 
                    keypoints_3d_pred_seq = keypoints_3d_pred.view(batch_size, dt-1, -1)
                    keypoints_3d_gt_seq = keypoints_3d_gt.view(batch_size, dt-1, -1)

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

                # plot visualization
                if singleview_dataset:
                    keypoints_3d_gt = op.root_centering(keypoints_3d_gt, config.kind, inverse=True)
                    keypoints_3d_pred = op.root_centering(keypoints_3d_pred, config.kind, inverse=True)

                # save answers for evalulation
                if not is_train:
                    if keypoints_per_frame:
                        keypoints_3d_pred = keypoints_3d_pred.view(batch_size, -1, *keypoints_3d_pred.shape[-2:])[:,-1]
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
    model = KeypointsRNNCell(config.model.num_joints*3, 
                             config.model.num_joints*3, 
                             hidden_dim=config.model.hidden_dim, 
                             bidirectional=config.model.bidirectional,
                             num_layers=config.model.num_layers,
                             use_dropout=config.opt.use_dropout,
                             droupout_rate=config.opt.droupout_rate,
                             normalize_input=config.opt.normalize_input,
                             layer_norm=config.model.layer_norm).to(device)
    # criterion
    criterion = {
        "MSE": KeypointsMSELoss,
        "MAE": KeypointsMAELoss
    }[config.opt.criterion]()

    # optimizer
    opt = None
    opt = torch.optim.Adam(model.parameters(), lr=config.opt.lr)
            
    # use_temporal_discriminator
    opt_discr, discriminator = None, None
    if use_temporal_discriminator:
        discriminator = Seq2VecCNN(config.model.num_joints*3,
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
        # saving    
        if master and save_model and MAKE_EXPERIMENT_DIR:
            print (f'Saving model at {experiment_dir}...')
            save(experiment_dir, model, opt, epoch, discriminator, opt_discr, use_temporal_discriminator)
        print(f"epoch: {epoch}, iters: {n_iters_total_train}, done.")

    print("Done.")

if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)
