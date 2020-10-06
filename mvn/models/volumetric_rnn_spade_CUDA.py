import numpy as np
import pickle
import random
from collections import defaultdict
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers
import os

from mvn.models import pose_resnet, pose_hrnet
from time import time
from mvn.utils.op import get_coord_volumes, unproject_heatmaps, integrate_tensor_3d_with_coordinates, make_3d_heatmap
from mvn.utils.multiview import update_camera
from mvn.utils.misc import get_capacity, description
from mvn.utils import volumetric, cfg
from mvn.models.v2v import V2VModel
from mvn.models.rnn import KeypointsRNNCell



from IPython.core.debugger import set_trace
STYLE_VECTOR_CONST = None


class VolumetricRNNSpade(jit.ScriptModule):

    __constants__ = [ 'kind', 
    'num_joints',
    'keypoints_per_frame',
    'dt', 
    'pivot_type', 
    'volume_softmax', 
    'volume_multiplier', 'volume_size', 'volume_aggregation_method',
    'cuboid_side', 'cuboid_multiplier', 'rotation',
    'pelvis_type', 'temporal_condition_type', 'v2v_type', 'v2v_normalization_type', 'gt_initialization', 'include_pivot', 
    'temporal_condition_type',
    'volume_features_dim',
    'style_vector_dim',
    'use_rnn',
    'heatmap_from_keypoints',
    'v2v_type',
    'rnn_layers',
    'rnn_bidirectional',
    'rnn_hidden_dim'
    ]

    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.kind = config.model.kind
        self.num_joints = config.model.backbone.num_joints
        self.keypoints_per_frame = config.dataset.keypoints_per_frame if \
                                         hasattr(config.dataset, 'keypoints_per_frame') else False

        # temporal 
        self.dt  = config.dataset.dt
        self.pivot_type = config.dataset.pivot_type

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size
        self.volume_aggregation_method = config.model.volume_aggregation_method

        # cuboid
        self.cuboid_side = config.model.cuboid_side
        self.cuboid_multiplier = config.model.cuboid_multiplier
        self.rotation = config.model.rotation

        # pelvis
        self.pelvis_type = config.model.pelvis_type if hasattr(config.model, 'pelvis_type') else 'gt'
        self.temporal_condition_type = config.model.temporal_condition_type
        self.v2v_type = config.model.v2v_type
        self.v2v_normalization_type = config.model.v2v_normalization_type
        assert self.v2v_normalization_type in ['group_norm','batch_norm']
        self.gt_initialization = config.model.gt_initialization if hasattr(config.model, 'gt_initialization') else False
        self.include_pivot = True

        assert self.temporal_condition_type == 'spade'

        # modules dimensions
        self.volume_features_dim = config.model.volume_features_dim
        self.style_vector_dim = self.num_joints
            
        ############
        # BACKBONE #   
        ############
        backbone = pose_resnet.get_pose_net(config.model.backbone,
                                                 device=device,
                                                 strict=True).to(device)
        self.backbone = torch.jit.trace(backbone,
                                        torch.randn(1,3,384,384).to(device)) # .to(device)

        print ('Only {} backbone is used...'.format(config.model.backbone.name))
        
        #######
        # RNN #
        #######
        self.heatmap_from_keypoints = config.model.heatmap_from_keypoints if hasattr(config.model, 'heatmap_from_keypoints') else True
        self.use_rnn = config.model.use_rnn if hasattr(config.model, 'use_rnn') else False
        # if self.use_rnn:
        #     if hasattr(config.model.rnn, 'experiment_dir'):
        #         config_rnn = cfg.load_config(os.path.join(config.model.rnn.experiment_dir, 'config.yaml'))
        #     else:
        #         config_rnn = config.model.rnn

        #     hx = torch.randn(config_rnn.model.num_layers * (2 if config_rnn.model.bidirectional else 1), 
        #                     1, 
        #                     config_rnn.model.hidden_dim)#.to(device)
        #     cx = torch.randn(config_rnn.model.num_layers * (2 if config_rnn.model.bidirectional else 1),
        #                      1, 
        #                      config_rnn.model.hidden_dim)#.to(device)
        #     hidden = (hx, cx)

        #     self.rnn_layers = config_rnn.model.num_layers
        #     self.rnn_bidirectional = config_rnn.model.bidirectional
        #     self.rnn_hidden_dim = config_rnn.model.hidden_dim

        #     self.rnn_model = torch.jit.trace(KeypointsRNNCell(config_rnn.model.num_joints*3, 
        #                                                      config_rnn.model.num_joints*3, 
        #                                                      hidden_dim=config_rnn.model.hidden_dim, 
        #                                                      bidirectional=config_rnn.model.bidirectional,
        #                                                      num_layers=config_rnn.model.num_layers,
        #                                                      use_dropout=config_rnn.opt.use_dropout,
        #                                                      droupout_rate=config_rnn.opt.droupout_rate,
        #                                                      normalize_input=config_rnn.opt.normalize_input,
        #                                                      layer_norm=config_rnn.model.layer_norm),
        #                                     (torch.randn(1,1,51), hidden)) # .to(device)

        #######
        # V2V #
        #######
        assert self.v2v_type == 'conf'
        use_compound_norm = config.model.use_compound_norm if hasattr(config.model, 'use_compound_norm') else True
        self.volume_net = torch.jit.trace(V2VModel(self.volume_features_dim,
                                                   self.num_joints,
                                                   v2v_normalization_type=self.v2v_normalization_type,
                                                   config=config.model.v2v_configuration,
                                                   style_vector_dim=self.style_vector_dim,
                                                   params_evolution=False,
                                                   style_forward=False,
                                                   use_compound_norm=use_compound_norm,
                                                   temporal_condition_type=self.temporal_condition_type).to(device),
                                         (torch.randn(1,self.volume_features_dim,32,32,32).to(device),
                                         torch.randn(1,self.num_joints,32,32,32).to(device))) # .to(device)
    
        if self.volume_features_dim != 256:    
            self.process_features = torch.jit.trace(nn.Sequential(nn.Conv2d(256, self.volume_features_dim, 1)).to(device),
                                                     torch.randn(1,256,96,96).to(device)) # .to(device)

        description(self)

    @jit.script_method
    def get_coord_volumes(self, 
                        keypoints
                        ):
        
        device = keypoints.device
        bs_dt = keypoints.shape[:-2]

        sides = torch.tensor([self.cuboid_side, self.cuboid_side, self.cuboid_side], dtype=torch.float).to(device)

        # default base_points are the coordinate's origins
        base_points = torch.zeros((bs_dt + (3,)), dtype=torch.float).to(device)
        
        # get root (pelvis) from keypoints
        if keypoints.shape[-2] == 1: 
            base_points = keypoints.squeeze(-2)
        else:   
            if self.kind == "coco":
                base_points = (keypoints[:,:, 11, :3] + keypoints[:,:,12, :3]) / 2
            elif self.kind == "mpii":
                base_points = keypoints[:,:, 6, :3] 

        position = base_points - sides / 2

        # build cuboids
        cuboids = None

        # build coord volume
        xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device, dtype=torch.float),
                                       torch.arange(self.volume_size, device=device, dtype=torch.float),
                                       torch.arange(self.volume_size, device=device, dtype=torch.float))
        grid = torch.stack([xxx, yyy, zzz], dim=-1) #.type(torch.float) 
        grid = grid.view((-1, 3)) # L,3
        grid = grid.view([1]*len(bs_dt) + grid.shape).repeat(bs_dt + [1]*len(grid.shape)) # (bs,dt,L,3)

        grid_coord = torch.zeros_like(grid)
        grid_coord[:,:,:, 0] = position[:,:, 0].unsqueeze(-1) + (sides[0] / (self.volume_size - 1)) * grid[:,:,:, 0]
        grid_coord[:,:,:, 1] = position[:,:, 1].unsqueeze(-1) + (sides[1] / (self.volume_size - 1)) * grid[:,:,:, 1]
        grid_coord[:,:,:, 2] = position[:,:, 2].unsqueeze(-1) + (sides[2] / (self.volume_size - 1)) * grid[:,:,:, 2]

        grid_coord = grid_coord.view(bs_dt + (self.volume_size, self.volume_size, self.volume_size, 3))
    
        return grid_coord, cuboids, base_points

    @jit.script_method
    def unproject_heatmaps(self,
                            heatmaps,
                            proj_matricies, 
                            coord_volumes):

        device = heatmaps.device
        
        batch_size = heatmaps.shape[0]
        n_views =  heatmaps.shape[1]
        n_joints = heatmaps.shape[2]
        heatmap_shape = heatmaps.shape[3:]
        
        volume_shape = coord_volumes.shape[1:4]

        volume_batch = [] 

        # TODO: speed up this this loop
        for batch_i in range(batch_size):
            coord_volume = coord_volumes[batch_i] 
            grid_coord = coord_volume.reshape((-1, 3))

            volume_batch_to_aggregate = torch.zeros((n_views, n_joints,) + volume_shape, device=device)

            for view_i in range(n_views):
                heatmap = heatmaps[batch_i, view_i]
                heatmap = heatmap.unsqueeze(0)

                grid_coord_proj = torch.cat([grid_coord,
                                            torch.ones((grid_coord.shape[0], 1), 
                                            dtype=grid_coord.dtype, 
                                            device=device)], dim=1) @ proj_matricies[batch_i, view_i].t()


                invalid_mask = grid_coord_proj[:, 2] <= torch.tensor(0.0).to(device)  # depth must be larger than 0.0
                zeros_mask = grid_coord_proj[:, 2] == torch.tensor(0.0).to(device)
                zeros_mask = zeros_mask.flatten()

                grid_coord_proj[zeros_mask, 2] = torch.tensor(1.0).to(device)  # not to divide by zero
                # convert back to euclidean
                grid_coord_proj = (grid_coord_proj.transpose(1, 0)[:-1] / grid_coord_proj.transpose(1, 0)[-1]).transpose(1, 0)

                # transform to [-1.0, 1.0] range
                grid_coord_proj_transformed = torch.zeros_like(grid_coord_proj)
                grid_coord_proj_transformed[:, 0] = torch.tensor(2.0).to(device) * (grid_coord_proj[:, 0] / heatmap_shape[0] - 0.5)
                grid_coord_proj_transformed[:, 1] = torch.tensor(2.0).to(device) * (grid_coord_proj[:, 1] / heatmap_shape[1] - 0.5)
                grid_coord_proj = grid_coord_proj_transformed # [N,2]
                # prepare to F.grid_sample
                grid_coord_proj = grid_coord_proj.unsqueeze(1).unsqueeze(0)
                current_volume = F.grid_sample(heatmap, grid_coord_proj)

                # zero out non-valid points
                current_volume = current_volume.view(n_joints, -1)
                current_volume[:, invalid_mask] = torch.tensor(0.0).to(device)

                # reshape back to volume
                current_volume = current_volume.view((n_joints,) + volume_shape)

                # collect
                volume_batch_to_aggregate[view_i] = current_volume

            volume_batch.append(volume_batch_to_aggregate)
        volume_batch = torch.cat(volume_batch, 0)

        return volume_batch

    @jit.script_method
    def integrate_tensor_3d_with_coordinates(self, volumes, coord_volumes):
        batch_size, n_volumes, x_size, y_size, z_size = volumes.shape

        volumes = volumes.reshape((batch_size, n_volumes, -1))
        volumes = nn.functional.softmax(volumes, dim=2)

        volumes = volumes.reshape((batch_size, n_volumes, x_size, y_size, z_size))
        coordinates = torch.einsum("bnxyz, bxyzc -> bnc", volumes, coord_volumes)

        return coordinates, volumes

    @jit.script_method
    def make_3d_heatmap(self, coord_volumes, tri_keypoints_3d):
        '''
        use_topk - non-differentiable way
        '''
        coord_volume_unsq = coord_volumes.unsqueeze(1)
        keypoints_gt_i_unsq = tri_keypoints_3d.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        dists = torch.sqrt(((coord_volume_unsq - keypoints_gt_i_unsq) ** 2).sum(-1))
        
        H = 1./(torch.pow(dists,2) + 1e-5)

        style_vector_volumes = F.softmax(H.view(H.shape[:-3] + (-1,)),dim=-1).view(dists.shape)
        return style_vector_volumes


    @jit.script_method
    def forward(self, 
                images_batch, 
                keypoints_3d_gt,
                proj_matricies_batch): 

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]
        fictive_view = 1
        assert self.dt == dt

        ######################
        # FEATURE ECTRACTION #   
        ######################
        features = self.backbone(images_batch.view((-1, 3,) + image_shape))

        if self.volume_features_dim != 256:
            features = self.process_features(features)
        features_shape = features.shape[-2:]
        features_channels = features.shape[1]
        features = features.view((batch_size, -1, features_channels,)  + features_shape)

        ##########
        # PELVIS #   
        ##########            
        # assert self.pelvis_type == 'gt':

        keypoints_3d_gt = keypoints_3d_gt[:,:,:,:3] # bs,dt,
        keypoints_shape = keypoints_3d_gt.shape[-2:]
        #####################
        # VOLUMES CREATING  #   
        #####################
        coord_volumes, _, base_points = self.get_coord_volumes(keypoints_3d_gt)
        
        ###############
        # V2V FORWARD #   
        ###############
        coord_volumes = coord_volumes.view(-1, self.volume_size, self.volume_size, self.volume_size,3)
        proj_matricies_batch = proj_matricies_batch.view((-1, fictive_view,) + proj_matricies_batch.shape[2:])
        features = features.view((-1, fictive_view, features_channels,) + features_shape)

        # lift each feature-map to distinct volume and aggregate 
        unproj_features = self.unproject_heatmaps(features,  
                                                 proj_matricies_batch, 
                                                 coord_volumes
                                                 )

        unproj_features = unproj_features.view((batch_size, -1,) + unproj_features.shape[-4:])
        coord_volumes = coord_volumes.view((batch_size, -1,) + coord_volumes.shape[-4:])
        
        vol_keypoints_3d_list = []
        volumes_list = []

        # if self.use_rnn:
        #     rnn_keypoints_list = []
        #     # FIX: change for layer_norm, see `train_rnn.py`
        #     hx = torch.randn(self.rnn_model.num_layers * (2 if self.rnn_model.bidirectional else 1), 
        #                     batch_size, 
        #                     self.rnn_model.hidden_dim).to(device)
        #     cx = torch.randn(self.rnn_model.num_layers * (2 if self.rnn_model.bidirectional else 1),
        #                      batch_size, 
        #                      self.rnn_model.hidden_dim).to(device)
        #     hidden = (hx, cx)
        style_vector_volumes = torch.zeros(batch_size, 
                                          self.num_joints,
                                          self.volume_size,
                                          self.volume_size,
                                          self.volume_size)
        for t in range(dt): 
            features_volumes_t = unproj_features[:,t]
            coord_volumes_t = coord_volumes[:,t]

            # torch.cuda.empty_cache()
            # get initial observation w\o spade
            # if t==0:
                # if self.gt_initialization:
                #     style_vector_volumes = self.make_3d_heatmap(coord_volumes[:,0], keypoints_3d_gt[:,0])
                #     volumes = self.volume_net(features_volumes_t, style_vector_volumes)
                # else:
                # volumes = self.volume_net(features_volumes_t, style_vector_volumes)
            volumes = self.volume_net(features_volumes_t, style_vector_volumes)
            # volumes, style_vector_volumes
            vol_keypoints_3d, volumes = self.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                                coord_volumes_t)
            # vol_keypoints_3d, volumes
            # if self.use_rnn and t < (dt-1):
            #     keypoints_3d_prev = tri_keypoints_3d[:,:t+1].contiguous()
            #     keypoints_3d_prev = keypoints_3d_prev.view(*keypoints_3d_prev.shape[:2], -1)
            #     mean = keypoints_3d_prev.mean(1).unsqueeze(1) # FIX: ZEROS at first iteration
            #     std = keypoints_3d_prev.std(1).unsqueeze(1) if t > 0 else 1
            #     rnn_keypoints, hidden = self.rnn_model(vol_keypoints_3d.view(batch_size, 1, -1), 
            #                                            hidden,
            #                                            mean=mean,
            #                                            std=std)
            #     rnn_keypoints = rnn_keypoints.view(batch_size, *keypoints_shape)
            #     style_vector_volumes = self.make_3d_heatmap(coord_volumes[:,t+1], rnn_keypoints)
            #     rnn_keypoints_list.append(rnn_keypoints)
            #     # rnn_keypoints
            # else:
            if self.heatmap_from_keypoints:
                style_vector_volumes = self.make_3d_heatmap(coord_volumes_t, vol_keypoints_3d)
            else:
                style_vector_volumes = volumes
                
            volumes_list.append(volumes)
            vol_keypoints_3d_list.append(vol_keypoints_3d)

        vol_keypoints_3d = torch.stack(vol_keypoints_3d_list, 1)
        volumes = torch.stack(volumes_list, 1)
        rnn_keypoints_3d = None
        # if self.use_rnn:
        #     rnn_keypoints_3d = torch.stack(rnn_keypoints_list, 1)

        return (vol_keypoints_3d,
                rnn_keypoints_3d, # features
                volumes,
                coord_volumes,
                base_points
                )












