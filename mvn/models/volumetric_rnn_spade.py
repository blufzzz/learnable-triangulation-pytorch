import numpy as np
import pickle
import random
from collections import defaultdict
import sys
import torch
from torch import nn
import torch.nn.functional as F
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


class VolumetricRNNSpade(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.kind = config.model.kind
        self.num_joints = config.model.backbone.num_joints
        self.keypoints_per_frame = config.dataset.keypoints_per_frame if \
                                         hasattr(config.dataset, 'keypoints_per_frame') else False

        # temporal 
        self.dt  = config.dataset.dt
        self.pivot_type = config.dataset.pivot_type
        self.pivot_index =  {'first':self.dt-1,
                            'intermediate':self.dt//2}[self.pivot_type]
        self.aux_indexes = list(range(self.dt))
        self.aux_indexes.remove(self.pivot_index)

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
        self.backbone = pose_resnet.get_pose_net(config.model.backbone,
                                                 device=device,
                                                 strict=True)
        
        print ('Only {} backbone is used...'.format(config.model.backbone.name))
        
        #######
        # RNN #
        #######
        self.use_rnn = config.model.use_rnn if hasattr(config.model, 'use_rnn') else False
        if self.use_rnn:
            if hasattr(config.model.rnn, 'experiment_dir'):
                config_rnn = cfg.load_config(os.path.join(config.model.rnn.experiment_dir, 'config.yaml'))
            else:
                config_rnn = config.model.rnn

            self.rnn_model = KeypointsRNNCell(config_rnn.model.num_joints*3, 
                                             config_rnn.model.num_joints*3, 
                                             hidden_dim=config_rnn.model.hidden_dim, 
                                             bidirectional=config_rnn.model.bidirectional,
                                             num_layers=config_rnn.model.num_layers,
                                             use_dropout=config_rnn.opt.use_dropout,
                                             droupout_rate=config_rnn.opt.droupout_rate,
                                             normalize_input=config_rnn.opt.normalize_input,
                                             layer_norm=config_rnn.model.layer_norm).to(device)

        #######
        # V2V #
        #######
        if self.v2v_type == 'v1':
            self.volume_net = V2VModel_v1(self.volume_features_dim,
                                          self.num_joints,
                                          normalization_type=self.v2v_normalization_type,
                                          volume_size=self.volume_size,
                                          temporal_condition_type = self.temporal_condition_type,
                                          style_vector_dim = self.style_vector_dim)

        elif self.v2v_type == 'conf':
            use_compound_norm = config.model.use_compound_norm if hasattr(config.model, 'use_compound_norm') else True
            self.volume_net = V2VModel(self.volume_features_dim,
                                       self.num_joints,
                                       v2v_normalization_type=self.v2v_normalization_type,
                                       config=config.model.v2v_configuration,
                                       style_vector_dim=self.style_vector_dim,
                                       params_evolution=False,
                                       style_forward=False,
                                       use_compound_norm=use_compound_norm,
                                       temporal_condition_type=self.temporal_condition_type)
        
        if self.volume_features_dim != 256:    
            self.process_features = nn.Sequential(nn.Conv2d(256, self.volume_features_dim, 1))
        else:
            self.process_features = nn.Sequential()    

        description(self)


    def forward(self, 
                images_batch, 
                batch, 
                randomize_style=False, 
                const_style_vector=False,
                return_me_vector = False,
                debug=False,
                master=True):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]
        fictive_view = 1
        assert self.dt == dt

        ######################
        # FEATURE ECTRACTION #   
        ######################
        _, features, _, _, _ = self.backbone(images_batch.view(-1, 3, *image_shape))

        features = self.process_features(features)
        features_shape = features.shape[-2:]
        features_channels = features.shape[1]
        features = features.view(batch_size, -1, features_channels, *features_shape)

        proj_matricies_batch = update_camera(batch, batch_size, image_shape, features_shape, dt, device)

        ##########
        # PELVIS #   
        ##########            
        if self.pelvis_type == 'gt':
            tri_keypoints_3d = torch.from_numpy(np.array(batch['keypoints_3d'])).type(torch.float).to(device)
            if tri_keypoints_3d.dim() == 4:
                self.keypoints_for_each_frame = True
            elif tri_keypoints_3d.dim() == 3:
                self.keypoints_for_each_frame = False
            else:
                raise RuntimeError('Broken tri_keypoints_3d shape')     
        else:
            raise RuntimeError('In absence of precalculated pelvis or gt pelvis, self.use_volumetric_pelvis should be True') 

        tri_keypoints_3d = tri_keypoints_3d[...,:3]
        keypoints_shape = tri_keypoints_3d.shape[-2:]
        #####################
        # VOLUMES CREATING  #   
        #####################
        coord_volumes, _, base_points = get_coord_volumes(self.kind, 
                                                            self.training, 
                                                            self.rotation,
                                                            self.cuboid_side,
                                                            self.volume_size, 
                                                            device,
                                                            keypoints=tri_keypoints_3d
                                                            )
        ###############
        # V2V FORWARD #   
        ###############
        coord_volumes = coord_volumes.view(-1, self.volume_size, self.volume_size, self.volume_size,3)
        proj_matricies_batch = proj_matricies_batch.view(-1, fictive_view, *proj_matricies_batch.shape[2:])
        features = features.view(-1, fictive_view, features_channels, *features_shape)

        # lift each feature-map to distinct volume and aggregate 
        unproj_features = unproject_heatmaps(features,  
                                             proj_matricies_batch, 
                                             coord_volumes, 
                                             volume_aggregation_method=self.volume_aggregation_method,
                                             vol_confidences=None,
                                             fictive_views=None
                                             )
        unproj_features = unproj_features.view(batch_size, -1, *unproj_features.shape[-4:])
        coord_volumes = coord_volumes.view(batch_size, -1, *coord_volumes.shape[-4:])
        
        vol_keypoints_3d_list = []
        volumes_list = []
        rnn_keypoints_list = []

        # if self.gt_initialization:
        #     style_vector_volumes = make_3d_heatmap(coord_volumes[:,0], tri_keypoints_3d[:,0], use_topk=False)
        # else:   
        #     style_vector_volumes = torch.ones(batch_size, 
        #                                       self.style_vector_dim, 
        #                                       self.volume_size, 
        #                                       self.volume_size, 
        #                                       self.volume_size).to(device) / (self.volume_size**3)

        if self.use_rnn:
            # FIX: change for layer_norm, see `train_rnn.py`
            hx = torch.randn(self.rnn_model.num_layers * (2 if self.rnn_model.bidirectional else 1), 
                            batch_size, 
                            self.rnn_model.hidden_dim).to(device)
            cx = torch.randn(self.rnn_model.num_layers * (2 if self.rnn_model.bidirectional else 1),
                             batch_size, 
                             self.rnn_model.hidden_dim).to(device)
            hidden = (hx, cx)

        for t in range(dt): 
            features_volumes_t= unproj_features[:,t,...]
            coord_volumes_t = coord_volumes[:,t,...]

            torch.cuda.empty_cache()
            # get initial observation w\o spade
            if t==0:
                volumes = self.volume_net(features_volumes_t, params=None)
                # volumes
            else:         
                volumes = self.volume_net(features_volumes_t, params=style_vector_volumes)
                # volumes, style_vector_volumes
            vol_keypoints_3d, volumes = integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                             coord_volumes_t,
                                                                             softmax=self.volume_softmax)
            # vol_keypoints_3d, volumes
            if self.use_rnn and t < (dt-1):
                rnn_keypoints, hidden = self.rnn_model(vol_keypoints_3d.view(batch_size, 1, -1), hidden)
                rnn_keypoints = rnn_keypoints.view(batch_size, *keypoints_shape)
                style_vector_volumes = make_3d_heatmap(coord_volumes[:,t+1], rnn_keypoints, use_topk=False)
                rnn_keypoints_list.append(rnn_keypoints)
                # rnn_keypoints
            else:
                style_vector_volumes = volumes
                
            volumes_list.append(volumes)
            vol_keypoints_3d_list.append(vol_keypoints_3d)

        vol_keypoints_3d = torch.stack(vol_keypoints_3d_list, 1)
        volumes = torch.stack(volumes_list, 1)
        rnn_keypoints_3d = torch.stack(rnn_keypoints_list, 1)

        return (vol_keypoints_3d,
                rnn_keypoints_3d, # features
                volumes,
                None, # vol_confidences
                None, # cuboids
                coord_volumes,
                base_points
                )












