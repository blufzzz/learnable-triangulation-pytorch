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
from mvn.utils.op import get_coord_volumes, unproject_heatmaps, integrate_tensor_3d_with_coordinates
from mvn.utils.multiview import update_camera
from mvn.utils.misc import get_capacity, description
from mvn.utils import volumetric
from mvn.models.v2v import V2VModel, R2D, SqueezeLayer
from mvn.models.v2v_models import V2VModel_v1
from mvn.models.temporal import Seq2VecRNN,\
                                Seq2VecCNN, \
                                Seq2VecRNN2D, \
                                Seq2VecCNN2D, \
                                get_encoder, \
                                FeatureDecoderLSTM, \
                                StylePosesLSTM

from pytorch_convolutional_rnn.convolutional_rnn import Conv3dLSTM
from IPython.core.debugger import set_trace


class VolumetricTemporalLSTM(nn.Module):

    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.kind = config.model.kind
        self.num_joints = config.model.backbone.num_joints
        self.dt  = config.dataset.dt
        self.keypoints_per_frame = config.dataset.keypoints_per_frame if \
                                         hasattr(config.dataset, 'keypoints_per_frame') else True
        assert self.keypoints_per_frame

        self.pivot_index =  {'first':self.dt-1,
                            'intermediate':self.dt//2}[config.dataset.pivot_type]

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
        self.pelvis_type = config.model.pelvis_type if \
                            hasattr(config.model, 'pelvis_type') else 'gt'

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if \
                                        hasattr(config.model, 'transfer_cmu_to_human36m') else False

        self.v2v_type = config.model.v2v_type
        self.v2v_normalization_type = config.model.v2v_normalization_type
        self.include_pivot = config.model.include_pivot
         
        # before v2v
        self.lstm_on_feature_volumes = config.model.lstm_on_feature_volumes if \
                                       hasattr(config.model, 'lstm_on_feature_volumes') else False

        # after v2v                               
        self.lstm_on_pose_volumes = config.model.lstm_on_pose_volumes if \
                                    hasattr(config.model, 'lstm_on_pose_volumes') else True
                                    
        self.volume_features_dim = config.model.volume_features_dim

        # pose
        if self.lstm_on_pose_volumes:
            self.lstm_in_channels = config.model.lstm_in_channels
            self.lstm_out_channels = config.model.lstm_out_channels
            self.lstm_bidirectional = config.model.lstm_bidirectional
            self.lstm_layers = config.model.lstm_layers

        # features
        if self.lstm_on_feature_volumes:
            self.lstm_bidirectional_features = config.model.lstm_bidirectional_features
            self.lstm_layers_features = config.model.lstm_layers_features

        self.disentangle = config.model.disentangle
        self.entangle_processing_type = config.model.entangle_processing_type
        self.epn_normalization_type = config.model.entangle_processing_normalization_type

        self.use_final_processing = self.lstm_on_pose_volumes and (self.lstm_out_channels != self.num_joints) and not self.disentangle

        # modules
        self.backbone = pose_resnet.get_pose_net(config.model.backbone,
                                                 device=device,
                                                 strict=True)
        
        if self.lstm_on_pose_volumes:
            self.lstm3d = Conv3dLSTM(in_channels=self.lstm_in_channels, 
                                     out_channels=self.lstm_out_channels,
                                     bidirectional=self.lstm_bidirectional,
                                     num_layers=self.lstm_layers,
                                     batch_first=True,
                                     kernel_size=3)

        if self.lstm_on_feature_volumes:
            self.lstm3d_feature_volumes = Conv3dLSTM(in_channels=self.volume_features_dim, 
                                                     out_channels=self.volume_features_dim,
                                                     bidirectional=self.lstm_bidirectional_features,
                                                     num_layers=self.lstm_layers_features,
                                                     batch_first=True,
                                                     kernel_size=3)

        if self.disentangle:
            if self.entangle_processing_type == 'stack':
                self.entangle_processing_net = V2VModel(self.lstm_in_channels + self.lstm_out_channels,
                                                        self.num_joints,
                                                        v2v_normalization_type=self.epn_normalization_type,
                                                        config=config.model.epn_configuration)

            elif self.entangle_processing_type == 'sum':
                assert self.lstm_in_channels == self.lstm_out_channels
                self.entangle_processing_net = V2VModel(self.lstm_out_channels,
                                                        self.num_joints,
                                                        v2v_normalization_type=self.epn_normalization_type,
                                                        config=config.model.epn_configuration)    

            else:
                raise RuntimeError('Unknown entangle_processing_type, supported [`stack`,`sum`]')



        if self.v2v_type == 'v1':
            self.volume_net = V2VModel_v1(self.volume_features_dim,
                                          self.lstm_in_channels if self.lstm_on_pose_volumes else self.num_joints,
                                          normalization_type=self.v2v_normalization_type,
                                          volume_size=self.volume_size)

        elif self.v2v_type == 'conf':
            self.volume_net = V2VModel(v2v_input_features_dim,
                                       self.lstm_in_channels if self.lstm_on_pose_volumes else self.num_joints,
                                       v2v_normalization_type=self.v2v_normalization_type,
                                       config=config.model.v2v_configuration)

        self.process_features = nn.Conv2d(256, self.volume_features_dim, 1)

        if self.use_final_processing:
            self.final_processing = nn.Conv3d(self.lstm_out_channels, self.num_joints, 1)

        description(self)

    def forward(self, images_batch, batch, randomize_style=False):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]
        fictive_view = 1
        assert self.dt == dt
     
        # forward backbone
        heatmaps, features, _, vol_confidences, _ = self.backbone(images_batch.view(-1, 3, *image_shape))
        
        # extract aux_features
        features = self.process_features(features)
        features_shape = features.shape[-2:]
        features_channels = features.shape[1]

        proj_matricies_batch = update_camera(batch, batch_size, image_shape, features_shape, dt, device)

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

        # amend coord_volumes position                                                         
        coord_volumes, _, base_points = get_coord_volumes(self.kind, 
                                                                self.training, 
                                                                self.rotation,
                                                                self.cuboid_side,
                                                                self.volume_size, 
                                                                device,
                                                                keypoints=tri_keypoints_3d
                                                                )

        coord_volumes = coord_volumes.view(-1, self.volume_size, self.volume_size, self.volume_size,3)
        proj_matricies_batch = proj_matricies_batch.view(-1, fictive_view, *proj_matricies_batch.shape[2:])
        features = features.view(-1, fictive_view, features_channels, *features_shape)

        # lift each feature-map to distinct volume and aggregate 
        volumes = unproject_heatmaps(features,  
                                     proj_matricies_batch, 
                                     coord_volumes, 
                                     volume_aggregation_method=self.volume_aggregation_method,
                                     vol_confidences=vol_confidences,
                                     fictive_views=None
                                     )
        if self.lstm_on_feature_volumes:
            volumes = volumes.view(batch_size, dt, *volumes.shape[-4:])
            volumes, _ = self.lstm3d_feature_volumes(volumes, None)
            volumes = volumes.view(batch_size*dt, *volumes.shape[-4:])

        volumes = self.volume_net(volumes) 
        # squeezed

        if self.lstm_on_pose_volumes:
            # unqueezed
            volumes = volumes.view(batch_size, dt, *volumes.shape[-4:])
            rnn_volumes, _ = self.lstm3d(volumes, None) 
            if self.disentangle:
                
                if self.entangle_processing_type == 'stack':
                    volumes = torch.cat([volumes, rnn_volumes], 2)
                else: #self.entangle_processing_type == 'sum':
                    volumes = volumes + rnn_volumes
                volumes = volumes.view(-1, *volumes.shape[-4:])    
                volumes = self.entangle_processing_net(volumes)
                # squeezed
            else:
                volumes = rnn_volumes
                volumes = volumes.view(batch_size*dt, *volumes.shape[-4:])  
                # squeezed    

        if self.use_final_processing:
            volumes = volumes.view(-1, *volumes.shape[-4:])     
            volumes = self.final_processing(volumes)   
            # squeezed    

        # integral 3d
        vol_keypoints_3d, volumes = integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)
        volumes = volumes.view(batch_size, dt, *volumes.shape[-4:])
        vol_keypoints_3d = vol_keypoints_3d.view(batch_size, dt, *vol_keypoints_3d.shape[-2:]) 

        coord_volumes = coord_volumes.view(batch_size*dt, *coord_volumes.shape[-4:]) 
        base_points = base_points.view(batch_size*dt, *base_points.shape[-1:]) 


        return (vol_keypoints_3d,
                None, # features
                volumes,
                None, # vol_confidences
                None, # cuboids
                coord_volumes,
                base_points
                )











