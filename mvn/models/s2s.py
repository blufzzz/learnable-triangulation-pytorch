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

from IPython.core.debugger import set_trace
from pytorch_convolutional_rnn.convolutional_rnn import Conv3dLSTM
from mvn.models.e3d_lstm import E3DLSTM

STYLE_VECTOR_CONST = None


class S2SModel(nn.Module):

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

        # cuboid
        self.cuboid_side = config.model.cuboid_side
        self.cuboid_multiplier = config.model.cuboid_multiplier
        self.rotation = config.model.rotation

        # pelvis
        self.pelvis_type = config.model.pelvis_type if hasattr(config.model, 'pelvis_type') else 'gt'

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, 'transfer_cmu_to_human36m') else False

        self.f2v_type = config.model.f2v_type
        self.f2v_intermediate_channels = config.model.f2v_intermediate_channels
        self.f2v_normalization_type = config.model.f2v_normalization_type
        self.motion_extractor_from = config.model.motion_extractor_from if hasattr(config.model, 'motion_extractor_from') else None
        
        self.style_grad_for_backbone = config.model.style_grad_for_backbone
        self.include_pivot = config.model.include_pivot

        # modules dimensions
        self.style_vector_dim = config.model.style_vector_dim

        ############
        # BACKBONE #   
        ############
        if not (self.f2v_type == 'r2d' and self.motion_extractor_from == 'rgb'):
            self.backbone = pose_resnet.get_pose_net(config.model.backbone,
                                                     device=device,
                                                     strict=True)


        ###########
        # ENCODER #
        ###########
        if not (self.f2v_type == 'r2d' and self.motion_extractor_from == 'rgb'):
            self.encoder_type = config.model.encoder_type
            self.encoder_capacity = config.model.encoder_capacity_multiplier
            self.encoder_normalization_type = config.model.encoder_normalization_type
            self.upscale_bottleneck = config.model.upscale_bottleneck   
            self.encoded_feature_space = config.model.encoded_feature_space

            self.encoder = get_encoder(self.encoder_type,
                                       config.model.backbone.name,
                                       self.encoded_feature_space,
                                       self.upscale_bottleneck,
                                       capacity = self.encoder_capacity,
                                       spatial_dimension = 2 if (self.f2v_type[-2:] == '2d' or \
                                                             self.f2v_type in ['v2v','r2d', 'c3dlstm','e3dlstm']) else 1,
                                       encoder_normalization_type = self.encoder_normalization_type)


        ######################################
        # DEFINE TEMPORAL FEATURE ECTRACTION #   
        ######################################
        if self.f2v_type == 'r2d':
            self.resize_images_for_me = config.model.resize_images_for_me if hasattr(config.model, 'resize_images_for_me') else False
            self.images_me_target_size = config.model.images_me_target_size if hasattr(config.model, 'images_me_target_size') else None
            n_r2d_layers = config.model.n_r2d_layers if hasattr(config.model, 'n_r2d_layers') else 2
            upscale_me_heatmap = config.model.upscale_me_heatmap if hasattr(config.model, 'upscale_me_heatmap') else False
            n_upscale_layers = config.model.n_upscale_layers if hasattr(config.model, 'n_upscale_layers') else 3
            change_stride_layers = config.model.change_stride_layers if hasattr(config.model, 'change_stride_layers') else None
            use_time_avg_pooling = config.model.use_time_avg_pooling if hasattr(config.model, 'use_time_avg_pooling') else False
            time_dim = config.model.time_dim if hasattr(config.model, 'time_dim') else None
            output_heatmap_shape = config.model.output_heatmap_shape if hasattr(config.model, 'output_heatmap_shape') else [96,96]

            if hasattr(config.model, 'output_volume_dim'):
                output_volume_dim = config.model.output_volume_dim
            else:    
                output_volume_dim = {'unprojecting':2,
                                     'interpolate':3}[self.spade_broadcasting_type]

            if change_stride_layers is None:                         
                time = (self.dt-1 if self.pivot_type == 'first' else self.dt//2)  
            else:
                time = time_dim

            self.features_sequence_to_vector = R2D(device,
                                                    self.style_vector_dim, 
                                                    self.f2v_normalization_type, 
                                                    n_r2d_layers, 
                                                    output_volume_dim,
                                                    time=time,
                                                    upscale_heatmap=upscale_me_heatmap,
                                                    change_stride_layers=change_stride_layers,
                                                    n_upscale_layers = n_upscale_layers,
                                                    use_time_avg_pooling=use_time_avg_pooling)
        
        elif self.f2v_type == 'c3dlstm':
            self.lstm_layers = config.model.lstm_layers
            self.features_sequence_to_vector = Conv3dLSTM(in_channels=self.encoded_feature_space, 
                                                         out_channels=self.style_vector_dim,
                                                         bidirectional=False,
                                                         num_layers=self.lstm_layers,
                                                         batch_first=True,
                                                         kernel_size=3)

        elif self.f2v_type == 'e3dlstm':
            self.lstm_layers = config.model.lstm_layers
            self.eidetic_lstm_tau = config.model.eidetic_lstm_tau
            self.features_sequence_to_vector = E3DLSTM(input_shape=[self.encoded_feature_space, 
                                                        self.volume_size,
                                                        self.volume_size,
                                                        self.volume_size],
                                                        hidden_size=self.style_vector_dim, 
                                                        num_layers = self.lstm_layers, 
                                                        kernel_size=(3, 3, 3),
                                                        tau=self.eidetic_lstm_tau)    


        else:           
            if self.f2v_type == 'lstm':
                self.features_sequence_to_vector = Seq2VecRNN(input_features_dim = self.encoded_feature_space,
                                                              output_features_dim = self.style_vector_dim,
                                                              hidden_dim = self.f2v_intermediate_channels)
            elif self.f2v_type == 'cnn':
                self.features_sequence_to_vector = Seq2VecCNN(input_features_dim = self.encoded_feature_space,
                                                              output_features_dim=  self.style_vector_dim,
                                                              intermediate_channels = self.f2v_intermediate_channels,
                                                              normalization_type = self.f2v_normalization_type,
                                                              dt=self.dt if self.include_pivot else self.dt-1,
                                                              kernel_size = 3)

            elif self.f2v_type == 'lstm2d':
                self.features_sequence_to_vector = Seq2VecRNN2D(input_features_dim = self.encoded_feature_space,
                                                                output_features_dim=  self.style_vector_dim,
                                                                hidden_dim = self.f2v_intermediate_channels)     
            elif self.f2v_type == 'cnn2d':
                self.features_sequence_to_vector = Seq2VecCNN2D(input_features_dim = self.encoded_feature_space,
                                                                output_features_dim=  self.style_vector_dim,
                                                                intermediate_channels = self.f2v_intermediate_channels,
                                                                normalization_type = self.f2v_normalization_type,
                                                                dt=self.dt if self.include_pivot else self.dt-1,
                                                                kernel_size = 3)

            elif self.f2v_type == 'v2v':
                upscale_to_heatmap = config.model.f2v_configuration.upscale_to_heatmap if \
                                    hasattr(config.model.f2v_configuration, 'upscale_to_heatmap') else False
                time_avg_pool_only = config.model.f2v_configuration.time_avg_pool_only if \
                                    hasattr(config.model.f2v_configuration, 'time_avg_pool_only') else False                    
                time_conv_avg_pool = config.model.f2v_configuration.time_conv_avg_pool if \
                                    hasattr(config.model.f2v_configuration, 'time_conv_avg_pool') else False

                back_layer_output_channels = config.model.f2v_configuration.back_layer_output_channels if \
                                    hasattr(config.model.f2v_configuration, 'back_layer_output_channels') else 64                        

                features_sequence_to_vector_backbone = V2VModel(self.encoded_feature_space,
                                                                    self.style_vector_dim,
                                                                    v2v_normalization_type=self.f2v_normalization_type,
                                                                    config=config.model.f2v_configuration,
                                                                    style_vector_dim=None,
                                                                    temporal_condition_type=None,
                                                                    back_layer_output_channels = back_layer_output_channels)
                if upscale_to_heatmap:
                    assert self.spade_broadcasting_type == 'unprojecting'

                    self.features_sequence_to_vector = nn.Sequential(features_sequence_to_vector_backbone,
                                                                    nn.ConvTranspose3d(self.style_vector_dim,
                                                                                         self.style_vector_dim,
                                                                                         kernel_size=[1,2,2],
                                                                                         stride=[1,2,2]),
                                                                    nn.GroupNorm(32,self.style_vector_dim),
                                                                    nn.ConvTranspose3d(self.style_vector_dim,
                                                                                         self.style_vector_dim,
                                                                                         kernel_size=[1,2,2],
                                                                                         stride=[1,2,2]),
                                                                    nn.AdaptiveAvgPool3d([1, 96,96]),
                                                                    SqueezeLayer(-3))
                
                elif time_avg_pool_only:
                    self.features_sequence_to_vector = nn.Sequential(features_sequence_to_vector_backbone,
                                                                     nn.AdaptiveAvgPool3d([1, 96,96]),
                                                                     SqueezeLayer(-3))

                elif time_conv_avg_pool:
                    assert self.pivot_type == 'first'
                    time_dim_after_v2v = self.dt if self.include_pivot else self.dt-1
                    self.features_sequence_to_vector = nn.Sequential(features_sequence_to_vector_backbone,
                                                                     nn.Conv3d(self.style_vector_dim,
                                                                                self.style_vector_dim,
                                                                                kernel_size=(time_dim_after_v2v,1,1)),
                                                                     nn.AdaptiveAvgPool3d([1, 96,96]),
                                                                     SqueezeLayer(-3))

                else:
                    self.features_sequence_to_vector = features_sequence_to_vector_backbone 
        
        self.style_to_volumes = nn.Conv3d(self.style_vector_dim, self.num_joints, 1)              
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
        assert self.dt == dt

        process_nothing = (self.f2v_type == 'r2d' and self.motion_extractor_from == 'rgb')
        images_batch_for_features = images_batch    

        ######################
        # FEATURE ECTRACTION #   
        ######################
        if not process_nothing:
            grad_context = torch.autograd.enable_grad if self.style_grad_for_backbone else torch.no_grad
            with grad_context():
                heatmaps, features, _, vol_confidences, bottleneck = self.backbone(images_batch_for_features.view(-1, 3, *image_shape))

            # extract aux_features
            features_shape = features.shape[-2:]
            features_channels = features.shape[1]

            features = features.view(batch_size, -1, features_channels, *features_shape)
            aux_indexes_in_output = self.aux_indexes
            aux_features = features if self.include_pivot else features[:,aux_indexes_in_output,...].contiguous()

            # features for style_vector reasoning    
            aux_features = aux_features.view(-1, *aux_features.shape[-3:])

            # extract aux_bottleneck
            if bottleneck is not None:
                bottleneck_shape = bottleneck.shape[-2:]
                bottleneck_channels = bottleneck.shape[1]
                bottleneck = bottleneck.view(batch_size, -1, bottleneck_channels, *bottleneck_shape)
                aux_bottleneck = bottleneck if self.include_pivot else bottleneck[:,aux_indexes_in_output,...].contiguous()
                aux_bottleneck = aux_bottleneck.view(-1, bottleneck_channels, *bottleneck_shape)   

        ###############################
        # TEMPORAL FEATURE ECTRACTION #   
        ###############################
        if self.f2v_type == 'r2d':
            if self.motion_extractor_from == 'rgb':
                aux_images = images_batch if self.include_pivot else images_batch[:,self.aux_indexes].contiguous()
                if self.resize_images_for_me:
                    aux_images = F.interpolate(aux_images.view(-1, 3, *image_shape),size=self.images_me_target_size,  mode='bilinear')
                    aux_images = aux_images.view(batch_size, -1, 3, *self.images_me_target_size)
                    assert aux_images.dim() > 4 # assert time dim
                    assert aux_images.shape[1] > 1 # assert time dim
                style_vector  = self.features_sequence_to_vector(aux_images.transpose(1,2), return_me_vector=False)
            elif self.motion_extractor_from == 'features':
                style_vector = self.features_sequence_to_vector(aux_features.view(batch_size,
                                                                        -1, # time
                                                                        features_channels,
                                                                        *features_shape).transpose(1,2))
            elif self.motion_extractor_from == 'bottleneck':
                style_vector = self.features_sequence_to_vector(aux_bottleneck.view(batch_size,
                                                                         -1, # time 
                                                                         bottleneck_channels, 
                                                                         *bottleneck_shape).transpose(1,2))    
            else:
                raise RuntimeError('Wrong `motion_extractor_from`')    

        else:    
            if self.encoder_type == 'backbone':
                aux_bottleneck = aux_bottleneck
                encoded_features = self.encoder(aux_bottleneck)
            elif self.encoder_type == 'features': 
                aux_features = aux_features
                encoded_features = self.encoder(aux_features)
            else:
                raise RuntimeError('Unknown encoder')    
            
            if self.f2v_type in ['c3dlstm', 'e3dlstm']:
                ##############################
                # AUXILARY VOLUMES CREATING  #   
                ##############################
                proj_matricies_batch = update_camera(batch, batch_size, image_shape, features_shape, dt, device)
                tri_keypoints_3d = torch.from_numpy(np.array(batch['keypoints_3d'])[...,:3]).type(torch.float).to(device)
                assert tri_keypoints_3d.dim() == 4
                assert self.include_pivot
                self.keypoints_for_each_frame = True
                fictive_view = 1
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
                encoded_features = encoded_features.unsqueeze(1) # add fictive view
                # lift each feature-map to distinct volume and aggregate 
                volumes = unproject_heatmaps(encoded_features,  
                                             proj_matricies_batch, 
                                             coord_volumes, 
                                             volume_aggregation_method='no_aggregation',
                                             vol_confidences=vol_confidences,
                                             fictive_views=None
                                             )

                volumes = volumes.view(batch_size, dt, *volumes.shape[-4:])
                volumes, _ = self.features_sequence_to_vector(volumes, None)
                style_vector = volumes[:,-1,...] # take the last hiddent state

                # take coord volume and base point corresponding to final estimation
                coord_volumes = coord_volumes.view(batch_size, dt, *coord_volumes.shape[-4:])[:,-1,...]
                base_points = base_points[:,-1,...]

            else:
                encoded_features = encoded_features.view(batch_size, -1, *encoded_features.shape[1:])
                if self.f2v_type == 'v2v':
                    encoded_features = torch.transpose(encoded_features, 1,2)
                style_vector = self.features_sequence_to_vector(encoded_features)
 
        ##########
        # PELVIS #   
        ##########
        if not self.f2v_type in ['c3dlstm', 'e3dlstm']:            
            if self.pelvis_type =='gt':
                tri_keypoints_3d = torch.from_numpy(np.array(batch['keypoints_3d'])[...,:3]).type(torch.float).to(device)
            else:
                raise RuntimeError('In absence of precalculated pelvis or gt pelvis, self.use_volumetric_pelvis should be True') 
        
            ###########################
            # PIVOT VOLUMES CREATING  #   
            ###########################
            coord_volumes, _, base_points = get_coord_volumes(self.kind, 
                                                                self.training, 
                                                                self.rotation,
                                                                self.cuboid_side,
                                                                self.volume_size, 
                                                                device,
                                                                keypoints=tri_keypoints_3d
                                                                )

        style_vector_volumes = F.interpolate(style_vector,
                                             size=(self.volume_size,self.volume_size,self.volume_size), 
                                             mode='trilinear')
        # map to joints features
        style_vector_volumes = self.style_to_volumes(style_vector_volumes)
        vol_keypoints_3d, style_vector_volumes = integrate_tensor_3d_with_coordinates(style_vector_volumes * self.volume_multiplier,
                                                                                     coord_volumes,
                                                                                     softmax=self.volume_softmax)
        return [vol_keypoints_3d, 
                style_vector_volumes, 
                coord_volumes,
                base_points]














