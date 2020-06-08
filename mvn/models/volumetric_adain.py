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

STYLE_VECTOR_CONST = None


class VolumetricTemporalAdaINNet(nn.Module):

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

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, 'transfer_cmu_to_human36m') else False

        # modules params
        self.encoder_type = config.model.encoder_type
        self.encoder_capacity = config.model.encoder_capacity_multiplier
        self.encoder_normalization_type = config.model.encoder_normalization_type
        self.upscale_bottleneck = config.model.upscale_bottleneck

        self.f2v_type = config.model.f2v_type
        self.f2v_intermediate_channels = config.model.f2v_intermediate_channels
        self.f2v_normalization_type = config.model.f2v_normalization_type
        
        self.v2v_type = config.model.v2v_type
        self.v2v_normalization_type = config.model.v2v_normalization_type
        assert self.v2v_normalization_type in ['group_norm','batch_norm']

        self.use_style_vector_as_v2v_input = config.model.use_style_vector_as_v2v_input if \
                                                hasattr(config.model, 'use_style_vector_as_v2v_input') else False
        self.temporal_condition_type = config.model.temporal_condition_type
        self.spade_broadcasting_type = config.model.spade_broadcasting_type
        self.style_grad_for_backbone = config.model.style_grad_for_backbone
        self.include_pivot = config.model.include_pivot
        self.use_auxilary_backbone = hasattr(config.model, 'auxilary_backbone')
        self.params_evolution = config.model.params_evolution if hasattr(config.model, 'params_evolution') else False
        self.style_forward = config.model.style_forward if hasattr(config.model, 'style_forward') else False


        self.use_style_decoder = config.model.use_style_decoder if hasattr(config.model, 'use_style_decoder') else False
        if self.use_style_decoder:
            self.style_decoder_type = config.model.style_decoder_type if hasattr(config.model, 'style_decoder_type') else 'v2v'

        self.use_style_pose_lstm_loss = config.model.use_style_pose_lstm_loss if hasattr(config.model, 'use_style_pose_lstm_loss') else False

        self.use_motion_extractor = config.model.use_motion_extractor if hasattr(config.model, 'use_motion_extractor') else False
        if self.use_motion_extractor:   
            self.motion_extractor_type = config.model.motion_extractor_type
            self.motion_extractor_from = config.model.motion_extractor_from
            self.resize_images_for_me = config.model.resize_images_for_me if hasattr(config.model, 'resize_images_for_me') else False
            self.images_me_target_size = config.model.images_me_target_size if hasattr(config.model, 'images_me_target_size') else None
        
        # modules dimensions
        self.volume_features_dim = config.model.volume_features_dim
        self.style_vector_dim = config.model.style_vector_dim
        self.encoded_feature_space = config.model.encoded_feature_space
            
        ############
        # BACKBONE #   
        ############
        self.backbone = pose_resnet.get_pose_net(config.model.backbone,
                                                 device=device,
                                                 strict=True)
        
        if self.use_auxilary_backbone:
            print ('Using auxilary {} backbone...'.format(config.model.auxilary_backbone.name))
            self.auxilary_backbone = pose_resnet.get_pose_net(config.model.auxilary_backbone,
                                                              device=device,
                                                              strict=True)  
        else:
            print ('Only {} backbone is used...'.format(config.model.backbone.name))
    
        ######################################
        # DEFINE TEMPORAL FEATURE ECTRACTION #   
        ######################################
        if self.use_style_pose_lstm_loss:
            assert self.pivot_type == 'intermediate' and self.keypoints_per_frame

            self.use_style_volume_for_SPL = config.model.use_style_volume_for_SPL if hasattr(config.model, 'use_style_volume_for_SPL') else True
            self.use_style_me_for_SPL = config.model.use_style_me_for_SPL if hasattr(config.model, 'use_style_me_for_SPL') else False
            self.use_style_tensor_for_SPL = config.model.use_style_tensor_for_SPL if hasattr(config.model, 'use_style_tensor_for_SPL') else False

            style_shape = None
            time_dim = self.dt//2
            self.upscale_style_for_SPL = self.use_style_tensor_for_SPL or self.use_style_me_for_SPL
            if self.upscale_style_for_SPL:
                style_shape = {'v2v':[time_dim,24,24],
                               'r2d':[5, 28, 28]}[self.f2v_type] # dt=9


            hidden_dim = config.model.style_pose_lstm_hidden_dim if hasattr(config.model, 'style_pose_lstm_hidden_dim') else 128
            self.style_pose_lstm_loss_decoder = StylePosesLSTM(self.style_vector_dim,
                                                                style_shape=style_shape,
                                                                upscale_style=self.upscale_style_for_SPL,
                                                                pose_space=self.num_joints,
                                                                hidden_dim=hidden_dim,
                                                                volume_size=self.volume_size, 
                                                                n_layers=3)

        if self.use_motion_extractor:
            if self.motion_extractor_type == 'r2d':

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

                self.motion_extractor = R2D(device,
                                            self.style_vector_dim, 
                                            self.f2v_normalization_type, 
                                            n_r2d_layers, 
                                            output_volume_dim,
                                            time=time,
                                            upscale_heatmap=upscale_me_heatmap,
                                            change_stride_layers=change_stride_layers,
                                            n_upscale_layers = n_upscale_layers,
                                            use_time_avg_pooling=use_time_avg_pooling)
            else:
                raise RuntimeError('Wrong motion_extractor_type') 
        
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


            elif self.f2v_type == 'const':
                style_vector_parameter_shape = config.model.style_vector_parameter_shape if \
                                                hasattr(config.model,'style_vector_parameter_shape') else None
                if style_vector_parameter_shape is None:
                    if self.temporal_condition_type in ['adain', 'adain_mlp']:
                        raise NotImplementedError
                    else:
                        style_vector_parameter_shape = {'unprojecting':[96,96],
                                                        'interpolate':[self.volume_size,self.volume_size,self.volume_size]}[self.spade_broadcasting_type]
                self.style_vector_parameter = nn.Parameter(data = torch.randn(self.style_vector_dim, *style_vector_parameter_shape))
                nn.init.xavier_normal_(self.style_vector_parameter.data)                
            else:
                raise RuntimeError('Wrong features_sequence_to_vector type')

            if self.f2v_type != 'const' and (not self.use_motion_extractor):
                self.encoder = get_encoder(self.encoder_type,
                                           config.model.backbone.name,
                                           self.encoded_feature_space,
                                           self.upscale_bottleneck,
                                           capacity = self.encoder_capacity,
                                           spatial_dimension = 2 if (self.f2v_type[-2:] == '2d' or \
                                                                 self.f2v_type == 'v2v') else 1,
                                           encoder_normalization_type = self.encoder_normalization_type)
        

        v2v_input_features_dim = (self.volume_features_dim + self.style_vector_dim) if \
                                  self.temporal_condition_type in ['stack', 'stack_poses'] else self.volume_features_dim
                                 
        if self.v2v_type == 'v1':
            self.volume_net = V2VModel_v1(v2v_input_features_dim,
                                          self.num_joints,
                                          normalization_type=self.v2v_normalization_type,
                                          volume_size=self.volume_size,
                                          temporal_condition_type = self.temporal_condition_type,
                                          style_vector_dim = self.style_vector_dim)

        elif self.v2v_type == 'conf':
            use_compound_norm = config.model.use_compound_norm if hasattr(config.model, 'use_compound_norm') else True
            self.volume_net = V2VModel(v2v_input_features_dim,
                                       self.num_joints,
                                       v2v_normalization_type=self.v2v_normalization_type,
                                       config=config.model.v2v_configuration,
                                       style_vector_dim=self.style_vector_dim,
                                       params_evolution=self.params_evolution,
                                       style_forward=self.style_forward,
                                       use_compound_norm=use_compound_norm,
                                       temporal_condition_type=self.temporal_condition_type)

        
        if self.volume_features_dim != 256:    
            self.process_features = nn.Conv2d(256, self.volume_features_dim, 1)
        else:
            self.process_features = nn.Sequential()    

        if self.use_style_decoder:
            self.style_decoder_part = config.model.style_decoder_part if hasattr(config.model, 'style_decoder_part') else 'after_pivot'
            if self.style_decoder_part == 'after_pivot':
                assert self.pivot_type == 'intermediate'
            if self.style_decoder_type == 'v2v':
                self.style_decoder = V2VModel(self.style_vector_dim,
                                               self.encoded_feature_space,
                                               v2v_normalization_type=self.f2v_normalization_type,
                                               config=config.model.style_decoder_configuration,
                                               style_vector_dim=None,
                                               temporal_condition_type=None)
            elif self.style_decoder_type == 'lstm':
                assert self.style_decoder_part == 'after_pivot'
                hidden_dim = config.model.style_decoder_hidden_dim if hasattr(config.model, 'style_decoder_hidden_dim') else 512
                self.style_decoder = FeatureDecoderLSTM(self.style_vector_dim,
                                                        self.encoded_feature_space,
                                                        hidden_dim=hidden_dim)
            else:
                raise RuntimeError('Wrong `style_decoder_type`')    

        self.STYLE_VECTOR_CONST=None
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

        return_me_vector = self.use_style_me_for_SPL if hasattr(self, 'use_style_me_for_SPL') else False
        decode_second_part = self.use_style_decoder and (self.style_decoder_part == 'after_pivot')
        decode_first_part = self.use_style_decoder and (self.style_decoder_part == 'before_pivot')
        
        # FIX
        process_only_pivot_image = (self.use_motion_extractor and self.motion_extractor_from == 'rgb') or self.f2v_type == 'const'
        process_first_half_of_images = self.use_style_pose_lstm_loss and (not decode_second_part)

        if process_only_pivot_image:
            images_batch_for_features = images_batch[:,self.pivot_index]
        elif process_first_half_of_images:
            images_batch_for_features = images_batch[:,:self.pivot_index+1].contiguous()
        else:
            images_batch_for_features = images_batch    

        # if master:
        #     set_trace()  

        ######################
        # FEATURE ECTRACTION #   
        ######################
        if self.use_auxilary_backbone:
            raise NotImplementedError()
        else:    
            # forward backbone
            heatmaps, features, _, vol_confidences, bottleneck = self.backbone(images_batch_for_features.view(-1, 3, *image_shape))

            # extract aux_features
            features_shape = features.shape[-2:]
            features_channels = features.shape[1]

            if process_only_pivot_image:
                pivot_features = features
                original_pivot_features = pivot_features.clone().detach()
                pivot_features = self.process_features(pivot_features).unsqueeze(1)
            else:    
                features = features.view(batch_size, -1, features_channels, *features_shape)
                aux_indexes_in_output = np.arange(self.pivot_index) if process_first_half_of_images else self.aux_indexes

                pivot_features = features[:,self.pivot_index,...]
                original_pivot_features = pivot_features.clone().detach()
                pivot_features = self.process_features(pivot_features).unsqueeze(1)
                aux_features = features if self.include_pivot else features[:,aux_indexes_in_output,...].contiguous()

                # set_trace()

                if decode_second_part:
                    # before pivot
                    aux_features = aux_features[:,:dt//2,...].contiguous()
                    # after pivot
                    features_for_loss = features[:,(dt//2)+1:,...].clone().detach()
                
                elif decode_first_part:
                    features_for_loss = aux_features.clone().detach()

                # features for style_vector reasoning    
                aux_features = aux_features.view(-1, *aux_features.shape[-3:])

                # extract aux_bottleneck
                if bottleneck is not None:
                    bottleneck_shape = bottleneck.shape[-2:]
                    bottleneck_channels = bottleneck.shape[1]
                    bottleneck = bottleneck.view(batch_size, -1, bottleneck_channels, *bottleneck_shape)
                    aux_bottleneck = bottleneck if self.include_pivot else bottleneck[:,aux_indexes_in_output,...].contiguous()
                    aux_bottleneck = aux_bottleneck.view(-1, bottleneck_channels, *bottleneck_shape)   
        
        proj_matricies_batch = update_camera(batch, batch_size, image_shape, features_shape, dt, device)
        proj_matricies_batch = proj_matricies_batch[:,self.pivot_index,...].unsqueeze(1) # pivot camera 

        ###############################
        # TEMPORAL FEATURE ECTRACTION #   
        ###############################
        if self.use_motion_extractor:
            if self.motion_extractor_from == 'rgb':
                aux_images = images_batch[:,:self.pivot_index].contiguous()
                if self.resize_images_for_me:
                    aux_images = F.interpolate(aux_images.view(-1, 3, *image_shape),size=self.images_me_target_size,  mode='bilinear')
                    aux_images = aux_images.view(batch_size, -1, 3, *self.images_me_target_size)
                    assert aux_images.dim() > 4 # assert time dim
                    assert aux_images.shape[1] > 1 # assert time dim
                if return_me_vector:    
                    style_vector, style_vector_me = self.motion_extractor(aux_images.transpose(1,2), return_me_vector=True)
                else:
                    style_vector  = self.motion_extractor(aux_images.transpose(1,2), return_me_vector=False)
            elif self.motion_extractor_from == 'features':
                style_vector = self.motion_extractor(aux_features.view(batch_size,
                                                                        -1, # time
                                                                        features_channels,
                                                                        *features_shape).transpose(1,2))
            elif self.motion_extractor_from == 'bottleneck':
                style_vector = self.motion_extractor(aux_bottleneck.view(batch_size,
                                                                         -1, # time 
                                                                         bottleneck_channels, 
                                                                         *bottleneck_shape).transpose(1,2))    
            else:
                raise RuntimeError('Wrong `motion_extractor_from`')    
        elif self.f2v_type != 'const':    
            if self.encoder_type == 'backbone':
                aux_bottleneck = aux_bottleneck if self.style_grad_for_backbone else aux_bottleneck.clone().detach()
                encoded_features = self.encoder(aux_bottleneck)
            elif self.encoder_type == 'features': 
                aux_features = aux_features if self.style_grad_for_backbone else aux_features.clone().detach()
                encoded_features = self.encoder(aux_features)
            else:
                raise RuntimeError('Unknown encoder')    

            encoded_features = encoded_features.view(batch_size, -1, *encoded_features.shape[1:]) # [batch_size, dt-1, encoded_fetures_dim]
            if self.f2v_type == 'v2v':
                encoded_features = torch.transpose(encoded_features, 1,2) # [batch_size, encoded_fetures_dim[0], dt-1, encoded_fetures_dim[1:]]
            style_vector = self.features_sequence_to_vector(encoded_features) # [batch_size, style_vector_dim]
        
        elif self.f2v_type == 'const':
            style_vector = torch.stack([self.style_vector_parameter]*batch_size,0)
        else:
            raise RuntimeError('No tempora feature extractor has been defined!')    

        #########
        # DEBUG #   
        #########
        if randomize_style:
            idx = torch.randperm(style_vector.nelement())
            style_vector = style_vector.view(-1)[idx].view(style_vector.size())
        if const_style_vector:
            if self.STYLE_VECTOR_CONST is None:
                self.STYLE_VECTOR_CONST = style_vector.data
                print ('STYLE_VECTOR_CONST INITED')
            else:
                style_vector = torch.tensor(self.STYLE_VECTOR_CONST).to(device)

        ##########
        # PELVIS #   
        ##########            
        if self.pelvis_type =='gt':
            tri_keypoints_3d = torch.from_numpy(np.array(batch['keypoints_3d'])[...,:3]).type(torch.float).to(device)
        else:
            raise RuntimeError('In absence of precalculated pelvis or gt pelvis, self.use_volumetric_pelvis should be True') 
        
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

        if self.use_style_pose_lstm_loss:
            lstm_coord_volumes = coord_volumes[:,self.pivot_index+1:,...].contiguous()
        if self.keypoints_per_frame:    
            coord_volumes = coord_volumes[:,self.pivot_index,...]
            base_points = base_points[:,self.pivot_index,...]

        ###############
        # V2V FORWARD #   
        ###############
        unproj_features = unproject_heatmaps(pivot_features,  
                                            proj_matricies_batch,
                                            coord_volumes, 
                                            volume_aggregation_method=self.volume_aggregation_method,
                                            vol_confidences=vol_confidences
                                            )
        
        if not self.temporal_condition_type in ['adain', 'adain_mlp']:
            if self.spade_broadcasting_type == 'unprojecting':
                    style_shape = style_vector.shape[-2:]
                    if style_shape != features_shape:
                        proj_matricies_style = update_camera(batch, batch_size, image_shape, style_shape, dt, device)
                        proj_matricies_style = proj_matricies_style[:,self.pivot_index,...].unsqueeze(1) # pivot camera
                    else:
                        proj_matricies_style = proj_matricies_batch     
                    style_vector_volumes = unproject_heatmaps(style_vector.unsqueeze(1),  
                                                             proj_matricies_style, 
                                                             coord_volumes, 
                                                             volume_aggregation_method=self.volume_aggregation_method,
                                                             vol_confidences=vol_confidences
                                                             )

            elif self.spade_broadcasting_type == 'interpolate':
                style_vector_volumes = F.interpolate(style_vector,
                                                     size=(self.volume_size,self.volume_size,self.volume_size), 
                                                     mode='trilinear')
            else:
                raise KeyError('Unknown spade_broadcasting_type')

        ########################## 
        # VOLUMES FEEDING TO V2V #   
        ##########################
        torch.cuda.empty_cache()         
        if self.temporal_condition_type in ['adain', 'adain_mlp']:
            volumes = self.volume_net(unproj_features, params=style_vector)
        elif self.temporal_condition_type == 'spade':
            if self.use_style_vector_as_v2v_input:
                volumes = self.volume_net(style_vector_volumes, 
                                          params=unproj_features)
            else:    
                volumes = self.volume_net(unproj_features, 
                                          params=style_vector_volumes)
        elif self.temporal_condition_type == 'stack':
            volumes = self.volume_net(torch.cat([unproj_features, style_vector_volumes], 1))
        else:
            raise RuntimeError('Wrong self.temporal_condition_type, should be in [`adain`, `adain_mlp`, `stack`, `spade`]')    
                
        # integral 3d
        if self.style_forward:
            volumes, style_vector_volumes_output = volumes
        vol_keypoints_3d, volumes = integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                         coord_volumes,
                                                                         softmax=self.volume_softmax)

        decoded_features = None
        if self.use_style_decoder:
            if self.style_decoder_type == 'v2v':
                decoded_features = self.style_decoder(style_vector)
                decoded_features = torch.transpose(decoded_features,1,2)
            elif self.style_decoder_type == 'lstm':
                decoded_features = self.style_decoder(style_vector,
                                                      original_pivot_features)

        if self.use_style_pose_lstm_loss:
            time=lstm_coord_volumes.shape[1]
            assert time == dt//2

            if self.use_style_volume_for_SPL:
                lstm_volumes = self.style_pose_lstm_loss_decoder(style_vector_volumes, volumes, device, time=time)
            elif self.use_style_me_for_SPL:
                lstm_volumes = self.style_pose_lstm_loss_decoder(style_vector_me, volumes, device, time=time)
            else:    
                lstm_volumes = self.style_pose_lstm_loss_decoder(style_vector, volumes, device, time=time)

            lstm_volumes = lstm_volumes.view(-1, *lstm_volumes.shape[2:])
            lstm_coord_volumes = lstm_coord_volumes.view(-1, *lstm_coord_volumes.shape[2:])
            lstm_keypoints_3d, _ = integrate_tensor_3d_with_coordinates(lstm_volumes * self.volume_multiplier,
                                                                        lstm_coord_volumes,
                                                                        softmax=self.volume_softmax)

            lstm_keypoints_3d = lstm_keypoints_3d.view(batch_size, time, *lstm_keypoints_3d.shape[1:])
            vol_keypoints_3d = [vol_keypoints_3d, lstm_keypoints_3d]

        style_output = None
        if return_me_vector:
            style_output = [style_vector, style_vector_me]
        elif self.style_forward:
            style_output = [style_vector, style_vector_volumes_output]
        else:
            style_output = style_vector
        return [vol_keypoints_3d,
                features_for_loss if self.use_style_decoder else None,
                volumes,
                vol_confidences,
                None, # cuboids
                coord_volumes,
                base_points,
                style_output,
                unproj_features,
                decoded_features
                ]















