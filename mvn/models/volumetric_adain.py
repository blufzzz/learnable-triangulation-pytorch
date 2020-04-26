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
from mvn.models.v2v import V2VModel, C3D, R2D
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
        self.use_precalculated_pelvis = config.model.use_precalculated_pelvis
        self.use_gt_pelvis = config.model.use_gt_pelvis
        self.use_volumetric_pelvis = config.model.use_volumetric_pelvis

        assert self.use_precalculated_pelvis or self.use_gt_pelvis, 'One of the flags "use_<...>_pelvis" should be True'

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m

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

        self.temporal_condition_type = config.model.temporal_condition_type
        self.spade_broadcasting_type = config.model.spade_broadcasting_type
        self.style_grad_for_backbone = config.model.style_grad_for_backbone
        self.include_pivot = config.model.include_pivot
        self.use_auxilary_backbone = hasattr(config.model, 'auxilary_backbone')

        self.use_style_decoder = config.model.use_style_decoder if hasattr(config.model, 'use_style_decoder') else False
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

        if self.temporal_condition_type == 'stack':
            assert self.v2v_normalization_type in ['group_norm','batch_norm']
            
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
            style_shape = {'v2v':[5,24,24]}[self.f2v_type]
            hidden_dim = config.model.style_pose_lstm_hidden_dim if hasattr(config.model, 'style_pose_lstm_hidden_dim') else 128
            assert self.pivot_type == 'intermediate' and self.keypoints_per_frame
            self.style_pose_lstm_loss_decoder = StylePosesLSTM(self.style_vector_dim,
                                                                style_shape,
                                                                pose_space=self.num_joints,
                                                                hidden_dim=hidden_dim,
                                                                volume_size=self.volume_size, 
                                                                time=style_shape[0], 
                                                                n_layers=3)

        if self.use_motion_extractor:
            if self.motion_extractor_type == 'r2d':
                n_r2d_layers = config.model.n_r2d_layers if hasattr(config.model, 'n_r2d_layers') else 2
                upscale_me_heatmap = config.model.upscale_me_heatmap if hasattr(config.model, 'upscale_me_heatmap') else False
                
                if hasattr(config.model, 'output_volume_dim'):
                    output_volume_dim = config.model.output_volume_dim
                else:    
                    output_volume_dim = {'unprojecting':2,
                                         'interpolate':3}[self.spade_broadcasting_type]
                time = self.dt-1 if self.pivot_type == 'first' else self.dt//2
                self.motion_extractor = R2D(device,
                                            self.style_vector_dim, 
                                            self.f2v_normalization_type, 
                                            n_r2d_layers, 
                                            output_volume_dim,
                                            time=time,
                                            upscale_heatmap=upscale_me_heatmap)
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
                self.features_sequence_to_vector = V2VModel(self.encoded_feature_space,
                                                            self.style_vector_dim,
                                                            v2v_normalization_type=self.f2v_normalization_type,
                                                            config=config.model.f2v_configuration,
                                                            style_vector_dim=None,
                                                            temporal_condition_type=None)

            else:
                raise RuntimeError('Wrong features_sequence_to_vector type')

               
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
            self.volume_net = V2VModel(v2v_input_features_dim,
                                       self.num_joints,
                                       v2v_normalization_type=self.v2v_normalization_type,
                                       config=config.model.v2v_configuration,
                                       style_vector_dim=self.style_vector_dim,
                                       temporal_condition_type=self.temporal_condition_type)

        self.process_features = nn.Conv2d(256, self.volume_features_dim, 1)

        if self.use_style_decoder:
            assert self.pivot_type == 'intermediate'
            if self.style_decoder_type == 'v2v':
                self.style_decoder = V2VModel(self.style_vector_dim,
                                               self.encoded_feature_space,
                                               v2v_normalization_type=self.f2v_normalization_type,
                                               config=config.model.style_decoder_configuration,
                                               style_vector_dim=None,
                                               temporal_condition_type=None)
            elif self.style_decoder_type == 'lstm': 
                hidden_dim = config.model.style_decoder_hidden_dim if hasattr(config.model, 'style_decoder_hidden_dim') else 512
                self.style_decoder = FeatureDecoderLSTM(self.style_vector_dim,
                                                        self.encoded_feature_space,
                                                        hidden_dim=hidden_dim)


        self.STYLE_VECTOR_CONST=None
        description(self)

    def forward(self, 
                images_batch, 
                batch, 
                randomize_style=False, 
                const_style_vector=False,
                return_me_vector = False,
                debug=False):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]
        assert self.dt == dt

        backbone_indexes = self.dt

        if debug:
            set_trace()
        ######################
        # FEATURE ECTRACTION #   
        ######################
        if self.use_auxilary_backbone:
            raise RuntimeError
            # pivot_images_batch = images_batch[:,self.pivot_index,...]
            # aux_images_batch = images_batch if self.include_pivot else images_batch[:,self.aux_indexes,...].contiguous()
            # aux_images_batch = aux_images_batch.view(-1, 3, *image_shape)

            # aux_heatmaps, aux_features, _, _, aux_bottleneck = self.auxilary_backbone(aux_images_batch)
            # pivot_heatmaps, pivot_features, _, _, pivot_bottleneck = self.backbone(pivot_images_batch)

            # features_shape = pivot_features.shape[-2:]
            # features_channels = pivot_features.shape[1]
            # bottleneck_shape = aux_bottleneck.shape[-2:]
            # bottleneck_channels = aux_bottleneck.shape[1]

            # original_pivot_features = pivot_features.clone().detach()
            # pivot_features = self.process_features(pivot_features).unsqueeze(1)

            # if self.use_style_decoder or self.use_style_pose_lstm_loss:
            #     aux_features = aux_features.view(batch_size, -1, *aux_features.shape[-3:])
            #     aux_features = aux_features[:,:dt//2].contiguous()
            #     features_for_loss = aux_features[:,dt//2:].clone().detach()

            # if aux_features is not None:
            #     aux_features = aux_features.view(-1, *aux_features.shape[-3:])

        else:    
            # forward backbone
            heatmaps, features, _, vol_confidences, bottleneck = self.backbone(images_batch.view(-1, 3, *image_shape))

            if debug:
                set_trace()
            # extract aux_features
            features_shape = features.shape[-2:]
            features_channels = features.shape[1]
            features = features.view(batch_size, dt, features_channels, *features_shape)
            
            pivot_features = features[:,self.pivot_index,...]
            original_pivot_features = pivot_features.clone().detach()
            pivot_features = self.process_features(pivot_features).unsqueeze(1)
            aux_features = features if self.include_pivot else features[:,self.aux_indexes,...].contiguous()

            if debug:
                set_trace()
            if self.use_style_decoder or self.use_style_pose_lstm_loss:
                aux_features = aux_features[:,:dt//2,...].contiguous()
                features_for_loss = features[:,(dt//2)+1:,...].clone().detach()
            aux_features = aux_features.view(-1, *aux_features.shape[-3:])

            # extract aux_bottleneck
            bottleneck_shape = bottleneck.shape[-2:]
            bottleneck_channels = bottleneck.shape[1]
            bottleneck = bottleneck.view(batch_size, dt, bottleneck_channels, *bottleneck_shape)
            aux_bottleneck = bottleneck if self.include_pivot else bottleneck[:,self.aux_indexes,...].contiguous()
            aux_bottleneck = aux_bottleneck.view(-1, bottleneck_channels, *bottleneck_shape)   
            
        proj_matricies_batch = update_camera(batch, batch_size, image_shape, features_shape, dt, device)
        proj_matricies_batch = proj_matricies_batch[:,self.pivot_index,...].unsqueeze(1) # pivot camera 

        if debug:
            set_trace()
        ###############################
        # TEMPORAL FEATURE ECTRACTION #   
        ###############################
        if self.use_motion_extractor:
            if self.motion_extractor_from == 'rgb':
                aux_images = images_batch[:,:self.pivot_index]
                if self.resize_images_for_me:
                    aux_images = F.interpolate(aux_images.view(-1, 3, *image_shape),size=self.images_me_target_size,  mode='bilinear')
                    aux_images = aux_images.view(batch_size, -1, 3, *self.images_me_target_size)
                if return_me_vector:    
                    style_vector, style_vector_me  = self.motion_extractor(aux_images.transpose(1,2), return_me_vector=return_me_vector)
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
        else:    
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
                # encoded_features = torch.transpose(encoded_features.unsqueeze(-1), 1,5).squeeze(1) # make time-dimension new z-coordinate
                encoded_features = torch.transpose(encoded_features, 1,2) # [batch_size, encoded_fetures_dim[0], dt-1, encoded_fetures_dim[1:]]
            style_vector = self.features_sequence_to_vector(encoded_features, device=device) # [batch_size, style_vector_dim]
        
        if debug:
            set_trace()
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
                    
        if self.use_gt_pelvis:
            tri_keypoints_3d = torch.from_numpy(np.array(batch['keypoints_3d'])[...,:3]).type(torch.float).to(device)
            if tri_keypoints_3d.dim() == 4:
                self.keypoints_per_frame = True
            elif tri_keypoints_3d.dim() == 3:
                self.keypoints_per_frame = False
        else:
            raise RuntimeError('In absence of precalculated pelvis or gt pelvis, self.use_volumetric_pelvis should be True') 
        

        if debug:
            set_trace()    
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

        if debug:
            set_trace()      
        # lift each feature-map to distinct volume and aggregate 
        unproj_features = unproject_heatmaps(pivot_features,  
                                            proj_matricies_batch, 
                                            coord_volumes, 
                                            volume_aggregation_method=self.volume_aggregation_method,
                                            vol_confidences=vol_confidences
                                            )
        if not self.temporal_condition_type == 'adain':
            if self.spade_broadcasting_type == 'unprojecting':
                    style_vector_volumes = unproject_heatmaps(style_vector.unsqueeze(1),  
                                                             proj_matricies_batch, 
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

        if debug:
            set_trace()      
        ##########################
        # VOLUMES FEEDING TO V2V #   
        ##########################         
        if self.temporal_condition_type == 'adain':
            volumes = self.volume_net(unproj_features, params=style_vector)
        elif self.temporal_condition_type == 'spade':       
            volumes = self.volume_net(unproj_features, params=style_vector_volumes)
        elif self.temporal_condition_type == 'stack': 
            volumes = self.volume_net(torch.cat([unproj_features, style_vector_volumes], 1))
        else:
            raise RuntimeError('Wrong self.temporal_condition_type, should be in [`adain`, `stack`, `spade`]')    
                
        # integral 3d
        vol_keypoints_3d, volumes = integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                         coord_volumes,
                                                                         softmax=self.volume_softmax)    

        if debug:
            set_trace()
        decoded_features = None
        if self.use_style_decoder:
            if self.style_decoder_type == 'v2v':
                decoded_features = self.style_decoder(style_vector)
                decoded_features = torch.transpose(decoded_features,1,2)

            elif self.style_decoder_type == 'lstm':
                decoded_features = self.style_decoder(style_vector,
                                                      original_pivot_features)

        if self.use_style_pose_lstm_loss:
            lstm_volumes = self.style_pose_lstm_loss_decoder(style_vector, volumes)
            time = lstm_volumes.shape[1]
            lstm_volumes = lstm_volumes.view(-1, *lstm_volumes.shape[2:])
            lstm_coord_volumes = lstm_coord_volumes.view(-1, *lstm_coord_volumes.shape[2:])
            lstm_keypoints_3d, _ = integrate_tensor_3d_with_coordinates(lstm_volumes * self.volume_multiplier,
                                                                        lstm_coord_volumes,
                                                                        softmax=self.volume_softmax)

            lstm_keypoints_3d = lstm_keypoints_3d.view(batch_size, time, *lstm_keypoints_3d.shape[1:])
            vol_keypoints_3d = [vol_keypoints_3d, lstm_keypoints_3d]
            # torch.cat([vol_keypoints_3d.unsqueeze(1),lstm_keypoints_3d],1)

        if debug:
            set_trace()      
        return [vol_keypoints_3d,
                features_for_loss if self.use_style_decoder else None,
                volumes,
                vol_confidences,
                None, # cuboids
                coord_volumes,
                base_points,
                [style_vector, style_vector_me] if return_me_vector else style_vector,
                unproj_features,
                decoded_features
                ]















