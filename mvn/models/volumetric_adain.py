import numpy as np
import pickle
import random
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F


from mvn.models import pose_resnet, pose_hrnet
from time import time
from mvn.utils.op import get_coord_volumes, unproject_heatmaps, integrate_tensor_3d_with_coordinates
from mvn.utils.multiview import update_camera
from mvn.utils.misc import get_capacity, description
from mvn.utils import volumetric
from mvn.models.v2v import V2VModel, C3D
from mvn.models.v2v_models import V2VModel_v1
from mvn.models.temporal import Seq2VecRNN,\
                                Seq2VecCNN, \
                                Seq2VecRNN2D, \
                                Seq2VecCNN2D, \
                                get_encoder

from IPython.core.debugger import set_trace

STYLE_VECTOR_CONST = None


class VolumetricTemporalAdaINNet(nn.Module):

    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.kind = config.model.kind
        self.num_joints = config.model.backbone.num_joints
        self.dt  = config.dataset.dt
        self.pivot_index =  {'first':self.dt-1,
                            'intermediate':self.dt//2}[config.dataset.pivot_type]
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

        self.use_motion_extractor = config.model.use_motion_extractor if hasattr(config.model, 'use_motion_extractor') else False
        if self.use_motion_extractor:   
            self.motion_extractor_type = config.model.motion_extractor_type
            self.motion_extractor_path = config.model.motion_extractor_path
            self.motion_extractor_from = config.model.motion_extractor_from
            self.motion_extractor_poolings = config.model.poolings
            self.motion_extractor_layers = config.model.n_layers if hasattr(config.model, 'n_layers') else 5

        self.use_f2v_output_as_v2v_input = config.model.use_f2v_output_as_v2v_input if \
                                             hasattr(config.model, 'use_f2v_output_as_v2v_input') else False

        if self.use_f2v_output_as_v2v_input:
            assert self.temporal_condition_type == 'spade'                               
        
        # modules dimensions
        self.volume_features_dim = config.model.volume_features_dim
        self.style_vector_dim = config.model.style_vector_dim
        self.encoded_feature_space = config.model.encoded_feature_space

        if self.temporal_condition_type == 'stack':
            assert self.v2v_normalization_type in ['group_norm','batch_norm']
            
        # modules
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

        if self.use_motion_extractor:
            if self.motion_extractor_type == 'c3d':
                self.motion_extractor = C3D(self.motion_extractor_poolings, 
                                            self.motion_extractor_layers)   # [3,11,384,384] -> [512, 1, 13, 13]


                me_out_channels = {1:64, 2:128, 3:256, 4:512, 5:512}[self.motion_extractor_layers]

                weights_dict = torch.load(self.motion_extractor_path, map_location=device)
                model_dict = self.motion_extractor.state_dict()
                new_pretrained_state_dict = {}
                
                for k, v in weights_dict.items():
                    if k in model_dict:
                        new_pretrained_state_dict[k] = weights_dict[k]
                    else:    
                        print (k, 'hasnt been loaded in C3D')

                self.motion_extractor.load_state_dict(new_pretrained_state_dict)
                print (f'Successfully loaded pretrained weights for motion_extractor {self.motion_extractor_type}')
                self.me_postprocess = nn.Conv3d(me_out_channels,self.style_vector_dim,kernel_size=1)    
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
                                                             self.f2v_type == 'v2v')  else 1,
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
        self.STYLE_VECTOR_CONST=None
        description(self)


    def forward(self, 
                images_batch, 
                batch, 
                randomize_style=False, 
                return_style_vector=False, 
                const_style_vector=False,
                return_unproj_features=False):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]
        assert self.dt == dt

        if self.use_auxilary_backbone:

            pivot_images_batch = images_batch[:,self.pivot_index,...]

            aux_images_batch = images_batch if self.include_pivot else images_batch[:,self.aux_indexes,...].contiguous()
            aux_images_batch = aux_images_batch.view(-1, 3, *image_shape)

            aux_heatmaps, aux_features, _, _, aux_bottleneck = self.auxilary_backbone(aux_images_batch)
            pivot_heatmaps, pivot_features, pivot_alg_confidences, pivot_vol_confidences, pivot_bottleneck = self.backbone(pivot_images_batch)

            features_shape = pivot_features.shape[-2:]
            features_channels = pivot_features.shape[1]
            bottleneck_shape = aux_bottleneck.shape[-2:]
            bottleneck_channels = aux_bottleneck.shape[1]

            pivot_features = self.process_features(pivot_features).unsqueeze(1)
            if aux_features is not None:
                aux_features = aux_features.view(-1, *aux_features.shape[-3:])

        else:    
            # forward backbone
            heatmaps, features, _, vol_confidences, bottleneck = self.backbone(images_batch.view(-1, 3, *image_shape))
            
            # extract aux_features
            features_shape = features.shape[-2:]
            features_channels = features.shape[1]
            features = features.view(batch_size, dt, features_channels, *features_shape)
            
            pivot_features = features[:,self.pivot_index,...]
            pivot_features = self.process_features(pivot_features).unsqueeze(1)
            aux_features = features if self.include_pivot else features[:,self.aux_indexes,...].contiguous()
            aux_features = aux_features.view(-1, *aux_features.shape[-3:])

            # extract aux_bottleneck
            bottleneck_shape = bottleneck.shape[-2:]
            bottleneck_channels = bottleneck.shape[1]
            bottleneck = bottleneck.view(batch_size, dt, bottleneck_channels, *bottleneck_shape)
            aux_bottleneck = bottleneck if self.include_pivot else bottleneck[:,self.aux_indexes,...].contiguous()
            aux_bottleneck = aux_bottleneck.view(-1, bottleneck_channels, *bottleneck_shape)   

        proj_matricies_batch = update_camera(batch, batch_size, image_shape, features_shape, dt, device)
        proj_matricies_batch = proj_matricies_batch[:,self.pivot_index,...].unsqueeze(1) # pivot camera 

        ###############################
        # TEMPORAL FEATURE ECTRACTION #   
        ###############################

        if self.use_motion_extractor:
            if self.motion_extractor_from == 'rgb':
                style_vector = self.motion_extractor(images_batch.view(batch_size, 3, dt, *image_shape))
            elif self.motion_extractor_from == 'features':    
                style_vector = self.motion_extractor(aux_features.view(batch_size, dt-1, features_shape, *features_channels))
            elif self.motion_extractor_from == 'bottleneck':    
                style_vector = self.motion_extractor(aux_bottleneck.view(batch_size, dt-1, bottleneck_shape, *bottleneck_channels))    
            else:
                raise RuntimeError('Wrong `motion_extractor_from`')    

            style_vector = self.me_postprocess(style_vector) 

        else:    
            if self.encoder_type == 'backbone':
                aux_bottleneck = aux_bottleneck if self.style_grad_for_backbone else aux_bottleneck.detach()
                encoded_features = self.encoder(aux_bottleneck)
            elif self.encoder_type == 'features': 
                aux_features = aux_features if self.style_grad_for_backbone else aux_features.detach()
                encoded_features = self.encoder(aux_features)
            else:
                raise RuntimeError('Unknown encoder')    

            encoded_features = encoded_features.view(batch_size, -1, *encoded_features.shape[1:]) # [batch_size, dt-1, encoded_fetures_dim]
            if self.f2v_type == 'v2v':
                encoded_features = torch.transpose(encoded_features, 1,2) # [batch_size, encoded_fetures_dim[0], dt-1, encoded_fetures_dim[1:]]
            style_vector = self.features_sequence_to_vector(encoded_features, device=device) # [batch_size, style_vector_dim]
        
        # using for debugging 
        if randomize_style:
            idx = torch.randperm(style_vector.nelement())
            style_vector = style_vector.view(-1)[idx].view(style_vector.size())
        if const_style_vector:
            if self.STYLE_VECTOR_CONST is None:
                self.STYLE_VECTOR_CONST = style_vector.data
            else:
                style_vector = torch.tensor(self.STYLE_VECTOR_CONST).to(device)
                    

            # set_trace()        

        if self.use_gt_pelvis:
            tri_keypoints_3d = torch.from_numpy(np.array(batch['keypoints_3d'])).type(torch.float).to(device)
        else:
            raise RuntimeError('In absence of precalculated pelvis or gt pelvis, self.use_volumetric_pelvis should be True') 
       
        # amend coord_volumes position                                                         
        coord_volumes, cuboids, base_points = get_coord_volumes(self.kind, 
                                                                self.training, 
                                                                self.rotation,
                                                                self.cuboid_side,
                                                                self.volume_size, 
                                                                device,
                                                                keypoints=tri_keypoints_3d
                                                                )
        
        # lift each feature-map to distinct volume and aggregate 
        unproj_features = unproject_heatmaps(pivot_features,  
                                            proj_matricies_batch, 
                                            coord_volumes, 
                                            volume_aggregation_method=self.volume_aggregation_method,
                                            vol_confidences=vol_confidences
                                            )


        if self.temporal_condition_type == 'adain':
            volumes = self.volume_net(unproj_features, params=style_vector)

        elif self.temporal_condition_type == 'spade':
            if self.spade_broadcasting_type == 'unprojecting':
                style_vector = style_vector.unsqueeze(1)
                style_vector_volumes = unproject_heatmaps(style_vector,  
                                                         proj_matricies_batch, 
                                                         coord_volumes, 
                                                         volume_aggregation_method=self.volume_aggregation_method,
                                                         vol_confidences=vol_confidences
                                                         )
            elif self.spade_broadcasting_type == 'interpolate':
                unsqueeze_style = not (self.f2v_type == 'v2v' or self.use_motion_extractor)
                style_vector_volumes = F.interpolate(style_vector.unsqueeze(-1) if unsqueeze_style else style_vector,
                                                     size=(self.volume_size,self.volume_size,self.volume_size), 
                                                     mode='trilinear')
                
            else:
                raise KeyError('Unknown spade_broadcasting_type')       
                
            volumes = self.volume_net(unproj_features, params=style_vector_volumes)

        elif self.temporal_condition_type in ['stack', 'stack_poses'] :
            
            if self.f2v_type == 'v2v':
                style_vector_volumes = F.interpolate(style_vector,
                                                     size=(self.volume_size,self.volume_size,self.volume_size), 
                                                     mode='trilinear')

            else:    
                style_vector = style_vector.unsqueeze(1)
                style_vector_volumes = unproject_heatmaps(style_vector,  
                                                             proj_matricies_batch, 
                                                             coord_volumes, 
                                                             volume_aggregation_method=self.volume_aggregation_method,
                                                             vol_confidences=vol_confidences
                                                             )

                if self.temporal_condition_type == 'stack_poses':
                    assert (self.pivot_index == self.dt-1)
                    before_pivot_features = aux_features.view(batch_size, dt-1, *aux_features.shape[1:])[:,self.pivot_index - 1].contiguous()
                    before_pivot_features = self.process_features(before_pivot_features)
                    approximated_unproj_features = unproject_heatmaps(before_pivot_features,  
                                                                      proj_matricies_batch, 
                                                                      coord_volumes, 
                                                                      volume_aggregation_method=self.volume_aggregation_method,
                                                                      vol_confidences=vol_confidences
                                                                      )

                    style_vector_volumes = style_vector_volumes + approximated_unproj_features

            volumes = self.volume_net(torch.cat([unproj_features, style_vector_volumes], 1))

        else:
            raise RuntimeError('Wrong self.temporal_condition_type, should be in [`adain`, `stack`, `spade`]')    
                
        # integral 3d
        vol_keypoints_3d, volumes = integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                            coord_volumes,
                                                                            softmax=self.volume_softmax)
        
        return [vol_keypoints_3d,
                None if self.use_auxilary_backbone else features,
                volumes,
                vol_confidences,
                cuboids,
                coord_volumes,
                base_points
                ] + ([style_vector] if return_style_vector else []) + \
                ([unproj_features] if return_unproj_features else [])



class VolumetricTemporalFRAdaINNet(nn.Module):

    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.kind = config.model.kind
        self.num_joints = config.model.backbone.num_joints
        self.dt  = config.dataset.dt
        self.pivot_index =  {'first':self.dt-1,
                            'intermediate':self.dt//2}[config.dataset.pivot_type]
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
        assert self.temporal_condition_type == 'spade'
        
        # modules dimensions
        self.volume_features_dim = config.model.volume_features_dim
        self.style_vector_dim = config.model.style_vector_dim
        self.encoded_feature_space = config.model.encoded_feature_space
            
        # modules
        self.backbone = pose_resnet.get_pose_net(config.model.backbone,
                                                 device=device,
                                                 strict=True)

        if self.f2v_type == 'lstm2d':
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
        else:
            raise RuntimeError('Wrong features_sequence_to_vector type')

          
        self.encoder = get_encoder(self.encoder_type,
                                   config.model.backbone.name,
                                   self.encoded_feature_space,
                                   self.upscale_bottleneck,
                                   capacity = self.encoder_capacity,
                                   spatial_dimension = 2 if self.f2v_type[-2:] == '2d' else 1,
                                   encoder_normalization_type = self.encoder_normalization_type)

        if self.v2v_type == 'v1':
            self.volume_net = V2VModel_v1(self.style_vector_dim,
                                          self.num_joints,
                                          normalization_type=self.v2v_normalization_type,
                                          volume_size=self.volume_size,
                                          temporal_condition_type = self.temporal_condition_type,
                                          style_vector_dim = self.volume_features_dim)

        elif self.v2v_type == 'conf':
            self.volume_net = V2VModel(self.style_vector_dim,
                                       self.num_joints,
                                       v2v_normalization_type=self.v2v_normalization_type,
                                       config=config.model.v2v_configuration,
                                       temporal_condition_type=self.temporal_condition_type,
                                       style_vector_dim=self.volume_features_dim)
            
        self.process_features = nn.Conv2d(256, self.volume_features_dim, 1)

        description(self)

    def forward(self, images_batch, batch, randomize_style=False):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]
        assert self.dt == dt

        # forward backbone
        heatmaps, features, _, vol_confidences, bottleneck = self.backbone(images_batch.view(-1, 3, *image_shape))
        
        # extract aux_features
        features_shape = features.shape[-2:]
        features_channels = features.shape[1]
        features = features.view(batch_size, dt, features_channels, *features_shape)
        
        pivot_features = features[:,self.pivot_index,...]
        pivot_features = self.process_features(pivot_features)
        aux_features = features if self.include_pivot else features[:,self.aux_indexes,...].contiguous()
        aux_features = aux_features.view(-1, *aux_features.shape[-3:])

        # extract aux_bottleneck
        bottleneck_shape = bottleneck.shape[-2:]
        bottleneck_channels = bottleneck.shape[1]
        bottleneck = bottleneck.view(batch_size, dt, bottleneck_channels, *bottleneck_shape)
        aux_bottleneck = bottleneck if self.include_pivot else bottleneck[:,self.aux_indexes,...].contiguous()
        aux_bottleneck = aux_bottleneck.view(-1, bottleneck_channels, *bottleneck_shape)
        
        proj_matricies_batch = update_camera(batch, batch_size, image_shape, features_shape, dt, device)
        proj_matricies_batch = proj_matricies_batch[:,self.pivot_index,...].unsqueeze(1) # pivot camera 

        if self.encoder_type == 'backbone':
            aux_bottleneck = aux_bottleneck if self.style_grad_for_backbone else aux_bottleneck.detach()
            encoded_features = self.encoder(aux_bottleneck)
        elif self.encoder_type == 'features': 
            aux_features = aux_features if self.style_grad_for_backbone else aux_features.detach()
            encoded_features = self.encoder(aux_features)
        else:
            raise RuntimeError('Unknown encoder')    

        encoded_features = encoded_features.view(batch_size, -1, *encoded_features.shape[1:]) # [batch_size, dt-1, encoded_fetures_dim]
        style_vector = self.features_sequence_to_vector(encoded_features, device=device) # [batch_size, 1, style_vector_dim]
        
        if self.use_gt_pelvis:
            tri_keypoints_3d = torch.from_numpy(np.array(batch['keypoints_3d'])).type(torch.float).to(device)
        else:
            raise RuntimeError('In absence of precalculated pelvis or gt pelvis, self.use_volumetric_pelvis should be True') 
        
        # amend coord_volumes position                                                         
        coord_volumes, cuboids, base_points = get_coord_volumes(self.kind, 
                                                                self.training, 
                                                                self.rotation,
                                                                self.cuboid_side,
                                                                self.volume_size, 
                                                                device,
                                                                keypoints=tri_keypoints_3d
                                                                )
        
        # lift each feature-map to distinct volume and aggregate 
        volumes = unproject_heatmaps(style_vector.unsqueeze(1),  
                                     proj_matricies_batch, 
                                     coord_volumes, 
                                     volume_aggregation_method=self.volume_aggregation_method,
                                     vol_confidences=vol_confidences
                                     )

        if self.spade_broadcasting_type == 'unprojecting':
            style_vector_volumes = unproject_heatmaps(pivot_features.unsqueeze(1),  
                                                     proj_matricies_batch, 
                                                     coord_volumes, 
                                                     volume_aggregation_method=self.volume_aggregation_method,
                                                     vol_confidences=vol_confidences
                                                     )
        elif self.spade_broadcasting_type == 'interpolate':
            style_vector_volumes = F.interpolate(pivot_features.unsqueeze(-1),
                                         size=(self.volume_size,self.volume_size,self.volume_size), 
                                         mode='trilinear')
            
        else:
            raise KeyError('Unknown spade_broadcasting_type') 


        volumes = self.volume_net(volumes, params=style_vector_volumes)

        vol_keypoints_3d, volumes = integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                            coord_volumes,
                                                                            softmax=self.volume_softmax)
  
        return (vol_keypoints_3d,
                features,
                volumes,
                vol_confidences,
                cuboids,
                coord_volumes,
                base_points
                )











