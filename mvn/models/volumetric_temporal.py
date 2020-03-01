from copy import deepcopy
import numpy as np
import pickle
import random
from collections import defaultdict

import torch
from torch import nn

from mvn.models import pose_resnet, pose_hrnet
from time import time
from mvn.utils import op
from mvn.utils import multiview
from mvn.utils import img
from mvn.utils import misc
from mvn.utils import volumetric
from mvn.models.v2v import V2VModel, V2VModel_v2, SPADE
from mvn.models.temporal import Seq2VecRNN,\
                                Seq2VecCNN, \
                                Seq2VecRNN2D, \
                                Seq2VecCNN2D, \
                                FeaturesEncoder_Bottleneck,\
                                FeaturesEncoder_DenseNet, \
                                FeaturesEncoder_Features2D, \
                                FeaturesEncoder_Bottleneck2D

from IPython.core.debugger import set_trace
from mvn.utils.op import get_coord_volumes


CHANNELS_LIST = [32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128,\
                                  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,\
                                  128, 128, 128, 128, 128, 64, 64, 64, 32, 32, 32, 32, 32]


CHANNELS_LIST_v2 = [128,128,  128,128,\
                    32,32,32,  128,128,  32,32,32, 128,128,  64,64,64,  128,128,\
                    128,128,  128,128,  128,128,  256,256,256,  256,256,\
                    128,128,128,  128,  128,128,  128,  64,64,64,  64,  32,32,32,  32,  32,32,  32,\
                    32,32] # 32,32

def get_capacity(model):
    s_total = 0
    for param in model.parameters():
        s_total+=param.numel()
    return round(s_total / (10**6),2)

def description(model):
    for k, m in model._modules.items():
        print ('{}:  {}M params'.format(k,get_capacity(m)))

def update_camera(batch, batch_size, image_shape, features_shape, dt, device):
    # change camera intrinsics
    new_cameras = deepcopy(batch['cameras'])
    for view_i in range(dt):
        for batch_i in range(batch_size):
            new_cameras[view_i][batch_i].update_after_resize(image_shape, features_shape)

    proj_matricies_batch = torch.stack([torch.stack([torch.from_numpy(camera.projection) \
                                        for camera in camera_batch], dim=0) \
                                        for camera_batch in new_cameras], dim=0).transpose(1, 0) 
    return proj_matricies_batch.float().to(device) # (batch_size, dt, 3, 4)
 

def get_encoder(encoder_type, 
                backbone_type,
                encoded_feature_space, 
                upscale_bottleneck,
                capacity=2, 
                spatial_dimension=1, 
                encoder_normalization_type='batch_norm'):
    
    assert spatial_dimension in [1,2], 'Wrong spatial_dimension! Only 1 and 2 are supported'
    encoder_input_channels = {'features':{'resnet152':256},
                              'backbone':{'resnet152':2048}}[encoder_type][backbone_type]



    if encoder_type == "backbone":
        if spatial_dimension == 1:
            return  FeaturesEncoder_Bottleneck(encoded_feature_space,
                                               encoder_input_channels,     
                                               C = capacity, 
                                               normalization_type=encoder_normalization_type)
        else: #spatial_dimension == 2:

            input_size, target_size = {'resnet152':[12,96]}[backbone_type] 

            return  FeaturesEncoder_Bottleneck2D(encoded_feature_space,
                                                 encoder_input_channels,
                                                 C=capacity, 
                                                 normalization_type=encoder_normalization_type, 
                                                 upscale=upscale_bottleneck, 
                                                 input_size=input_size, 
                                                 target_size=target_size,
                                                 upscale_kernel_size=2)
    elif encoder_type == "features":
        if spatial_dimension == 1:
            raise NotImplementedError()
        else: #spatial_dimension == 2:
            return  FeaturesEncoder_Features2D(encoder_input_channels,
                                               encoded_feature_space,
                                               C = capacity, 
                                               normalization_type=encoder_normalization_type)
    else:
        raise RuntimeError('Wrong encoder_type! Only `features` and `backbone` are supported')


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
        self.style_grad_for_backbone = config.model.style_grad_for_backbone
        self.include_pivot = config.model.include_pivot
        self.use_auxilary_backbone = hasattr(config.model, 'auxilary_backbone')
        
        # modules dimensions
        self.volume_features_dim = config.model.volume_features_dim
        self.style_vector_dim = config.model.style_vector_dim
        self.encoded_feature_space = config.model.encoded_feature_space
        
        if self.temporal_condition_type == 'adain':
            assert self.v2v_normalization_type in ['adain','ada-group_norm','group-ada_norm']

        elif self.temporal_condition_type == 'stack':
            assert self.v2v_normalization_type in ['group_norm','batch_norm']
        else:
            raise RuntimeError('Wrong self.temporal_condition_type, supported: [`adain`, `stack`]')        
            
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
        else:
            raise RuntimeError('Wrong features_sequence_to_vector type')

          
        self.encoder = get_encoder(self.encoder_type,
                                   config.model.backbone.name,
                                   self.encoded_feature_space,
                                   self.upscale_bottleneck,
                                   capacity = self.encoder_capacity,
                                   spatial_dimension = 2 if self.f2v_type[-2:] == '2d' else 1,
                                   encoder_normalization_type = self.encoder_normalization_type)
        

        v2v_input_features_dim = self.volume_features_dim if self.temporal_condition_type == 'adain' else \
                                 (self.volume_features_dim + self.style_vector_dim)

        self.volume_net = {'v1':V2VModel,
                           'v2':V2VModel_v2}[self.v2v_type](v2v_input_features_dim,
                                                           self.num_joints,
                                                           normalization_type=self.v2v_normalization_type,
                                                           volume_size=self.volume_size)

        if self.temporal_condition_type == 'adain':
            channels_list = {'v1':CHANNELS_LIST,
                             'v2':CHANNELS_LIST_v2}[self.v2v_type]    

            if self.f2v_type[-2:] == '2d':
                # spatial adaptive denormalization
                self.affine_mappings = nn.ModuleList([SPADE(self.style_vector_dim, 2*C, S) for C,S in channels_list.items()]) 
            else:
                # adaptive denormalization
                self.affine_mappings = nn.ModuleList([nn.Linear(self.style_vector_dim, 2*C) for C in channels_list])    

                              
            # for i in [33,36]:  
            #     for parameter in self.affine_mappings[i].parameters():
            #         parameter.requires_grad = False
           
        self.process_features = nn.Conv2d(256, self.volume_features_dim, 1)

        description(self)

    def forward(self, images_batch, batch, randomize_style=False):

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

        if self.encoder_type == 'backbone':
            aux_bottleneck = aux_bottleneck if self.style_grad_for_backbone else aux_bottleneck.detach()
            encoded_features = self.encoder(aux_bottleneck)
        elif self.encoder_type == 'features': 
            aux_features = aux_features if self.style_grad_for_backbone else aux_features.detach()
            encoded_features = self.encoder(aux_features)
        else:
            raise RuntimeError('Unknown encoder')    

        encoded_features = encoded_features.view(batch_size, -1, *encoded_features.shape[1:])
        style_vector = self.features_sequence_to_vector(encoded_features, device=device) # [batch_size, 512]
        
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
        volumes = op.unproject_heatmaps(pivot_features,  
                                        proj_matricies_batch, 
                                        coord_volumes, 
                                        volume_aggregation_method=self.volume_aggregation_method,
                                        vol_confidences=vol_confidences
                                        )

        #randomize
        if randomize_style:    
            idx = torch.randperm(style_vector.nelement())
            style_vector = style_vector.view(-1)[idx].view(style_vector.size())

        if self.temporal_condition_type == 'adain':
            adain_params = [affine_map(style_vector) for affine_map in self.affine_mappings]
            volumes = self.volume_net(volumes, params=adain_params)

        elif self.temporal_condition_type == 'stack':
            style_vector = style_vector.unsqueeze(1)
            style_vector_volumes = op.unproject_heatmaps(style_vector,  
                                                         proj_matricies_batch, 
                                                         coord_volumes, 
                                                         volume_aggregation_method=self.volume_aggregation_method,
                                                         vol_confidences=vol_confidences
                                                         )
            volumes = self.volume_net(torch.cat([volumes, style_vector_volumes], 1))

        else:
            raise RuntimeError('Wrong self.temporal_condition_type, should be in [`adain`, `stack`]')    
                
        # integral 3d
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                            coord_volumes,
                                                                            softmax=self.volume_softmax)
  
        return (vol_keypoints_3d,
                None if self.use_auxilary_backbone else features,
                volumes,
                vol_confidences,
                cuboids,
                coord_volumes,
                base_points
                )






class VolumetricTemporalGridDeformation(nn.Module):

    def __init__(self, config, device='cuda:0'):
        super().__init__()

        assert config.dataset.pivot_type == 'first', "pivot_type should be first"
        self.num_joints = config.model.backbone.num_joints

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size
        self.volume_aggregation_method = config.model.volume_aggregation_method

        self.cuboid_side = config.model.cuboid_side
        self.cuboid_multiplier = config.model.cuboid_multiplier if hasattr(config.model, "cuboid_multiplier") else 1.0
        self.rotation = config.model.rotation

        self.kind = config.model.kind

        self.use_precalculated_pelvis = config.model.use_precalculated_pelvis
        self.use_gt_pelvis = config.model.use_gt_pelvis

        assert self.use_precalculated_pelvis or self.use_gt_pelvis, 'One of the flags "use_<...>_pelvis" should be True'

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        # modules params
        self.style_grad_for_backbone = config.model.style_grad_for_backbone
        self.include_pivot = config.model.include_pivot
        self.style_net = config.model.style_net
        self.use_auxilary_backbone = hasattr(config.model, 'auxilary_backbone')

        # modules dimensions
        self.volume_features_dim = config.model.volume_features_dim
        self.style_vector_dim = config.model.style_vector_dim
        self.f2v_intermediate_channels = config.model.f2v_intermediate_channels
        self.normalization_type = config.model.normalization_type
        assert self.normalization_type not in ['adain','ada-group_norm','group-ada_norm']
        self.max_cell_size_multiplier = config.model.max_cell_size_multiplier
        self.dt = config.dataset.dt
        self.volume_size  = config.model.volume_size

        # modules
        self.backbone = pose_resnet.get_pose_net(config.model.backbone,
                                                 device=device,
                                                 strict=True)
        
        if self.use_auxilary_backbone:
            print ('Using auxilary backbone...')
            self.auxilary_backbone = pose_resnet.get_pose_net(config.model.auxilary_backbone,
                                                              device=device,
                                                              strict=True)  
        else:
            print ('{} backbone is used...'.format(config.model.backbone.name))

        if self.style_net == 'lstm_2d':
            self.features_sequence_to_vector = Seq2VecRNN2D(input_features_dim = 256,
                                                            output_features_dim =  self.style_vector_dim,
                                                            hidden_dim = self.f2v_intermediate_channels)
        elif self.style_net == 'cnn_2d':
            self.features_sequence_to_vector = Seq2VecCNN2D(input_features_dim=256, 
                                                            output_features_dim = self.style_vector_dim,
                                                            dt = self.dt-1,
                                                            intermediate_channel = self.f2v_intermediate_channels,
                                                            normalization_type=self.normalization_type)
        else:
            raise RuntimeError('Wrong style_net `lstm_2d` and `cnn_2d` are supported')    

        self.volume_net = V2VModel(self.volume_features_dim,
                                   self.num_joints,
                                   normalization_type=self.normalization_type,
                                   volume_size=self.volume_size)

        self.grid_deformator = V2VModel(self.style_vector_dim,
                                       3,
                                       normalization_type=self.normalization_type,
                                       volume_size=self.volume_size)

        self.process_features = nn.Conv2d(256, self.volume_features_dim, 1)

    def forward(self, images_batch, batch):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]

        if self.use_auxilary_backbone:

            pivot_images_batch = images_batch[:,-1,...]

            aux_images_batch = images_batch if self.include_pivot else images_batch[:,:-1,...]
            aux_images_batch = aux_images_batch.view(-1, 3, *image_shape)

            aux_heatmaps, aux_features, _, aux_vol_confidences, aux_bottleneck = self.auxilary_backbone(aux_images_batch)
            pivot_heatmaps, pivot_features, pivot_alg_confidences, vol_confidences, pivot_bottleneck = self.backbone(pivot_images_batch)

            features_shape = pivot_features.shape[-2:]
            features_channels = pivot_features.shape[1]
            bottleneck_shape = aux_bottleneck.shape[-2:]
            bottleneck_channels = aux_bottleneck.shape[1]

            pivot_features = self.process_features(pivot_features).unsqueeze(1)
            aux_features = aux_features.view(batch_size, -1, *aux_features.shape[-3:])

        else:    
            # forward backbone
            heatmaps, features, _, vol_confidences, bottleneck = self.backbone(images_batch.view(-1, 3, *image_shape))
            
            # extract aux_features
            features_shape = features.shape[-2:]
            features_channels = features.shape[1]
            features = features.view(batch_size, dt, features_channels, *features_shape)
            
            pivot_features = features[:,-1,...]
            pivot_features = self.process_features(pivot_features).unsqueeze(1)
            aux_features = features if self.include_pivot else features[:,:-1,...].contiguous()
        

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(dt):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, features_shape)

        proj_matricies_batch = torch.stack([torch.stack([torch.from_numpy(camera.projection) \
                                            for camera in camera_batch], dim=0) \
                                            for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, dt, 3, 4)
        proj_matricies_batch = proj_matricies_batch.float().to(device)
        proj_matricies_batch = proj_matricies_batch[:,-1,...].unsqueeze(1) # pivot camera 

        if not self.style_grad_for_backbone:
            aux_features = aux_features.detach()
        style_features = self.features_sequence_to_vector(aux_features, device=device).unsqueeze(1) # [batch_size,1, 32, 96,96]
    
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
        
        # lift each featuremap to distinct volume and aggregate 
        volumes = op.unproject_heatmaps(pivot_features,  
                                        proj_matricies_batch, 
                                        coord_volumes, 
                                        volume_aggregation_method=self.volume_aggregation_method,
                                        vol_confidences=vol_confidences
                                        )

        volumes_aux = op.unproject_heatmaps(style_features,  
                                            proj_matricies_batch, 
                                            coord_volumes, 
                                            volume_aggregation_method=self.volume_aggregation_method,
                                            vol_confidences=vol_confidences
                                            )


        if self.volume_aggregation_method == 'no_aggregation':
            volumes = torch.cat(volumes, 0)
            volumes_aux = torch.cat(volumes_aux, 0)

        volumes = volumes.view(batch_size, 
                               -1, # features
                               self.volume_size, 
                               self.volume_size, 
                               self.volume_size)

        volumes = self.volume_net(volumes)
        cell_size = self.cuboid_side * self.volume_size
        grid_offsets = self.grid_deformator(volumes_aux)
        grid_offsets = grid_offsets.transpose(4,1)
        grid_offsets = grid_offsets.sigmoid() * grid_offsets.sign() * cell_size * self.max_cell_size_multiplier


        # integral 3d
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                            coord_volumes + grid_offsets, 
                                                                            softmax=self.volume_softmax)

        return (vol_keypoints_3d,
                None if self.use_auxilary_backbone else features,
                volumes,
                vol_confidences,
                cuboids,
                coord_volumes,
                base_points,
                grid_offsets
                )


















