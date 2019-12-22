from copy import deepcopy
import numpy as np
import pickle
import random

import torch
from torch import nn

from mvn.models import pose_resnet
from time import time
from mvn.utils import op
from mvn.utils import multiview
from mvn.utils import img
from mvn.utils import misc
from mvn.utils import volumetric
from mvn.models.v2v import V2VModel, V2VModelAdaIN, V2VModelAdaIN_MiddleVector
from mvn.models.temporal import Seq2VecRNN,\
                                FeaturesAR_RNN,\
                                FeaturesAR_CNN1D,\
                                FeaturesAR_CNN2D_UNet,\
                                FeaturesAR_CNN2D_ResNet,\
                                FeaturesEncoder_Bottleneck,\
                                FeaturesEncoder_DenseNet

from IPython.core.debugger import set_trace
from mvn.utils.op import get_coord_volumes

CHANNELS_LIST = [16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128,\
                                  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,\
                                  128, 128, 128, 128, 128, 64, 64, 64, 32, 32, 32, 32, 32]


class VolumetricTemporalNet(nn.Module):

    '''
    The model is designed to work with `dt` number of consecutive frames
    '''

    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.dt = config.dataset.dt
        self.num_joints = config.model.backbone.num_joints
        self.volume_aggregation_method = config.model.volume_aggregation_method # if hasattr(config.model, 'volume_aggregation_method') else None

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size

        self.cuboid_side = config.model.cuboid_side
        self.cuboid_multiplier = config.model.cuboid_multiplier if hasattr(config.model, "cuboid_multiplier") else 1.0
        self.rotation = config.model.rotation if hasattr(config.model, "rotation") else False

        self.process_frames_independently = config.model.process_frames_independently if hasattr(config.model, 'process_frames_independently') else False

        self.kind = config.model.kind

        self.use_precalculated_pelvis = config.model.use_precalculated_pelvis if hasattr(config.model, "use_precalculated_pelvis") else False
        self.use_gt_pelvis = config.model.use_gt_pelvis if hasattr(config.model, "use_gt_pelvis") else False
        self.use_volumetric_pelvis = config.model.use_volumetric_pelvis if hasattr(config.model, "use_volumetric_pelvis") else False
        self.use_separate_v2v_for_basepoint = config.model.use_separate_v2v_for_basepoint if hasattr(config.model, "use_separate_v2v_for_basepoint") else False

        assert self.use_precalculated_pelvis or self.use_volumetric_pelvis or self.use_gt_pelvis, 'One of the flags "use_<...>_pelvis" should be True'

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True

        # modules
        self.volumes_multipliers = config.model.volumes_multipliers if hasattr(config.model, "volumes_multipliers") else [1.]*self.dt
        assert len(self.volumes_multipliers) == self.dt
        self.backbone = pose_resnet.get_pose_net(config.model.backbone)
        self.return_heatmaps = config.model.backbone.return_heatmaps if hasattr(config.model.backbone, 'return_heatmaps') else False  
        self.features_channels = config.model.features_channels if hasattr(config.model, 'features_channels') else 32  
        self.process_features = nn.Sequential(
            nn.Conv2d(256, self.features_channels, 1)
        )
        self.volume_net = {
            "channel_stack":V2VModel(self.features_channels*self.dt, self.num_joints)
            # "lstm":V2VModelLSTM(self.features_channels, self.num_joints)
        }[config.model.v2v_type]

    def forward(self, images_batch, batch, root_keypoints=None):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]
        assert self.dt == dt

        # reshape for backbone forward
        images_batch = images_batch.view(-1, 3, *image_shape)

        # forward backbone

        heatmaps, features, _, vol_confidences , _= self.backbone(images_batch)
        features = self.process_features(features)

        features_shape = features.shape[-2:]

        # reshape back
        images_batch = images_batch.view(batch_size, dt, 3, *image_shape)
        features = features.view(batch_size, dt, self.features_channels,*features_shape)

        if self.volume_aggregation_method.startswith('conf'):
            vol_confidences = vol_confidences.view(batch_size, dt, *vol_confidences.shape[1:])
            # norm vol confidences
            if self.volume_aggregation_method == 'conf_norm':
                vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(dt):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, features_shape)

        proj_matricies_batch = torch.stack([torch.stack([torch.from_numpy(camera.projection) \
                                            for camera in camera_batch], dim=0) \
                                            for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, dt, 3, 4)

        proj_matricies_batch = proj_matricies_batch.float().to(device)
        
        if self.use_precalculated_pelvis:
            tri_keypoints_3d = torch.from_numpy(np.array(batch['pred_keypoints_3d'])).type(torch.float).to(device)
        elif self.use_gt_pelvis:
            tri_keypoints_3d = torch.from_numpy(np.array(batch['keypoints_3d'])).type(torch.float).to(device)
        else:
            raise RuntimeError('In absence of precalculated pelvis or gt pelvis should be True') 
       
        # amend coord_volumes position                                                         
        coord_volumes, cuboids, base_points = get_coord_volumes(self.kind, 
                                                                self.training, 
                                                                self.rotation,
                                                                self.cuboid_side,
                                                                self.volume_size, 
                                                                device,
                                                                keypoints=tri_keypoints_3d
                                                                )

        if self.process_frames_independently:
            coord_volumes = coord_volumes.view(-1, *coord_volumes.shape[-4:])
            proj_matricies_batch = proj_matricies_batch.view(-1, 1, *proj_matricies_batch.shape[-2:])
            features = features.view(batch_size*dt, 1, self.features_channels,*features_shape)

        # lift each featuremap to distinct volume and aggregate 
        volumes = op.unproject_heatmaps(features,  
                                        proj_matricies_batch, 
                                        coord_volumes, 
                                        volume_aggregation_method=self.volume_aggregation_method,
                                        vol_confidences=vol_confidences,
                                        volumes_multipliers = self.volumes_multipliers
                                        )

        # cat along view dimension
        if self.volume_aggregation_method == 'no_aggregation':        
            volumes = torch.stack(volumes, 0) # [batch_size, dt, ...]
            volumes = volumes.view(batch_size, dt*self.features_channels, self.volume_size, self.volume_size, self.volume_size)  
        # inference
        volumes = self.volume_net(volumes)
        # integral 3d
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return (vol_keypoints_3d,
                features,
                volumes,
                vol_confidences,
                cuboids,
                coord_volumes,
                base_points
                )



class VolumetricLSTMAdaINNet(nn.Module):

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
        self.rotation = config.model.rotation if hasattr(config.model, "rotation") else False

        self.kind = config.model.kind

        self.use_precalculated_pelvis = config.model.use_precalculated_pelvis if hasattr(config.model, "use_precalculated_pelvis") else False
        self.use_gt_pelvis = config.model.use_gt_pelvis if hasattr(config.model, "use_gt_pelvis") else False
        self.use_volumetric_pelvis = config.model.use_volumetric_pelvis if hasattr(config.model, "use_volumetric_pelvis") else False
        self.use_separate_v2v_for_basepoint = config.model.use_separate_v2v_for_basepoint if hasattr(config.model, "use_separate_v2v_for_basepoint") else False

        assert self.use_precalculated_pelvis or self.use_gt_pelvis, 'One of the flags "use_<...>_pelvis" should be True'

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        # modules params
        self.features_dim = config.model.features_dim if hasattr(config.model, 'features_dim') else 256
        self.style_vector_dim = config.model.style_vector_dim if hasattr(config.model, 'style_vector_dim') else 256
        self.features_regressor_base_channels = config.model.features_regressor_base_channels if hasattr(config.model, 'features_regressor_base_channels') else 8
        self.adain_type = config.model.adain_type
        self.style_grad_for_backbone = config.model.style_grad_for_backbone
        self.encoder_type = config.model.encoder_type
        self.pretrained_encoder = config.model.pretrained_encoder
        self.encoded_feature_space = config.model.encoded_feature_space
        self.encoder_capacity_multiplier = config.model.encoder_capacity_multiplier

        # modules
        self.backbone = pose_resnet.get_pose_net(config.model.backbone) 
        self.encoder = {
            # "custom":FeaturesEncoder(256, self.encoded_feature_space, pretrained = self.pretrained_encoder),
            "densenet":FeaturesEncoder_DenseNet(256, 
                                                self.encoded_feature_space, 
                                                pretrained = self.pretrained_encoder),
            "backbone":FeaturesEncoder_Bottleneck(self.encoded_feature_space,
                                                  C = self.encoder_capacity_multiplier)
        }[self.encoder_type]

        self.volume_net = {
            "all":V2VModelAdaIN(32, self.num_joints),
            "middle":V2VModelAdaIN_MiddleVector(32, self.num_joints)
        }[self.adain_type]

        self.features_sequence_to_vector = Seq2VecRNN(self.encoded_feature_space,
                                                      self.style_vector_dim)

        if self.adain_type == 'all':
            self.affine_mappings = nn.ModuleList([nn.Linear(self.style_vector_dim, 2*C) for C in CHANNELS_LIST]) # 51
        elif self.adain_type =='middle':
            self.affine_mappings = nn.ModuleList([nn.Linear(self.style_vector_dim, 2*128) for _ in range(2)])
        else:
            raise RuntimeError('Wrong adain_type') 
                
        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )

    def forward(self, images_batch, batch, root_keypoints=None):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]

        # reshape for backbone forward
        images_batch = images_batch.view(-1, 3, *image_shape)

        # forward backbone
        heatmaps, features, alg_confidences, vol_confidences, bottleneck = self.backbone(images_batch)

        # reshape back and take only last view (pivot)
        images_batch = images_batch.view(batch_size, dt, *images_batch.shape[1:])[:,-1,...].unsqueeze(1)
        
        # calcualte shapes
        features_shape = features.shape[-2:]
        features_channels = features.shape[1]

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(dt):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, features_shape)

        proj_matricies_batch = torch.stack([torch.stack([torch.from_numpy(camera.projection) \
                                            for camera in camera_batch], dim=0) \
                                            for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, dt, 3, 4)

        proj_matricies_batch = proj_matricies_batch.float().to(device)
        proj_matricies_batch = proj_matricies_batch[:,-1,...].unsqueeze(1) 

        features = features.view(batch_size, dt, features_channels, *features_shape)
        pivot_features = features[:,-1,...]
        style_features = features[:,:-1,...].contiguous()
        pivot_features = self.process_features(pivot_features).unsqueeze(1) # add fictive view

        if self.encoder_type == 'backbone':
            bottleneck_shape = bottleneck.shape[-2:]
            bottleneck_channels = bottleneck.shape[1]
            bottleneck = bottleneck.view(batch_size, dt, bottleneck_channels, *bottleneck_shape)
            bottleneck = bottleneck[:,:-1,...].contiguous()
            bottleneck = bottleneck.view(batch_size*(dt-1), bottleneck_channels, *bottleneck_shape)
            if not self.style_grad_for_backbone:
                bottleneck = bottleneck.detach()
            encoded_features = self.encoder(bottleneck)
        else:
            style_features = style_features.view(batch_size*(dt-1), features_channels, *features_shape)
            if self.style_grad_for_backbone:
                style_features = style_features.detach()
            encoded_features = self.encoder(style_features)

        encoded_features = encoded_features.view(batch_size, (dt-1), self.encoded_feature_space)
        style_vector = self.features_sequence_to_vector(encoded_features, device=device) # [batch_size, 512]
        
        if self.use_precalculated_pelvis:
            tri_keypoints_3d = torch.from_numpy(np.array(batch['pred_keypoints_3d'])).type(torch.float).to(device)

        elif self.use_gt_pelvis:
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
        if self.volume_aggregation_method == 'no_aggregation':
            volumes = torch.cat(volumes, 0)

        # inference
        adain_params = [affine_map(style_vector) for affine_map in self.affine_mappings]

        volumes = self.volume_net(volumes, adain_params=adain_params)
        # integral 3d
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return (vol_keypoints_3d,
                features,
                volumes,
                vol_confidences,
                cuboids,
                coord_volumes,
                base_points
                )


# FIX IT!
class VolumetricFRAdaINNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()
        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size
        self.volume_aggregation_method = config.model.volume_aggregation_method
        self.dt = config.dataset.dt 
        self.cuboid_side = config.model.cuboid_side
        self.cuboid_multiplier = config.model.cuboid_multiplier if hasattr(config.model, "cuboid_multiplier") else 1.0
        self.rotation = config.model.rotation if hasattr(config.model, "rotation") else False

        self.kind = config.model.kind

        self.use_precalculated_pelvis = config.model.use_precalculated_pelvis if hasattr(config.model, "use_precalculated_pelvis") else False
        self.use_gt_pelvis = config.model.use_gt_pelvis if hasattr(config.model, "use_gt_pelvis") else False
        self.use_volumetric_pelvis = config.model.use_volumetric_pelvis if hasattr(config.model, "use_volumetric_pelvis") else False
        self.use_separate_v2v_for_basepoint = config.model.use_separate_v2v_for_basepoint if hasattr(config.model, "use_separate_v2v_for_basepoint") else False

        assert self.use_precalculated_pelvis or self.use_gt_pelvis, 'One of the flags "use_<...>_pelvis" should be True'

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        # modules params
        self.intermediate_features_dim = config.model.intermediate_features_dim if hasattr(config.model, 'intermediate_features_dim') else 32
        self.num_joints = config.model.backbone.num_joints
        self.features_regressor_base_channels = config.model.features_regressor_base_channels if hasattr(config.model, 'features_regressor_base_channels') else 8
        self.adain_type = config.model.adain_type
        self.fr_grad_for_backbone = config.model.fr_grad_for_backbone if hasattr(config.model, 'fr_grad_for_backbone') else False
        
        # modules
        self.backbone = pose_resnet.get_pose_net(config.model.backbone)
        self.volume_net = {
            "all":V2VModelAdaIN(self.intermediate_features_dim, self.num_joints),
            "middle":V2VModelAdaIN_MiddleVector(self.intermediate_features_dim, self.num_joints)    
        }[self.adain_type]
        self.process_features = nn.Sequential(
            nn.Conv2d(256, self.intermediate_features_dim, 1)
        )
        self.features_regressor = {
            # "rnn": FeaturesAR_RNN(self.intermediate_features_dim, self.intermediate_features_dim),
            # "conv1d": FeaturesAR_CNN1D(self.intermediate_features_dim, self.intermediate_features_dim),
            "conv2d_unet": FeaturesAR_CNN2D_UNet(self.intermediate_features_dim*(self.dt-1),
                                                 self.intermediate_features_dim,
                                                 C = self.features_regressor_base_channels)
            # "conv2d_resnet": FeaturesAR_CNN2D_ResNet(self.intermediate_features_dim*self.dt, self.intermediate_features_dim)
        }[config.model.features_regressor]
                

    def forward(self, images_batch, batch, root_keypoints=None):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]

        # reshape for backbone forward
        images_batch = images_batch.view(-1, 3, *image_shape)

        # forward backbone
        heatmaps, features, alg_confidences, vol_confidences, _ = self.backbone(images_batch)
        features = self.process_features(features) # [bs, 256, 96, 96] -> [bs, 32, 96, 96] 
        
        # calcualte shapes
        features_shape = features.shape[-2:]
        features_channels = features.shape[1] # 32

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(dt):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, features_shape)

        proj_matricies_batch = torch.stack([torch.stack([torch.from_numpy(camera.projection) \
                                            for camera in camera_batch], dim=0) \
                                            for camera_batch in new_cameras], dim=0).transpose(1, 0)  
                                            #shape (batch_size, dt, 3, 4)

        proj_matricies_batch = proj_matricies_batch.float().to(device)
        proj_matricies_batch = proj_matricies_batch[:,-1,...].unsqueeze(1)

        # process features before lifting to volume
        features = features.view(batch_size, dt, features_channels, *features_shape)
        features_pivot = features[:,-1,...].unsqueeze(1)
        if not self.fr_grad_for_backbone:
            features_aux = features[:,:-1,...].view(batch_size, -1, *features_shape).detach()
        else:
            features_aux = features[:,:-1,...].view(batch_size, -1, *features_shape)
            

        features_pred = self.features_regressor(features_aux).unsqueeze(1)

        if self.use_precalculated_pelvis:
            tri_keypoints_3d = torch.from_numpy(np.array(batch['pred_keypoints_3d'])).type(torch.float).to(device)

        elif self.use_gt_pelvis:
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
        volumes_pred = op.unproject_heatmaps(features_pred,  
                                             proj_matricies_batch, 
                                             coord_volumes, 
                                             volume_aggregation_method=self.volume_aggregation_method,
                                             vol_confidences=vol_confidences
                                             )

        volumes = op.unproject_heatmaps(features_pivot,  
                                        proj_matricies_batch, 
                                        coord_volumes, 
                                        volume_aggregation_method=self.volume_aggregation_method,
                                        vol_confidences=vol_confidences
                                        )

        if self.volume_aggregation_method == 'no_aggregation':
            volumes = torch.cat(volumes, 0)
            volumes_pred = torch.cat(volumes_pred, 0)

        # inference
        if self.adain_type == 'all':
            volumes_stacked = self.volume_net(torch.cat([volumes, volumes_pred]), [None]*len(CHANNELS_LIST))
        elif self.adain_type == 'middle':
            volumes_stacked = self.volume_net(torch.cat([volumes, volumes_pred]))
            
        # integral 3d
        volumes = volumes_stacked[batch_size:]
        volumes_pred = volumes_stacked[batch_size:]

        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)
        vol_keypoints_3d_pred, volumes_pred = op.integrate_tensor_3d_with_coordinates(volumes_pred * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)


        return (vol_keypoints_3d,
                vol_keypoints_3d_pred,
                features_pivot,
                features_pred,
                volumes,
                volumes_pred,
                vol_confidences,
                cuboids,
                coord_volumes,
                base_points
                )        
        