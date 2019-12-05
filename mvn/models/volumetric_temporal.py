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
                                FeaturesAR_CNN2D_ResNet

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
        self.backbone = pose_resnet.get_pose_net(config.model.backbone)
        self.return_heatmaps = config.model.backbone.return_heatmaps if hasattr(config.model.backbone, 'return_heatmaps') else False  

        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )
        self.volume_net = {
            "original":V2VModel(32, self.num_joints),
            "lstm":V2VModelLSTM(32, self.num_joints)
        }[config.model.v2v_type]

    def forward(self, images_batch, batch, root_keypoints=None):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]

        # reshape for backbone forward
        images_batch = images_batch.view(-1, *images_batch.shape[2:])

        # forward backbone
        heatmaps, features, _, vol_confidences = self.backbone(images_batch)

        # reshape back
        images_batch = images_batch.view(batch_size, dt, *images_batch.shape[1:])
        heatmaps = heatmaps.view(batch_size, dt, *heatmaps.shape[1:]) if self.return_heatmaps else heatmaps
        features = features.view(batch_size, dt, *features.shape[1:])

        # calcualte shapes
        image_shape, features_shape = tuple(images_batch.shape[3:]), tuple(features.shape[3:])

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
        
        # process features before lifting to volume
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.unsqueeze(1) # [bs*dt, 1, features_shape]
                    
        if self.use_precalculated_pelvis:
            tri_keypoints_3d = torch.from_numpy(np.array(batch['pred_keypoints_3d'])).type(torch.float).to(device)

        elif self.use_gt_pelvis:
            tri_keypoints_3d = torch.from_numpy(np.array(batch['keypoints_3d'])).type(torch.float).to(device)

        elif self.use_volumetric_pelvis:

            coord_volumes, _, _ = get_coord_volumes(self.kind, 
                                                    self.training, 
                                                    self.rotation,
                                                    self.cuboid_side*self.cuboid_multiplier,
                                                    self.volume_size, 
                                                    device, 
                                                    batch_size = batch_size,
                                                    dt = dt
                                                    )

            volumes = op.unproject_heatmaps(features,
                                              proj_matricies_batch, 
                                              coord_volumes, 
                                              volume_aggregation_method=self.volume_aggregation_method, 
                                              vol_confidences=vol_confidences)
        
            # set True to save intermediate result                                                
            volumes_final = self.volume_net_basepoint(volumes) if self.use_separate_v2v_for_basepoint else self.volume_net(volumes)
            
            tri_keypoints_3d, volumes_final = op.integrate_tensor_3d_with_coordinates(volumes_final * self.volume_multiplier, 
                                                                        coord_volumes, 
                                                                        softmax=self.volume_softmax)
        
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

        if self.process_frames_independently:
            coord_volumes = coord_volumes.view(-1, *coord_volumes.shape[-4:])
            proj_matricies_batch = proj_matricies_batch.view(-1, 1, *proj_matricies_batch.shape[-2:])
                   
        # lift each featuremap to distinct volume and aggregate 
        volumes = op.unproject_heatmaps(features,  
                                        proj_matricies_batch, 
                                        coord_volumes, 
                                        volume_aggregation_method=self.volume_aggregation_method,
                                        vol_confidences=vol_confidences
                                        )

        # cat along view dimension
        if self.volume_aggregation_method == 'no_aggregation':        
            volumes = torch.cat(volumes, 0)  

        # inference
        volumes = self.volume_net(volumes)
        # integral 3d
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes_final * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

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

        # modules
        self.backbone = pose_resnet.get_pose_net(config.model.backbone) 
        self.volume_net = V2VModelAdaIN(32, self.num_joints)
        self.features_sequence_to_vector = Seq2VecRNN(256, output_features_dim=self.style_vector_dim)
        self.affine_mappings = nn.ModuleList([nn.Linear(self.style_vector_dim, 2*C) for C in CHANNELS_LIST]) # 51
        self.process_features = nn.Sequential(
            nn.Conv2d(256, self.features_dim, 1)
        )


    def forward(self, images_batch, batch, root_keypoints=None):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]

        # reshape for backbone forward
        images_batch = images_batch.view(-1, 3, *image_shape)

        # forward backbone
        heatmaps, features, alg_confidences, vol_confidences = self.backbone(images_batch)

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

        # process features before lifting to volume
        features = features.view(batch_size, dt, features_channels, *features_shape)
        pivot_features = features[:,-1,...] 
        features = features[:,:-1,...]
        style_vector = self.features_sequence_to_vector(features) # [batch_size, 512]
        pivot_features = self.process_features(pivot_features).unsqueeze(1) # add fictive view
        
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

        # modules
        self.backbone = pose_resnet.get_pose_net(config.model.backbone)
        self.volume_net = {
            "all":V2VModelAdaIN(self.intermediate_features_dim, self.num_joints)
            # "middle":V2VModelAdaIN_MiddleVector(self.intermediate_features_dim, self.num_joints)    
        }[config.model.adain_type]
        self.process_features = nn.Sequential(
            nn.Conv2d(256, self.intermediate_features_dim, 1)
        )
        self.features_regressor = {
            # "rnn": FeaturesAR_RNN(self.intermediate_features_dim, self.intermediate_features_dim),
            # "conv1d": FeaturesAR_CNN1D(self.intermediate_features_dim, self.intermediate_features_dim),
            "conv2d_unet": FeaturesAR_CNN2D_UNet(self.intermediate_features_dim*(self.dt-1), self.intermediate_features_dim)
            # "conv2d_resnet": FeaturesAR_CNN2D_ResNet(self.intermediate_features_dim*self.dt, self.intermediate_features_dim)
        }[config.model.features_regressor]
        self.volume_net.adain_params=[None]*len(CHANNELS_LIST)
                

    def forward(self, images_batch, batch, root_keypoints=None):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]

        # reshape for backbone forward
        images_batch = images_batch.view(-1, 3, *image_shape)

        # forward backbone
        heatmaps, features, alg_confidences, vol_confidences = self.backbone(images_batch)
        features = self.process_features(features) # [bs, 256, 96, 96] -> [bs, 32, 96, 96] 
        
        # calcualte shapes
        features_shape = features.shape[2:]
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
        
        volumes = torch.cat(volumes, 0)
        volumes_pred = torch.cat(volumes_pred, 0)
        # inference
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
        