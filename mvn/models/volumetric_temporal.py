from copy import deepcopy
import numpy as np
import pickle
import random

import torch
from torch import nn

from pose.models import hg, pose_resnet

from mvn.utils import op
from mvn.utils import multiview
from mvn.utils import img
from mvn.utils import misc
from mvn.utils import volumetric
from mvn.models.v2v import V2VModel, V2VRegModel, V2VModel_2, V2VModel_btnck, V2VModel_configured, V2VModel2Stages, V2VModel_3
from mvn.models.temporal import TemporalModel

from IPython.core.debugger import set_trace


class VolumetricTemporalNet(nn.Module):

    '''
    The model is designed to work with `dt` number of consecutive frames
    '''

    def __init__(self, config):
        super().__init__()

        self.num_joints = config.backbone.num_joints
        self.volume_aggregation_method = config.volume_aggregation_method # if hasattr(config, 'volume_aggregation_method') else None

        # volume
        self.volume_softmax = config.volume_softmax
        self.volume_multiplier = config.volume_multiplier
        self.volume_size = config.volume_size

        self.cuboid_side = config.cuboid_side
        self.cuboid_multiplier = config.cuboid_multiplier if hasattr(config, "cuboid_multiplier") else 1.0
        self.rotation = config.rotation if hasattr(config, "rotation") else False

        self.process_frames_independently = config.process_frames_independently if hasattr(config, 'process_frames_independently') else False

        self.kind = config.kind

        self.use_precalculated_pelvis = config.use_precalculated_pelvis if hasattr(config, "use_precalculated_pelvis") else False
        self.use_gt_pelvis = config.use_gt_pelvis if hasattr(config, "use_gt_pelvis") else False
        self.use_volumetric_pelvis = config.use_volumetric_pelvis if hasattr(config, "use_volumetric_pelvis") else False
        self.use_separate_v2v_for_basepoint = config.use_separate_v2v_for_basepoint if hasattr(config, "use_separate_v2v_for_basepoint") else False

        assert self.use_precalculated_pelvis or self.use_volumetric_pelvis or self.use_gt_pelvis, 'One of the flags "use_<...>_pelvis" should be True'

        # heatmap
        self.heatmap_softmax = config.heatmap_softmax
        self.heatmap_multiplier = config.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.transfer_cmu_to_human36m if hasattr(config, "transfer_cmu_to_human36m") else False

         if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True

        # modules
        self.backbone = pose_resnet.get_pose_net(config.backbone)
        self.return_heatmaps = config.backbone.return_heatmaps if hasattr(config.backbone, 'return_heatmaps') else False  

        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )

        self.volume_net = V2VModel(32, self.num_joints)
        
    def get_coord_volumes(self, 
                            cuboid_side, 
                            volume_size, 
                            device, 
                            keypoints = None,
                            batch_size = None,
                            dt = None
                            ):
    
        use_default_basepoint = keypoints is None
        bs_dt = (batch_size, dt) if use_default_basepoint else keypoints.shape[:-2]
        sides = torch.tensor([cuboid_side, cuboid_side, cuboid_side], dtype=torch.float).to(device)

        # default base_points are the coordinate's origins
        base_points = torch.zeros((*bs_dt, 3), dtype=torch.float).to(device)
        
        if not use_default_basepoint:    
            # get root (pelvis) from keypoints   
            if self.kind == "coco":
                base_points = (keypoints[...,11, :3] + keypoints[...,12, :3]) / 2
            elif self.kind == "mpii":
                base_points = keypoints[..., 6, :3] 

        position = base_points - sides / 2

        # build cuboids
        cuboids = None

        # build coord volume
        xxx, yyy, zzz = torch.meshgrid(torch.arange(volume_size, device=device),
                                        torch.arange(volume_size, device=device),
                                         torch.arange(volume_size, device=device))
        grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
        grid = grid.view((-1, 3))
        grid = grid.view(*[1]*len(bs_dt), *grid.shape).repeat(*keypoints.shape[:-2], *[1]*len(grid.shape))

        grid[..., 0] = position[..., 0].unsqueeze(-1) + (sides[0] / (volume_size - 1)) * grid[..., 0]
        grid[..., 1] = position[..., 1].unsqueeze(-1) + (sides[1] / (volume_size - 1)) * grid[..., 1]
        grid[..., 2] = position[..., 2].unsqueeze(-1) + (sides[2] / (volume_size - 1)) * grid[..., 2]
        
        if self.kind == "coco":
            axis = [0, 1, 0]  # y axis
        elif self.kind == "mpii":
            axis = [0, 0, 1]  # z axis
            
        # random rotation
        if self.training and self.rotation:    
            
            center = torch.from_numpy(base_points).type(torch.float).to(device).unsqueeze(-2)
            grid = grid - center
            
            grid = torch.stack([volumetric.rotate_coord_volume(coord_grid,\
                                np.random.uniform(0.0, 2 * np.pi), axis) for coord_grid in grid])
            
            grid = grid + center
        
        grid = grid.view(*bs_dt, volume_size, volume_size, volume_size, 3)
    
        return grid, cuboids, base_points

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

            coord_volumes, _, _ = self.get_coord_volumes(self.cuboid_side*self.cuboid_multiplier,
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
        coord_volumes, cuboids, base_points = self.get_coord_volumes(self.cuboid_side,
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

# TODO
class VolumetricAdaINConditionedTemporalNet(nn.Module):

    '''
    Volumetric AdaIN Conditioned Temporal Net
    The model is designed to work with `dt` number of consecutive frames
    '''

    def __init__(self, config):
        super().__init__()

        self.num_joints = config.backbone.num_joints

        # volume
        self.volume_softmax = config.volume_softmax
        self.volume_multiplier = config.volume_multiplier
        self.volume_size = config.volume_size

        self.cuboid_side = config.cuboid_side
        self.cuboid_multiplier = config.cuboid_multiplier if hasattr(config, "cuboid_multiplier") else 1.0
        self.rotation = config.rotation if hasattr(config, "rotation") else False

        self.process_frames_independently = config.process_frames_independently if hasattr(config, 'process_frames_independently') else False

        self.kind = config.kind

        self.use_precalculated_pelvis = config.use_precalculated_pelvis if hasattr(config, "use_precalculated_pelvis") else False
        self.use_gt_pelvis = config.use_gt_pelvis if hasattr(config, "use_gt_pelvis") else False
        self.use_volumetric_pelvis = config.use_volumetric_pelvis if hasattr(config, "use_volumetric_pelvis") else False
        self.use_separate_v2v_for_basepoint = config.use_separate_v2v_for_basepoint if hasattr(config, "use_separate_v2v_for_basepoint") else False

        assert self.use_precalculated_pelvis or self.use_volumetric_pelvis or self.use_gt_pelvis, 'One of the flags "use_<...>_pelvis" should be True'

        # heatmap
        self.heatmap_softmax = config.heatmap_softmax
        self.heatmap_multiplier = config.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.transfer_cmu_to_human36m if hasattr(config, "transfer_cmu_to_human36m") else False

         if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True

        # modules
        self.backbone = pose_resnet.get_pose_net(config.backbone)
        self.return_heatmaps = config.backbone.return_heatmaps if hasattr(config.backbone, 'return_heatmaps') else False  

        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )

        self.volume_net = V2VModel(32, self.num_joints)
        
    def get_coord_volumes(self, 
                            cuboid_side, 
                            volume_size, 
                            device, 
                            keypoints = None,
                            batch_size = None,
                            dt = None
                            ):
    
        use_default_basepoint = keypoints is None
        bs_dt = (batch_size, dt) if use_default_basepoint else keypoints.shape[:-2]
        sides = torch.tensor([cuboid_side, cuboid_side, cuboid_side], dtype=torch.float).to(device)

        # default base_points are the coordinate's origins
        base_points = torch.zeros((*bs_dt, 3), dtype=torch.float).to(device)
        
        if not use_default_basepoint:    
            # get root (pelvis) from keypoints   
            if self.kind == "coco":
                base_points = (keypoints[...,11, :3] + keypoints[...,12, :3]) / 2
            elif self.kind == "mpii":
                base_points = keypoints[..., 6, :3] 

        position = base_points - sides / 2

        # build cuboids
        cuboids = None

        # build coord volume
        xxx, yyy, zzz = torch.meshgrid(torch.arange(volume_size, device=device),
                                        torch.arange(volume_size, device=device),
                                         torch.arange(volume_size, device=device))
        grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
        grid = grid.view((-1, 3))
        grid = grid.view(*[1]*len(bs_dt), *grid.shape).repeat(*keypoints.shape[:-2], *[1]*len(grid.shape))

        grid[..., 0] = position[..., 0].unsqueeze(-1) + (sides[0] / (volume_size - 1)) * grid[..., 0]
        grid[..., 1] = position[..., 1].unsqueeze(-1) + (sides[1] / (volume_size - 1)) * grid[..., 1]
        grid[..., 2] = position[..., 2].unsqueeze(-1) + (sides[2] / (volume_size - 1)) * grid[..., 2]
        
        if self.kind == "coco":
            axis = [0, 1, 0]  # y axis
        elif self.kind == "mpii":
            axis = [0, 0, 1]  # z axis
            
        # random rotation
        if self.training and self.rotation:    
            
            center = torch.from_numpy(base_points).type(torch.float).to(device).unsqueeze(-2)
            grid = grid - center
            
            grid = torch.stack([volumetric.rotate_coord_volume(coord_grid,\
                                np.random.uniform(0.0, 2 * np.pi), axis) for coord_grid in grid])
            
            grid = grid + center
        
        grid = grid.view(*bs_dt, volume_size, volume_size, volume_size, 3)
    
        return grid, cuboids, base_points

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

            coord_volumes, _, _ = self.get_coord_volumes(self.cuboid_side*self.cuboid_multiplier,
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
        coord_volumes, cuboids, base_points = self.get_coord_volumes(self.cuboid_side,
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
