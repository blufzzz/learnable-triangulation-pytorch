from copy import deepcopy
import numpy as np
import pickle
import random

from scipy.optimize import least_squares

import torch
from torch import nn

from mvn.utils import op, multiview, img, misc, volumetric

from mvn.utils.op import get_coord_volumes, unproject_heatmaps, integrate_tensor_3d_with_coordinates

from mvn.models.temporal import get_encoder
from mvn.models import pose_resnet, pose_hrnet
from mvn.models.v2v import V2VModel
from mvn.models.v2v_models import V2VModel_v1
from IPython.core.debugger import set_trace


class Baseline(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints
        self.volume_aggregation_method = config.model.volume_aggregation_method

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size
        self.rotation = config.model.rotation

        self.cuboid_side = config.model.cuboid_side

        self.kind = config.model.kind
        self.pelvis_type = config.model.pelvis_type
        if self.pelvis_type != 'gt':
            self.pelvis_gradient = config.opt.pelvis_gradient if hasattr(config.opt, 'pelvis_gradient') else True
        self.pelvis_space_type =config.model.pelvis_space_type 
        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier
        self.backbone_features_dim = config.model.backbone.backbone_features_dim if \
                                        hasattr(config.model.backbone, 'backbone_features_dim') else 256
        self.volume_features_dim = config.model.volume_features_dim
        self.v2v_type = config.model.v2v_type

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        self.volume_additional_grid_offsets = config.model.volume_additional_grid_offsets if \
                                                 hasattr(config.model, 'volume_additional_grid_offsets') else False
        if self.volume_additional_grid_offsets:
            self.cell_size = self.cuboid_side / self.volume_size
            self.max_cell_size_multiplier = config.model.max_cell_size_multiplier

        # modules
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        self.style_vector_parameter = config.model.style_vector_parameter if hasattr(config.model,'style_vector_parameter') else False


        if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True

        if config.model.backbone.name in ['hrnet32', 'hrnet48']:
            self.backbone = pose_hrnet.get_pose_net(config.model.backbone, device=device)
        elif config.model.backbone.name in ['resnet152', 'resnet50']:    
            self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)
        else:
            raise    

        if config.model.backbone.return_heatmaps:
            for p in self.backbone.final_layer.parameters():
                p.requires_grad = False

        self.process_features = nn.Sequential(
            nn.Conv2d(self.backbone_features_dim, self.volume_features_dim, 1)
        )

        v2v_output_dim = self.num_joints + 3 if self.volume_additional_grid_offsets else self.num_joints

        if self.v2v_type == 'v1':     
            
            style_vector_dim = config.model.style_vector_dim if self.style_vector_parameter else None
            batch_size = config.opt.batch_size
            assert batch_size == config.opt.val_batch_size
            self.style_vector_shape = [batch_size, style_vector_dim, self.volume_size, self.volume_size, self.volume_size] if \
                                         self.style_vector_parameter else None

            self.volume_net = V2VModel_v1(self.volume_features_dim, 
                                          v2v_output_dim, 
                                          self.volume_size,
                                          style_vector_parameter = self.style_vector_parameter,
                                          style_vector_shape = self.style_vector_shape,
                                          style_vector_dim = style_vector_dim,
                                          normalization_type = config.model.normalization_type,
                                          temporal_condition_type = 'spade' if self.style_vector_parameter else None)

        elif self.v2v_type == 'conf':
            self.volume_net = V2VModel(self.volume_features_dim,
                                       v2v_output_dim,
                                       self.volume_size,
                                       config=config.model)
        else:
            raise RuntimeError('Unknown v2v_type')     

        if self.pelvis_type == 'bottleneck':
            self.pelvis_regressor = get_encoder('backbone', 
                                                config.model.backbone.name,
                                                3, # coordinates
                                                upscale_bottleneck=False,
                                                capacity=4, 
                                                spatial_dimension=1, 
                                                encoder_normalization_type='group_norm')
        elif self.pelvis_type == 'v2v':
            self.pelvis_regressor = V2VModel_v1(self.volume_features_dim, 
                                              1, 
                                              self.volume_size,
                                              normalization_type = config.model.normalization_type)    
                                   

    def forward(self, images, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]
        assert n_views == 1

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        _, features, _, vol_confidences, bottleneck = self.backbone(images)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        image_shape, features_shape = tuple(images.shape[3:]), tuple(features.shape[3:])

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras_batch = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras_batch[view_i][batch_i].update_after_resize(image_shape, features_shape)

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch_views], dim=0) for camera_batch_views in new_cameras_batch], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        R_s_inv = torch.stack([torch.stack([torch.from_numpy(np.linalg.inv(camera.R)) for camera in camera_batch_views], dim=0) for camera_batch_views in new_cameras_batch], dim=0).transpose(1, 0)
        t_s = torch.stack([torch.stack([torch.from_numpy(camera.t) for camera in camera_batch_views], dim=0) for camera_batch_views in new_cameras_batch], dim=0).transpose(1, 0)
        proj_matricies = proj_matricies.float().to(device)

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.view(batch_size, n_views, *features.shape[1:])

        if self.pelvis_type =='gt':
            tri_keypoints_3d = torch.from_numpy(np.array(batch['keypoints_3d'])[...,:3]).type(torch.float).to(device)
        elif self.pelvis_type in ['multistage', 'v2v']:
            if not self.pelvis_gradient:
                features_aux = features.clone().detach()
            else:
                features_aux = features    
            сoord_volumes_zeros, _, _ = get_coord_volumes(self.kind, 
                                                        False, 
                                                        False, 
                                                        self.cuboid_side, 
                                                        self.volume_size, 
                                                        device, 
                                                        keypoints=None,
                                                        batch_size=batch_size,
                                                        dt=None)

            unproj_features = unproject_heatmaps(features_aux,  
                                                 proj_matricies, 
                                                 сoord_volumes_zeros, 
                                                 volume_aggregation_method=self.volume_aggregation_method,
                                                 vol_confidences=vol_confidences
                                                 )
            if self.pelvis_type == 'v2v':
                volumes = self.pelvis_regressor(unproj_features)
            else:    
                volumes = self.volume_net(unproj_features)
            tri_keypoints_3d, volumes = integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                             сoord_volumes_zeros, 
                                                                             softmax=self.volume_softmax)

        elif self.pelvis_type == 'bottleneck':
            if not self.pelvis_gradient:
                bottleneck = bottleneck.clone().detach()

            tri_keypoints_3d = self.pelvis_regressor(bottleneck)
            tri_keypoints_3d = tri_keypoints_3d.unsqueeze(-1)

            R_s_inv = R_s_inv.squeeze(1).to(device)
            t_s = t_s.squeeze(1).to(device)
            # TODO: map to global space
            tri_keypoints_3d=R_s_inv@(tri_keypoints_3d - t_s)
            tri_keypoints_3d = tri_keypoints_3d.squeeze(-1).unsqueeze(-2)

        # set_trace()       
        # COORD VOLUMES
        coord_volumes, _, base_points = get_coord_volumes(self.kind, 
                                                            self.training, 
                                                            self.rotation,
                                                            self.cuboid_side,
                                                            self.volume_size, 
                                                            device,
                                                            keypoints=tri_keypoints_3d
                                                            )
        
        # lift to volume
        volumes = unproject_heatmaps(features, 
                                        proj_matricies, 
                                        coord_volumes, 
                                        volume_aggregation_method=self.volume_aggregation_method, 
                                        vol_confidences=vol_confidences)

        # integral 3d
        volumes = self.volume_net(volumes)

        if self.volume_additional_grid_offsets:
            grid_offsets = volumes[:,-3:,...].contiguous().transpose(4,1)
            volumes = volumes[:,:-3,...].contiguous()
            grid_offsets = grid_offsets.tanh() * self.cell_size * self.max_cell_size_multiplier
            coord_volumes = coord_volumes + grid_offsets

        vol_keypoints_3d, volumes = integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        # set_trace()
        return [vol_keypoints_3d,
               features, 
               volumes, 
               vol_confidences, 
               None, 
               coord_volumes, 
               base_points]
