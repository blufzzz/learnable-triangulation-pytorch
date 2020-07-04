from copy import deepcopy
import numpy as np
import pickle
import random

from scipy.optimize import least_squares

import torch
from torch import nn

from mvn.utils import op, multiview, img, misc, volumetric

from mvn.models import pose_resnet, pose_hrnet
from mvn.models.v2v import V2VModel
from mvn.models.v2v_models import V2VModel_v1
from IPython.core.debugger import set_trace



class VolumetricTriangulationNet(nn.Module):
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

        # heatmap
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
            self.volume_net = V2VModel_v1(self.volume_features_dim, 
                                          v2v_output_dim, 
                                          self.volume_size,
                                          normalization_type = config.model.v2v_normalization_type)

        elif self.v2v_type == 'conf':
            self.v2v_normalization_type = config.model.v2v_normalization_type
            self.volume_net = V2VModel(self.volume_features_dim,
                                       v2v_output_dim,
                                       v2v_normalization_type=self.v2v_normalization_type,
                                       config=config.model.v2v_configuration,
                                       style_vector_dim=None,
                                       params_evolution = False,
                                       temporal_condition_type=None)
        else:
            raise RuntimeError('Unknown v2v_type')                                    

    
    def build_coord_volumes(self, batch, batch_size, device):

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            if self.pelvis_type == 'gt':
                keypoints_3d = batch['keypoints_3d'][batch_i]
            else:
                raise RuntimeError('Wrong pelvis_type')

            if self.kind == "coco":
                base_point = (keypoints_3d[11, :3] + keypoints_3d[12, :3]) / 2
            elif self.kind == "mpii":
                base_point = keypoints_3d[6, :3]

            base_points[batch_i] = torch.from_numpy(base_point).to(device)

            # build cuboid
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device),
                                           torch.arange(self.volume_size, device=device),
                                           torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

            # random rotation
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            if self.kind == "coco":
                axis = [0, 1, 0]  # y axis
            elif self.kind == "mpii":
                axis = [0, 0, 1]  # z axis

            center = torch.from_numpy(base_point).type(torch.float).to(device)

            # rotate
            if self.rotation:
                coord_volume = coord_volume - center
                coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
                coord_volume = coord_volume + center

            # transfer
            if self.transfer_cmu_to_human36m:  # different world coordinates
                coord_volume = coord_volume.permute(0, 2, 1, 3)
                inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
                coord_volume = coord_volume.index_select(1, inv_idx)

            coord_volumes[batch_i] = coord_volume

        return coord_volumes, cuboids, base_points


    def forward(self, images, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        _, features, _, vol_confidences, _ = self.backbone(images)

        current_memory = torch.cuda.memory_allocated()
        
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
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, features_shape)

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        proj_matricies = proj_matricies.float().to(device)

        coord_volumes, cuboids, base_points = self.build_coord_volumes(batch, batch_size, device)

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.view(batch_size, n_views, *features.shape[1:])
        # lift to volume
        volumes = op.unproject_heatmaps(features, 
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

        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)
        # set_trace()
        return [vol_keypoints_3d,
               features, 
               volumes, 
               vol_confidences, 
               cuboids, 
               coord_volumes, 
               base_points]
