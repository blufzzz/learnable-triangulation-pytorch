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
from mvn.utils.op import get_coord_volumes, unproject_heatmaps, integrate_tensor_3d_with_coordinates, make_3d_heatmap, compose
from mvn.utils.multiview import update_camera
from mvn.utils.misc import get_capacity, description
from mvn.utils import volumetric, cfg
from mvn.models.v2v import V2VModel, BasisNet
from sklearn.decomposition import PCA
from IPython.core.debugger import set_trace


class VolumetricDecompositionNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()
        """docstring for VolumetricDecompositionNet"""

        self.kind = config.model.kind
        self.num_joints = config.model.backbone.num_joints
        self.keypoints_per_frame = config.dataset.keypoints_per_frame if \
                                         hasattr(config.dataset, 'keypoints_per_frame') else False

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
        self.v2v_type = config.model.v2v_type
        self.v2v_normalization_type = config.model.v2v_normalization_type
        assert self.v2v_normalization_type in ['group_norm','batch_norm']
        self.gt_initialization = config.model.gt_initialization if hasattr(config.model, 'gt_initialization') else False

        # modules dimensions
        self.volume_features_dim = config.model.volume_features_dim

        self.use_precalculated_basis = config.model.use_precalculated_basis
        self.decomposition_type = config.model.decomposition_type if hasattr(config.model, "decomposition_type") else None
        
        ##############
        # LOAD BASIS #
        ##############
        self.basis_type = config.model.basis_type
        self.only_basis_coefs = config.model.only_basis_coefs
        if self.use_precalculated_basis: 
            self.n_basis = config.model.n_basis 
            self.basis = torch.load(config.model.basis_path)[:self.n_basis].to(device)
        else:
            self.n_basis = config.model.n_basis
            raise NotImplementedError()

        ############
        # BACKBONE #   
        ############
        self.backbone = pose_resnet.get_pose_net(config.model.backbone,
                                                 device=device,
                                                 strict=True)
        
        print ('Only {} backbone is used...'.format(config.model.backbone.name))

        #######
        # V2V #
        #######
        assert self.v2v_type == 'conf'  
        return_bottleneck = not self.use_precalculated_basis
        output_vector = self.basis_type == 'keypoints'
        self.volume_net = V2VModel(self.volume_features_dim,
                                   self.n_basis if self.only_basis_coefs else self.num_joints*self.n_basis,
                                   v2v_normalization_type=self.v2v_normalization_type,
                                   config=config.model.v2v_configuration,
                                   return_bottleneck=return_bottleneck,
                                   back_layer_output_channels=config.model.v2v_configuration.back_layer_output_channels,
                                   output_vector=output_vector
                                   )
    
        if self.volume_features_dim != 256:    
            self.process_features = nn.Sequential(nn.Conv2d(256, self.volume_features_dim, 1))
        else:
            self.process_features = nn.Sequential()    

        ###############
        # BASIS HEADS #
        ###############
        if not self.use_precalculated_basis:
            # last channel from last bottleneck layer
            bottleneck_dim = config.model.v2v_configuration.bottleneck[-1]['params'][-1] 
            self.basis_net = BasisNet(input_dim=bottleneck_dim, n_basis=self.n_basis)

        description(self)


    def forward(self, 
                images_batch, 
                batch, 
                debug=False,
                master=True):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]
        fictive_view = 1
        ######################
        # FEATURE ECTRACTION #   
        ######################
        _, features, _, _, _ = self.backbone(images_batch.view(-1, 3, *image_shape))
        features = self.process_features(features)
        features_shape = features.shape[-2:]
        features_channels = features.shape[1]
        features = features.view(batch_size, -1, features_channels, *features_shape)

        proj_matricies_batch = update_camera(batch, batch_size, image_shape, features_shape, dt, device)
        ##########
        # PELVIS #   
        ##########            
        if self.pelvis_type == 'gt':
            tri_keypoints_3d = torch.from_numpy(np.array(batch['keypoints_3d'])).type(torch.float).to(device)
            if tri_keypoints_3d.dim() == 4:
                self.keypoints_for_each_frame = True
            elif tri_keypoints_3d.dim() == 3:
                self.keypoints_for_each_frame = False
            else:
                raise RuntimeError('Broken tri_keypoints_3d shape')     
        else:
            raise RuntimeError('In absence of precalculated pelvis or gt pelvis, self.use_volumetric_pelvis should be True') 

        tri_keypoints_3d = tri_keypoints_3d[...,:3]
        keypoints_shape = tri_keypoints_3d.shape[-2:]
        #####################
        # VOLUMES CREATING  #   
        #####################
        coord_volumes_pred, _, base_points_pred = get_coord_volumes(self.kind, 
                                                                    self.training, 
                                                                    self.rotation,
                                                                    self.cuboid_side,
                                                                    self.volume_size, 
                                                                    device,
                                                                    keypoints=tri_keypoints_3d
                                                                    )
        ###############
        # V2V FORWARD #   
        ###############
        proj_matricies_batch = proj_matricies_batch.view(-1, fictive_view, *proj_matricies_batch.shape[2:])
        features = features.view(-1, fictive_view, features_channels, *features_shape)

        # lift each feature-map to distinct volume and aggregate 
        unproj_features = unproject_heatmaps(features,  
                                             proj_matricies_batch, 
                                             coord_volumes_pred, 
                                             volume_aggregation_method=self.volume_aggregation_method,
                                             vol_confidences=None,
                                             fictive_views=None
                                             )

        unproj_features = unproj_features.squeeze(1) # get rid of the fictive single view

        coefficients, bottleneck = self.volume_net(unproj_features)
        
        if self.use_precalculated_basis:
            basis = self.basis
        else:
            basis = self.basis_net(bottleneck)

        if self.basis_type == 'keypoints':
            volumes_pred = None
            if self.only_basis_coefs:
                coefficients = coefficients.unsqueeze(-1).repeat(1,1,basis.shape[-1])
            keypoints_3d_pred = torch.einsum('bnj,bnj->bj', coefficients, basis.unsqueeze(0).repeat(batch_size,1,1))
            keypoints_3d_pred = keypoints_3d_pred.view(batch_size, self.num_joints, -1)
        else:
            coefficients = coefficients.view(batch_size, self.n_basis, self.num_joints, *coefficients.shape[-3:])
            volumes = compose(coefficients, basis, decomposition_type=self.decomposition_type)
            keypoints_3d_pred, volumes_pred = integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                                   coord_volumes_pred,
                                                                                   softmax=self.volume_softmax)

        # set_trace()
        return (keypoints_3d_pred, 
                volumes_pred, 
                coefficients, # T
                basis, # [U1,...,Un]
                coord_volumes_pred,
                base_points_pred)