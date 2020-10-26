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
from mvn.models.v2v import V2VModel, TuckerBasisNet
from mvn.models.image2lixel_common.nets.module import PoseNet, PoseNetTT
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

        self.use_precalculated_basis = config.model.use_precalculated_basis if hasattr(config.model, "use_precalculated_basis") \
                                        else (config.model.basis_source == 'precalculated')
        self.decomposition_type = config.model.decomposition_type if hasattr(config.model, "decomposition_type") else None

        self.joint_independent = config.model.joint_independent if hasattr(config.model, "joint_independent") else False

        self.dimensions_in_decomposition = config.model.dimensions_in_decomposition \
                                            if hasattr(config.model, "dimensions_in_decomposition") else ['j','x','y','z']

        self.ranks_in_decomposition = config.model.ranks_in_decomposition \
                                            if hasattr(config.model, "ranks_in_decomposition") else [1,1,1,1]

        assert len(self.ranks_in_decomposition) == len(self.dimensions_in_decomposition)
        
        ##############
        # LOAD BASIS #
        ##############
        self.basis_type = config.model.basis_type
        if hasattr(config.model, "basis_source"):
            self.basis_source = config.model.basis_source
        elif config.model.use_precalculated_basis:
            self.basis_source = 'precalculated'
        else:   
            self.basis_source = 'basis_net'
        self.only_basis_coefs = config.model.only_basis_coefs if hasattr(config.model, "n_basis") else False
        self.n_basis = config.model.n_basis if hasattr(config.model, "n_basis") else None
        
        if self.basis_source == 'precalculated': 
            self.basis = torch.load(config.model.basis_path)
            if self.decomposition_type == 'svd':
                self.basis = self.basis[:self.n_basis].to(device)
        elif self.basis_source ==  'basis_net':
            if self.decomposition_type == 'tucker':
                bottleneck_dim = config.model.v2v_configuration.bottleneck[-1]['params'][-1] 
                self.basis_net = TuckerBasisNet(input_dim=bottleneck_dim,
                                          volume_size=self.volume_size, 
                                          n_joints=self.num_joints,
                                          normalization_type=self.v2v_normalization_type)
            elif self.decomposition_type == 'tt':
                self.rank = config.model.rank
                self.v2v_output_dim = config.model.v2v_output_dim
                self.posenet3d_intermediate_features = config.model.posenet3d_intermediate_features
                self.basis_net = PoseNetTT(self.rank,
                                           self.volume_size, 
                                           self.num_joints, 
                                           input_features=self.v2v_output_dim, 
                                           intermediate_features=self.posenet3d_intermediate_features, 
                                           normalization_type=self.v2v_normalization_type, 
                                           joint_independent=self.joint_independent)
            else:
                raise RuntimeError('wrong `decomposition_type` for this `basis_source`')
        elif self.basis_source == 'optimized':
            if self.decomposition_type == 'tucker':
                if config.model.basis_path is not None:
                    basis = torch.load(config.model.basis_path)
                    self.basis = nn.ParameterList([nn.Parameter(basis[0].to(device)),
                                                    nn.Parameter(basis[1].to(device)),
                                                    nn.Parameter(basis[2].to(device)),
                                                    nn.Parameter(basis[3].to(device))])
                else:
                    self.basis = nn.ParameterList([nn.Parameter(torch.randn(self.num_joints, self.num_joints, device=device)),
                                                    nn.Parameter(torch.randn(self.volume_size, self.volume_size, device=device)),
                                                    nn.Parameter(torch.randn(self.volume_size, self.volume_size, device=device)),
                                                    nn.Parameter(torch.randn(self.volume_size, self.volume_size, device=device))])

        else:
            raise RuntimeError('Unknown `basis_source`:{}'.format(self.basis_source))

        # DELETE THIS
        self.gt_input = config.model.gt_input if hasattr(config.model, 'gt_input') else False

        ####################
        # RESIDUAL NETWORK #   
        ####################
        self.use_residual_network = config.model.use_residual_network if hasattr(config.model, 'use_residual_network') else False
        

        ############
        # BACKBONE #   
        ############
        if not self.gt_input:
            self.backbone = pose_resnet.get_pose_net(config.model.backbone,
                                                     device=device,
                                                     strict=True)
            if self.volume_features_dim != 256:    
                self.process_features = nn.Sequential(nn.Conv2d(256, self.volume_features_dim, 1))
            else:
                self.process_features = nn.Sequential()
        else:
            self.backbone = nn.Sequential()
            self.process_features = nn.Sequential()
            
            print ('Only {} backbone is used...'.format(config.model.backbone.name))

        #######
        # V2V #
        #######d
        assert self.v2v_type == 'conf'  
        
        return_bottleneck = (self.decomposition_type == 'tucker') and (self.basis_source == 'basis_net')
        
        if self.decomposition_type == 'tucker':
            v2v_output_dim = self.num_joints
        elif self.decomposition_type == 'tt':
            v2v_output_dim = self.v2v_output_dim
        else:
            v2v_output_dim = self.n_basis if self.only_basis_coefs else self.num_joints*self.n_basis*3

        output_vector = self.basis_type == 'keypoints'

        self.volume_net = V2VModel(self.num_joints if self.gt_input else self.volume_features_dim,
                                   v2v_output_dim,
                                   v2v_normalization_type=self.v2v_normalization_type,
                                   config=config.model.v2v_configuration,
                                   return_bottleneck=return_bottleneck,
                                   back_layer_output_channels=config.model.v2v_configuration.back_layer_output_channels,
                                   output_vector=output_vector
                                   )    

        description(self)


    def forward(self,  
                images_batch, 
                batch):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]
        fictive_view = 1
        ######################
        # FEATURE ECTRACTION #   
        ######################
        if not self.gt_input:
            _, features, _, _, _,_ = self.backbone(images_batch.view(-1, 3, *image_shape))
            features = self.process_features(features)
            features_shape = features.shape[-2:]
            features_channels = features.shape[1]
            features = features.view(-1, fictive_view, features_channels, *features_shape)

            proj_matricies_batch = update_camera(batch, batch_size, image_shape, features_shape, dt, device)
            proj_matricies_batch = proj_matricies_batch.view(-1, fictive_view, *proj_matricies_batch.shape[2:])
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
        if not self.gt_input:
            # lift each feature-map to distinct volume and aggregate 
            unproj_features = unproject_heatmaps(features,  
                                                 proj_matricies_batch, 
                                                 coord_volumes_pred, 
                                                 volume_aggregation_method=self.volume_aggregation_method,
                                                 vol_confidences=None,
                                                 fictive_views=None
                                                 )
            unproj_features = unproj_features.squeeze(1) # get rid of the fictive single view
        
        else:
            unproj_features = make_3d_heatmap(coord_volumes_pred, tri_keypoints_3d)

        coefficients, bottleneck = self.volume_net(unproj_features)

        if self.basis_source == 'basis_net':
            if isinstance(self.basis_net, PoseNetTT):
                basis = self.basis_net(coefficients) # for TT
            else:
                basis = self.basis_net(bottleneck) # for Tucker
        else:
            basis = self.basis

        if self.basis_type == 'keypoints':
            volumes_pred = None
            if self.only_basis_coefs:
                coefficients = coefficients.unsqueeze(-1).repeat(1,1,basis.shape[-1])
            else:
                coefficients = coefficients.view(batch_size, self.n_basis, self.num_joints*3)

            keypoints_3d_pred = torch.einsum('bnj,bnj->bj', coefficients, basis.unsqueeze(0).repeat(batch_size,1,1))
            keypoints_3d_pred = keypoints_3d_pred.view(batch_size, self.num_joints, -1)
        
        elif self.basis_type == 'heatmaps':
            volumes = compose(coefficients, basis, decomposition_type=self.decomposition_type, joint_independent=self.joint_independent)
            # check shapes
            keypoints_3d_pred, volumes_pred = integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                                   coord_volumes_pred,
                                                                                   softmax=self.volume_softmax)
        else:
            raise RuntimeError('unknown `basis_type`')

        return (keypoints_3d_pred, 
                volumes_pred, 
                coefficients, # T
                basis, # [U1,...,Un]
                coord_volumes_pred,
                base_points_pred)