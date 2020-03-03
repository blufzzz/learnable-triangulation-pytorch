from copy import deepcopy
import numpy as np
import pickle
import random
from collections import defaultdict

import torch
from torch import nn

from mvn.models import pose_resnet, pose_hrnet
from time import time
from mvn.utils.op import get_coord_volumes, unproject_heatmaps, integrate_tensor_3d_with_coordinates
from mvn.utils.multiview import update_camera
from mvn.utils.misc import get_capacity, description
from mvn.utils import volumetric
from mvn.models.v2v import V2VModel
from mvn.models.v2v_models import V2VModel_v2
from mvn.models.temporal import Seq2VecRNN,\
                                Seq2VecCNN, \
                                Seq2VecRNN2D, \
                                Seq2VecCNN2D, \
                                get_encoder

from IPython.core.debugger import set_trace
from mvn.utils.op import get_coord_volumes



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
        volumes = unproject_heatmaps(pivot_features,  
                                        proj_matricies_batch, 
                                        coord_volumes, 
                                        volume_aggregation_method=self.volume_aggregation_method,
                                        vol_confidences=vol_confidences
                                        )

        volumes_aux = unproject_heatmaps(style_features,  
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
        vol_keypoints_3d, volumes = integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
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


















