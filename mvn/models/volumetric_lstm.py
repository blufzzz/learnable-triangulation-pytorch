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
from mvn.models.v2v_models import V2VModel_v2, V2VModel_v1
from mvn.models.temporal import Seq2VecRNN,\
                                Seq2VecCNN, \
                                Seq2VecRNN2D, \
                                Seq2VecCNN2D, \
                                get_encoder

from pytorch_convolutional_rnn.convolutional_rnn import Conv3dLSTM, Conv3dPeepholeLSTM
from IPython.core.debugger import set_trace


class VolumetricTemporalLSTM(nn.Module):

    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.kind = config.model.kind
        self.num_joints = config.model.backbone.num_joints
        self.dt  = config.dataset.dt

        self.pivot_index =  {'first':self.dt-1,
                            'intermediate':self.dt//2}[config.dataset.pivot_type]

        assert config.dataset.pivot_type == 'first'

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

        self.v2v_type = config.model.v2v_type
        self.v2v_normalization_type = config.model.v2v_normalization_type
        self.include_pivot = config.model.include_pivot
        
        # modules dimensions
        self.volume_features_dim = config.model.volume_features_dim
        self.lstm_in_channels = config.model.lstm_in_channels
        self.lstm_out_channels = config.model.lstm_out_channels
        self.lstm_bidirectional = config.model.lstm_bidirectional
        self.lstm_layers = config.model.lstm_layers

        self.use_final_processing = (self.lstm_out_channels != self.num_joints)
        self.evaluate_only_last_volume = config.model.evaluate_only_last_volume

        # modules
        self.backbone = pose_resnet.get_pose_net(config.model.backbone,
                                                 device=device,
                                                 strict=True)
        
        
        self.lstm3d = Conv3dLSTM(in_channels=self.lstm_in_channels, 
                                 out_channels=self.lstm_out_channels,
                                 bidirectional=self.lstm_bidirectional,
                                 num_layers=self.lstm_layers,
                                 batch_first=True,
                                 kernel_size=3)

        if self.v2v_type == 'v1':
            raise NotImplementedError()
            # self.volume_net = V2VModel_v1(v2v_input_features_dim,
            #                            self.num_joints,
            #                            normalization_type=self.v2v_normalization_type,
            #                            volume_size=self.volume_size)

        elif self.v2v_type == 'v2':
            raise NotImplementedError()
            # self.volume_net = V2VModel_v2(v2v_input_features_dim,
            #                               self.num_joints,
            #                               normalization_type=self.v2v_normalization_type,
            #                               volume_size=self.volume_size)


        elif self.v2v_type == 'conf':
            self.volume_net = V2VModel(self.volume_features_dim,
                                        self.lstm_in_channels,
                                        config=config.model)
            
        self.process_features = nn.Conv2d(256, self.volume_features_dim, 1)

        if self.use_final_processing:
            raise NotImplementedError()
            # self.final_processing = get_final_processing()

        description(self)

    def forward(self, images_batch, batch, randomize_style=False):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]
        assert self.dt == dt
     
        # forward backbone
        heatmaps, features, _, vol_confidences, _ = self.backbone(images_batch.view(-1, 3, *image_shape))
        
        # extract aux_features
        features_shape = features.shape[-2:]
        features_channels = features.shape[1]
        features = self.process_features(features)
        features = features.view(-1, 1, features_channels, *features_shape)
        
        proj_matricies_batch = update_camera(batch, batch_size, image_shape, features_shape, dt, device)
        proj_matricies_batch = proj_matricies_batch.view(-1, 1, *proj_matricies_batch.shape[2:])

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
        volumes = unproject_heatmaps(features,  
                                     proj_matricies_batch, 
                                     coord_volumes,
                                     volume_aggregation_method=self.volume_aggregation_method,
                                     vol_confidences=vol_confidences,
                                     fictive_views=dt
                                     )

        volumes = self.volume_net(volumes, params=style_vector) 
        volumes = volumes.view(batch_size, dt, *volumes.shape[1:]) 

        volumes, _ = self.lstm3d(volumes, None)

        if self.use_final_processing:
            volumes = volumes.view(batch_size, *volumes.shape[2:])     
            volumes = self.final_processing(volumes)   
            volumes = volumes.view(batch_size, dt, *volumes.shape[1:])  

        if self.evaluate_only_last_volume:
            volumes = volumes[:,-1,...]
        else:
            volumes = volumes.view(batch_size, *volumes.shape[2:])    

        # integral 3d
        vol_keypoints_3d, volumes = integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                            coord_volumes,
                                                                            softmax=self.volume_softmax)
        
        if not self.evaluate_only_last_volume:
            volumes = volumes.view(batch_size, dt, *volumes.shape[1:])
            vol_keypoints_3d = vol_keypoints_3d.view(batch_size, dt, *vol_keypoints_3d.shape[1:])   
            vol_keypoints_3d = vol_keypoints_3d.view(batch_size, dt, *vol_keypoints_3d.shape[1:])             
            features = features.view(batch_size, dt, features_channels, *features_shape)          

        return (vol_keypoints_3d,
                features,
                volumes,
                vol_confidences,
                cuboids,
                coord_volumes,
                base_points
                )










