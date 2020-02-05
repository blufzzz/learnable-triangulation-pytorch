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
from mvn.models.v2v import V2VModel, V2VModelAdaIN_MiddleVector, V2VModel_v2
from mvn.models.temporal import Seq2VecRNN,\
                                Seq2VecCNN, \
                                FeaturesAR_RNN,\
                                FeaturesAR_CNN1D,\
                                FeaturesAR_CNN2D_UNet,\
                                FeaturesAR_CNN2D_ResNet,\
                                FeaturesEncoder_Bottleneck,\
                                FeaturesEncoder_DenseNet

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


STYLE_VEC = [[0.0000, 0.0000, 0.0066, 0.0000, 0.0548, 0.0314, 0.0312, 0.0201, 0.0098,
        0.0376, 0.0000, 0.0256, 0.0000, 0.0212, 0.0000, 0.0360, 0.0000, 0.0301,
        0.0000, 0.0097, 0.0542, 0.0045, 0.0420, 0.0010, 0.0231, 0.0642, 0.0089,
        0.0479, 0.0000, 0.0136, 0.0000, 0.0219, 0.0000, 0.0340, 0.0000, 0.0596,
        0.0148, 0.0000, 0.0521, 0.0037, 0.0134, 0.0000, 0.0574, 0.0000, 0.0103,
        0.0048, 0.0364, 0.0000, 0.0000, 0.0000, 0.0019, 0.0000, 0.0117, 0.0132,
        0.0212, 0.0368, 0.0000, 0.0211, 0.0072, 0.0583, 0.0326, 0.0000, 0.0000,
        0.0135, 0.0294, 0.0197, 0.0000, 0.0371, 0.0000, 0.0000, 0.0212, 0.0096,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0352, 0.0000, 0.0266, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0246, 0.0000, 0.0000, 0.0280, 0.0000,
        0.0073, 0.0000, 0.0000, 0.0000, 0.0000, 0.0252, 0.0000, 0.0000, 0.0094,
        0.0000, 0.0800, 0.0000, 0.0279, 0.0000, 0.0419, 0.0000, 0.0084, 0.0307,
        0.0671, 0.0323, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0086, 0.0000,
        0.0515, 0.0082, 0.0175, 0.0000, 0.0238, 0.0000, 0.0000, 0.0120, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0097, 0.0000, 0.0000, 0.0000, 0.0220, 0.0000,
        0.0000, 0.0000, 0.0670, 0.0123, 0.0000, 0.0000, 0.0000, 0.0000, 0.0559,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0159, 0.0000, 0.0110, 0.0000, 0.0105,
        0.0026, 0.0000, 0.0032, 0.0000, 0.0399, 0.0089, 0.0000, 0.0069, 0.0000,
        0.0000, 0.0225, 0.0000, 0.0351, 0.0000, 0.0000, 0.0000, 0.0000, 0.0343,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0211, 0.0660,
        0.0000, 0.0038, 0.0000, 0.0209, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0198, 0.0485, 0.0066, 0.0236, 0.0000, 0.0000, 0.0154, 0.0000, 0.0000,
        0.0000, 0.0122, 0.0000, 0.0675, 0.0019, 0.0000, 0.0011, 0.0000, 0.0297,
        0.0000, 0.0000, 0.0000, 0.0071, 0.0342, 0.0099, 0.0000, 0.0074, 0.0221,
        0.0000, 0.0000, 0.0188, 0.0085, 0.0000, 0.0000, 0.0000, 0.0380, 0.0198,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0017, 0.0000, 0.0000, 0.0379,
        0.0000, 0.0009, 0.0000, 0.0069, 0.0024, 0.0475, 0.0202, 0.0144, 0.0297,
        0.0089, 0.0254, 0.0000, 0.0000, 0.0000, 0.0371, 0.0467, 0.0000, 0.0000,
        0.0000, 0.0289, 0.0405, 0.0000]]


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
        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)
        self.return_heatmaps = config.model.backbone.return_heatmaps if hasattr(config.model.backbone, 'return_heatmaps') else False  
        self.features_channels = config.model.features_channels if hasattr(config.model, 'features_channels') else 32  
        self.process_features = nn.Sequential(
            nn.Conv2d(256, self.features_channels, 1)
        )
        self.volume_net = {
            "channel_stack":V2VModel(self.features_channels*self.dt, self.num_joints)
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


class VolumetricTemporalAdaINNet(nn.Module):

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
        self.f2v_type = config.model.f2v_type
        self.adain_type = config.model.adain_type
        self.style_grad_for_backbone = config.model.style_grad_for_backbone
        self.encoder_type = config.model.encoder_type
        self.pretrained_encoder = config.model.pretrained_encoder
        self.include_pivot = config.model.include_pivot
        self.use_auxilary_backbone = hasattr(config.model, 'auxilary_backbone')

        # modules dimensions
        self.volume_features_dim = config.model.volume_features_dim if hasattr(config.model, 'volume_features_dim') else 32
        self.style_vector_dim = config.model.style_vector_dim if hasattr(config.model, 'style_vector_dim') else 256
        self.intermediate_channels = config.model.intermediate_channels if hasattr(config.model, 'intermediate_channels') else 512
        self.encoded_feature_space = config.model.encoded_feature_space
        
        self.normalization_type = config.model.normalization_type if hasattr(config.model, 'normalization_type') else 'batch_norm'
        if self.adain_type in ['all', 'all_v2']:
            assert self.normalization_type in ['adain','ada-group_norm','group-ada_norm']

        # modules
        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)
        
        if self.use_auxilary_backbone:
            print ('Using auxilary backbone')
            self.auxilary_backbone = pose_resnet.get_pose_net(config.model.auxilary_backbone, device=device)


        if self.encoder_type == "densenet":
            self.encoder = FeaturesEncoder_DenseNet(256, 
                                                    self.encoded_feature_space, 
                                                    pretrained = self.pretrained_encoder)
        elif self.encoder_type == "backbone":
            self.encoder =  FeaturesEncoder_Bottleneck(self.encoded_feature_space,
                                                        C = config.model.encoder_capacity_multiplier)

        else:
            raise RuntimeError('Wrong encoder_type!')    
        
        if self.adain_type == 'all':    
            self.volume_net = V2VModel(self.volume_features_dim,
                                       self.num_joints,
                                       normalization_type=self.normalization_type)
            self.affine_mappings = nn.ModuleList([nn.Linear(self.style_vector_dim, 2*C) for C in CHANNELS_LIST]) #51

        elif self.adain_type == 'middle': 
            self.volume_net =V2VModelAdaIN_MiddleVector(self.volume_features_dim, 
                                                        self.num_joints, 
                                                        normalization_type=self.normalization_type)
            self.affine_mappings = nn.ModuleList([nn.Linear(self.style_vector_dim, 2*128) for _ in range(2)])

        elif self.adain_type == 'all_v2': 
            self.volume_net =V2VModel_v2(self.volume_features_dim, 
                                        self.num_joints, 
                                        normalization_type=self.normalization_type)

            self.affine_mappings = nn.ModuleList([nn.Linear(self.style_vector_dim, 2*C) for C in CHANNELS_LIST_v2]) 

        else:
            raise RuntimeError('Wrong adain_type')
        

        if self.f2v_type == 'lstm':    
            self.features_sequence_to_vector = Seq2VecRNN(self.encoded_feature_space,
                                                          self.style_vector_dim,
                                                          self.intermediate_channels)
        elif self.f2v_type == 'cnn':
            self.dt  = config.dataset.dt if self.include_pivot else config.dataset.dt - 1
            self.kernel_size = config.model.kernel_size
            self.features_sequence_to_vector =Seq2VecCNN(self.encoded_feature_space,
                                                          self.style_vector_dim,
                                                          self.intermediate_channels,
                                                          dt=self.dt,
                                                          kernel_size = self.kernel_size)
        else:
            raise RuntimeError('Wrong features_sequence_to_vector type')    

        self.process_features = nn.Conv2d(256, self.volume_features_dim, 1) if self.volume_features_dim != 256 else nn.Sequential()

    def forward(self, images_batch, batch, root_keypoints=None, random=False):

        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]

        if self.use_auxilary_backbone:

            pivot_images_batch = images_batch[:,-1,...]

            aux_images_batch = images_batch if self.include_pivot else images_batch[:,:-1,...]
            aux_images_batch = aux_images_batch.view(-1, 3, *image_shape)

            aux_heatmaps, aux_features, _, aux_vol_confidences, aux_bottleneck = self.auxilary_backbone(aux_images_batch)
            pivot_heatmaps, pivot_features, pivot_alg_confidences, vol_confidences, pivot_bottleneck = self.backbone(pivot_images_batch)
            pivot_features = self.process_features(pivot_features).unsqueeze(1)
            features_shape = pivot_features.shape[-2:]
            features_channels = pivot_features.shape[1]
            bottleneck_shape = aux_bottleneck.shape[-2:]
            bottleneck_channels = aux_bottleneck.shape[1]
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
            aux_features = aux_features.view(-1, features_channels, *features_shape)

            # extract aux_bottleneck
            bottleneck_shape = bottleneck.shape[-2:]
            bottleneck_channels = bottleneck.shape[1]
            bottleneck = bottleneck.view(batch_size, -1, bottleneck_channels, *bottleneck_shape)
            aux_bottleneck = bottleneck if self.include_pivot else bottleneck[:,:-1,...].contiguous()
            aux_bottleneck = aux_bottleneck.view(-1, bottleneck_channels, *bottleneck_shape)
        

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

        if self.encoder_type == 'backbone':
            if not self.style_grad_for_backbone:
                aux_bottleneck = aux_bottleneck.detach()
            encoded_features = self.encoder(aux_bottleneck)
        else:
            if self.style_grad_for_backbone:
                aux_features = aux_features.detach()
            encoded_features = self.encoder(aux_features)

        encoded_features = encoded_features.view(batch_size,
                                                 -1, # (dt-1) 
                                                 self.encoded_feature_space)

        if random:
            style_vector = torch.tensor(STYLE_VEC, requires_grad=True).to(device)
        else:    
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

        # print ('increased by:', (torch.cuda.memory_allocated() - current_memory) / (1024**2))
        # current_memory = torch.cuda.memory_allocated()
        # inference
        adain_params = [affine_map(style_vector) for affine_map in self.affine_mappings]

        volumes = self.volume_net(volumes, params=adain_params)
        # integral 3d
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return (vol_keypoints_3d,
                None if self.use_auxilary_backbone else features,
                volumes,
                vol_confidences,
                cuboids,
                coord_volumes,
                base_points,
                style_vector
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
        self.features_regressor_base_channels = config.model.features_regressor_base_channels if hasattr(config.model, 'features_regressor_base_channels') else 8
        self.adain_type = config.model.adain_type
        self.fr_grad_for_backbone = config.model.fr_grad_for_backbone if hasattr(config.model, 'fr_grad_for_backbone') else False
        self.normalization_type = config.model.normalization_type if hasattr(config.model, 'normalization_type') else 'batch_norm'
        
        # modules
        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)
        self.volume_net = {
            "all":V2VModel(self.intermediate_features_dim, 
                                self.num_joints, 
                                normalization_type='adain'),
            "middle":V2VModelAdaIN_MiddleVector(self.intermediate_features_dim, 
                                                self.num_joints, 
                                                normalization_type=self.normalization_type),
            "no_adain": V2VModel(self.intermediate_features_dim*2, 
                                 self.num_joints, 
                                 normalization_type=self.normalization_type)    
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
        
        if self.adain_type == 'no_adain':
            volumes = self.volume_net(torch.cat([volumes, volumes_pred], 1)) # stack along channel
            vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)
            vol_keypoints_3d_pred, volumes_pred = None, None
        else:    
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
        