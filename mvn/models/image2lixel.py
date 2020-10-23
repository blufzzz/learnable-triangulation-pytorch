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
from mvn.models.v2v import V2VModel
from sklearn.decomposition import PCA
from IPython.core.debugger import set_trace

# from mvn.models.image2lixel_common.nets.resnet import ResNetBackbone
from mvn.models.image2lixel_common.nets.module import PoseNet, Pose2Feat, MeshNet#, ParamRegressor
# from mvn.models.image2lixel_common.nets.loss import CoordLoss, ParamLoss, NormalVectorLoss, EdgeLengthLoss
# from mvn.models.image2lixel_common.utils.smpl import SMPL
# from mvn.models.image2lixel_common.utils.mano import MANO


class I2LModel(nn.Module):
    def __init__(self, config,  device='cuda:0'):
        super(I2LModel, self).__init__()

        self.volume_size = config.model.volume_size

        self.device = device
        self.cuboid_side = config.model.cuboid_side
        self.cuboid_multiplier = config.model.cuboid_multiplier
        self.rotation = config.model.rotation

        self.use_meshnet = config.model.use_meshnet

        self.num_joints = config.model.backbone.num_joints
        self.kind = config.model.kind
        assert self.kind == "mpii"
        self.normalization_type = config.model.pose_net_normalization_type
        self.pelvis_type = config.model.pelvis_type if hasattr(config.model, 'pelvis_type') else 'gt'
        self.pose_net = PoseNet(self.num_joints, normalization_type=self.normalization_type)

        self.backbone = pose_resnet.get_pose_net(config.model.backbone,
                                                 device=device,
                                                 strict=True)

        description(self)

    def forward(self, images_batch, batch):
        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]

        if self.pelvis_type == 'gt':
            tri_keypoints_3d = torch.from_numpy(np.array(batch['keypoints_3d'])).type(torch.float).to(device)
        else:
            raise RuntimeError('In absence of precalculated pelvis or gt pelvis, self.use_volumetric_pelvis should be True') 
        
        # posenet forward
        _, _, _, _, pose_img_feat = self.backbone(images_batch.view(-1, 3, *image_shape))
        coordinates = get_coord_volumes(self.kind, 
                                            self.training, 
                                            self.rotation,
                                            self.cuboid_side, 
                                            self.volume_size, 
                                            self.device, 
                                            keypoints=tri_keypoints_3d,
                                            batch_size=batch_size,
                                            dt=None,
                                            return_only_xyz=True,
                                            max_rotation_angle=np.pi/4 #2 * np.pi
                                            )
        joint_coord_img = self.pose_net(pose_img_feat, coordinates)

        base_points = tri_keypoints_3d[..., 6, :3]


        return (joint_coord_img, 
                None, 
                None, 
                None,
                None,
                base_points)
       
       
def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(vertex_num, joint_num, mode):
    pose_backbone = ResNetBackbone(cfg.resnet_type)
    pose_net = PoseNet(joint_num)
    pose2feat = Pose2Feat(joint_num)
    mesh_backbone = ResNetBackbone(cfg.resnet_type)
    mesh_net = MeshNet(vertex_num)
    param_regressor = ParamRegressor(joint_num)

    if mode == 'train':
        pose_backbone.init_weights()
        pose_net.apply(init_weights)
        pose2feat.apply(init_weights)
        mesh_backbone.init_weights()
        mesh_net.apply(init_weights)
        param_regressor.apply(init_weights)
   
    model = Model(pose_backbone, pose_net, pose2feat, mesh_backbone, mesh_net, param_regressor)
    return model