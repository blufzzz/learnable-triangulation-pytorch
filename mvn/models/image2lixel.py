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
from mvn.models.image2lixel_common.nets.module import PoseNet, Pose2Feat, MeshNet #, ParamRegressor
# from mvn.models.image2lixel_common.nets.loss import CoordLoss, ParamLoss, NormalVectorLoss, EdgeLengthLoss
# from mvn.models.image2lixel_common.utils.smpl import SMPL
# from mvn.models.image2lixel_common.utils.mano import MANO

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

class I2LModel(nn.Module):
    def __init__(self, config,  device='cuda:0'):
        super(I2LModel, self).__init__()

        self.volume_size = config.model.volume_size

        self.device = device
        self.cuboid_side = config.model.cuboid_side
        self.cuboid_multiplier = config.model.cuboid_multiplier
        self.rotation = config.model.rotation

        self.use_meshnet = config.model.use_meshnet
        self.sigma = config.model.sigma
        self.use_mesh_model = config.model.use_mesh_model

        self.return_coords_posenet = config.model.return_coords_posenet
        self.return_coords_meshnet = config.model.return_coords_meshnet
        self.joint_independent_posenet = config.model.joint_independent_posenet
        self.joint_independent_meshnet = config.model.joint_independent_meshnet

        self.rank = config.model.rank

        self.num_joints = config.model.backbone.num_joints
        self.kind = config.model.kind
        assert self.kind == "mpii"
        self.normalization_type = config.model.pose_net_normalization_type
        self.pelvis_type = config.model.pelvis_type if hasattr(config.model, 'pelvis_type') else 'gt'

        self.pose_net = PoseNet(self.num_joints, 
                                self.volume_size, 
                                return_coords=self.return_coords_posenet,
                                joint_independent=self.joint_independent_posenet,
                                rank=self.rank,
                                normalization_type=self.normalization_type)
        if self.use_meshnet:
            self.pose2feat = Pose2Feat(self.num_joints)
            self.mesh_backbone = pose_resnet.get_pose_net(config.model.backbone,
                                                         device=device,
                                                         strict=True,
                                                         skip_early=True)
            self.mesh_net = MeshNet(self.volume_size, 
                                    self.num_joints, 
                                    return_coords=self.return_coords_meshnet,
                                    joint_independent=self.joint_independent_posenet,
                                    rank=self.rank,
                                    normalization_type=self.normalization_type)

        self.backbone = pose_resnet.get_pose_net(config.model.backbone,
                                                 device=device,
                                                 strict=True)

        description(self)

        if self.training:
            # backbones are already inited
            self.pose_net.apply(init_weights)
            if self.use_meshnet:
                self.pose2feat.apply(init_weights)
                self.mesh_net.apply(init_weights)


    def make_gaussian_heatmap(self, joint_coord_img, sigma):
        x = torch.arange(self.volume_size)
        y = torch.arange(self.volume_size)
        z = torch.arange(self.volume_size) 
        zz,yy,xx = torch.meshgrid(z,y,x)
        xx = xx[None,None,:,:,:].cuda().float(); yy = yy[None,None,:,:,:].cuda().float(); zz = zz[None,None,:,:,:].cuda().float();
        
        x = joint_coord_img[:,:,0,None,None,None]; y = joint_coord_img[:,:,1,None,None,None]; z = joint_coord_img[:,:,2,None,None,None];
        heatmap = torch.exp(-(((xx-x)/sigma)**2)/2 -(((yy-y)/sigma)**2)/2 - (((zz-z)/sigma)**2)/2)
        return heatmap

    def forward(self, images_batch, batch):
        device = images_batch.device
        batch_size, dt = images_batch.shape[:2]
        image_shape = images_batch.shape[-2:]

        if self.pelvis_type == 'gt':
            tri_keypoints_3d = torch.from_numpy(np.array(batch['keypoints_3d'])).type(torch.float).to(device)
        else:
            raise RuntimeError('In absence of precalculated pelvis or gt pelvis, self.use_volumetric_pelvis should be True') 
        
        # posenet forward
        _, _, _, _, pose_img_feat, shared_img_feat = self.backbone(images_batch.view(-1, 3, *image_shape))
        return_only_xyz = self.return_coords_meshnet and self.return_coords_posenet
        coordinates = get_coord_volumes(self.kind, 
                                            self.training, 
                                            self.rotation,
                                            self.cuboid_side, 
                                            self.volume_size, 
                                            self.device, 
                                            keypoints=tri_keypoints_3d,
                                            batch_size=batch_size,
                                            dt=None, 
                                            return_only_xyz=return_only_xyz, 
                                            max_rotation_angle=np.pi/4 #2 * np.pi
                                            )

        base_points = tri_keypoints_3d[..., 6, :3]
        joint_coord_img = self.pose_net(pose_img_feat, coordinates)

        if not self.return_coords_posenet:
                volumes = compose(None, joint_coord_img, 'tt', joint_independent=self.joint_independent_posenet)
                joint_coord_img, volumes_pred = integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                                   coordinates[0],
                                                                                   softmax=self.volume_softmax)

        if self.use_meshnet:
            joint_heatmap = self.make_gaussian_heatmap(joint_coord_img, self.sigma)

            # interpolate - workaround to match `shared_img_feat` spatial dim
            joint_heatmap = F.interpolate(joint_heatmap, (64,64,64))
            shared_img_feat = self.pose2feat(shared_img_feat, joint_heatmap) #[1, 64, 64, 64])

            # meshnet forward
            _, _, _, _, mesh_img_feat, _ = self.mesh_backbone(shared_img_feat)
            mesh_coord_img = self.mesh_net(mesh_img_feat, coordinates)
            
            if self.use_mesh_model:    
                # joint coordinate outputs from mesh coordinates
                joint_img_from_mesh = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None,:,:].repeat(mesh_coord_img.shape[0],1,1), mesh_coord_img)
                joint_coord_img = joint_img_from_mesh
            else:
                joint_coord_img = mesh_coord_img

                if not self.return_coords_meshnet:
                    volumes = compose(None, joint_coord_img, 'tt', joint_independent=self.joint_independent_meshnet)
                    joint_coord_img, volumes_pred = integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                                       coordinates[0],
                                                                                       softmax=self.volume_softmax)


        return (joint_coord_img, 
                None, 
                None, 
                None,
                None,
                base_points)

       
