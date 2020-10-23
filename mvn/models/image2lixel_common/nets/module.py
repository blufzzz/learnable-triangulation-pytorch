import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from mvn.models.image2lixel_common.config import cfg
import torchgeometry as tgm
from mvn.models.image2lixel_common.nets.layer import make_conv_layers, make_deconv_layers, make_conv1d_layers, make_linear_layers

from IPython.core.debugger import set_trace

class PoseNet(nn.Module):
    def __init__(self, joint_num, volume_size, input_features=256, normalization_type='group_norm'):
        super(PoseNet, self).__init__()
        self.joint_num = joint_num
        self.normalization_type = normalization_type
        self.volume_size = volume_size
        if self.volume_size == 32:
            deconv_layers_channels = [2048,256,256]
        elif self.volume_size == 64:
            deconv_layers_channels = [2048,256,256, 256]
        else:
            raise RuntimeError('wrong `volume_size`')

        self.deconv = make_deconv_layers(deconv_layers_channels)
        self.conv_x = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False, normalization_type=normalization_type)
        self.conv_y = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False, normalization_type=normalization_type)
        self.conv_z_1 = make_conv1d_layers([2048,256*self.volume_size], kernel=1, stride=1, padding=0, normalization_type=normalization_type) # k=3, p=1
        self.conv_z_2 = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False, normalization_type=normalization_type)

    def soft_argmax_1d(self, heatmap1d, grid=None):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        if grid is not None:
            coord = torch.einsum("bjx, bx -> bj", heatmap1d, grid).unsqueeze(-1)
        else: 
            coord = heatmap1d * torch.cuda.comm.broadcast(torch.arange(heatmap_size).type(torch.cuda.FloatTensor), 
                                                            devices=[heatmap1d.device.index])[0]
            coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat, coordinates=None):

        x,y,z = None, None, None
        if coordinates is not None:
            x,y,z = coordinates

        img_feat_xy = self.deconv(img_feat)

        # x axis
        img_feat_x = img_feat_xy.mean((2))
        heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax_1d(heatmap_x, x)
        
        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y, y)
        
        # z axis
        img_feat_z = img_feat.mean((2,3))[:,:,None]
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1,256,self.volume_size)
        heatmap_z = self.conv_z_2(img_feat_z)
        coord_z = self.soft_argmax_1d(heatmap_z ,z)

        joint_coord = torch.cat((coord_x, coord_y, coord_z),2)

        return joint_coord


class PoseNet3D(nn.Module):
    def __init__(self, rank, size, joint_num, input_features=256, intermediate_features=256, normalization_type='group_norm', use_G_j = False):
        super(PoseNet3D, self).__init__()
        self.joint_num = joint_num
        self.rank = rank
        assert rank == 2
        self.size = size
        self.use_G_j = use_G_j
        self.intermediate_features = intermediate_features
        self.input_features = input_features
        self.normalization_type = normalization_type

        self.x_branch = self.make_branch(input_features, intermediate_features, 1, rank, kernel=[2,1,2], stride=[2,1,2], padding=0, bnrelu_final=False, normalization_type=normalization_type)
        self.y_branch = self.make_branch(input_features, intermediate_features, 1, rank, kernel=[1,2,2], stride=[1,2,2], padding=0, bnrelu_final=False, normalization_type=normalization_type)
        self.z_branch = self.make_branch(input_features, intermediate_features, 1, rank, kernel=[2,2,1], stride=[2,2,1], padding=0, bnrelu_final=False, normalization_type=normalization_type)
        self.j_branch = self.make_branch(input_features, intermediate_features, intermediate_features, rank=1, kernel=[2,2,2], stride=[2,2,2], padding=0, bnrelu_final=False, normalization_type=normalization_type)

        self.G_j_layer = nn.Linear(intermediate_features, joint_num*rank)
        self.G_z_layer = nn.Linear(size*(rank**2), size*rank)

    def make_branch(self, input_features, intermediate_features, output_features, rank, kernel=1, stride=1, padding=0, bnrelu_final=False, normalization_type='group_norm'):
        layers = []
        n_layers = int(np.log2(self.size))
        n_layers = n_layers-1 if rank==2 else n_layers
        for i in range(n_layers):
            out_channel = intermediate_features if i < n_layers-1 else output_features
            in_channel = intermediate_features if i > 0 else input_features

            layers.append(nn.Conv3d(in_channel, in_channel, kernel_size=kernel, stride=stride, padding=padding))
            layers.append(nn.LeakyReLU())
            layers.append(nn.GroupNorm(32, in_channel) \
                    if normalization_type=='group_norm' else \
                    nn.BatchNorm3d(in_channel)
                    )

            layers.append(nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0))
            layers.append(nn.LeakyReLU())

            # avoid normalization on the last layer
            if i < n_layers-1:
                layers.append(nn.GroupNorm(32, out_channel) \
                        if normalization_type=='group_norm' else \
                        nn.BatchNorm3d(out_channel)
                        )

        return nn.Sequential(*layers)

    def forward(self, img_feat_xyz, coordinates=None):

        batch_size = img_feat_xyz.shape[0]
        size = img_feat_xyz.shape[-1]
        x,y,z = None, None, None
        if coordinates is not None:
            x,y,z = coordinates

        img_feat_x = self.x_branch(img_feat_xyz).squeeze(1) # squeeze channel dim
        img_feat_y = self.y_branch(img_feat_xyz).squeeze(1)
        img_feat_z = self.z_branch(img_feat_xyz).squeeze(1) # squeeze channel and first dim
        img_feat_z = self.G_z_layer(img_feat_z.view(batch_size, -1))
        img_feat_z = img_feat_z.view(batch_size, self.rank, self.size)
        img_feat_j = self.j_branch(img_feat_xyz).squeeze(1)

        img_feat_j = self.G_j_layer(img_feat_j.squeeze(-1).squeeze(-1).squeeze(-1))
        img_feat_j = img_feat_j.view(batch_size, self.joint_num, self.rank)

        return img_feat_j, img_feat_x, torch.transpose(img_feat_y, 1,2), img_feat_z



class Pose2Feat(nn.Module):
    def __init__(self, volume_size, joint_num):
        super(Pose2Feat, self).__init__()
        self.joint_num = joint_num
        self.volume_size = volume_size
        self.conv = make_conv_layers([64+joint_num*self.volume_size,64])

    def forward(self, img_feat, joint_heatmap_3d):
        joint_heatmap_3d = joint_heatmap_3d.view(-1,self.joint_num*self.volume_size,self.volume_size,self.volume_size)
        feat = torch.cat((img_feat, joint_heatmap_3d),1)
        feat = self.conv(feat)
        return feat

class MeshNet(nn.Module):
    def __init__(self, volume_size, vertex_num):
        super(MeshNet, self).__init__()
        self.vertex_num = vertex_num
        self.volume_size = volume_size

         if self.volume_size == 32:
            deconv_layers_channels = [2048,256,256]
        elif self.volume_size == 64:
            deconv_layers_channels = [2048,256,256, 256]
        else:
            raise RuntimeError('wrong `volume_size`')

        self.deconv = make_deconv_layers(deconv_layers_channels)
        self.conv_x = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([2048,256*self.volume_size], kernel=1, stride=1, padding=0) 
        self.conv_z_2 = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False) 

    def soft_argmax_1d(self, heatmap1d, grid=None):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        if grid is not None:
            coord = torch.einsum("bjx, bx -> bj", heatmap1d, grid).unsqueeze(-1)
        else: 
            coord = heatmap1d * torch.cuda.comm.broadcast(torch.arange(heatmap_size).type(torch.cuda.FloatTensor), 
                                                            devices=[heatmap1d.device.index])[0]
            coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat, coordinates=None):

        x,y,z = None, None, None
        if coordinates is not None:
            x,y,z = coordinates

        img_feat_xy = self.deconv(img_feat)

        # x axis
        img_feat_x = img_feat_xy.mean((2))
        heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax_1d(heatmap_x ,x)
        
        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y, y)
        
        # z axis
        img_feat_z = img_feat.mean((2,3))[:,:,None]
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1,256,self.volume_size)
        heatmap_z = self.conv_z_2(img_feat_z) # problm
        coord_z = self.soft_argmax_1d(heatmap_z, z)

        mesh_coord = torch.cat((coord_x, coord_y, coord_z),2)
        return mesh_coord

class ParamRegressor(nn.Module):
    def __init__(self, joint_num):
        super(ParamRegressor, self).__init__()
        self.joint_num = joint_num
        self.fc = make_linear_layers([self.joint_num*3, 1024, 512], use_bn=True)
        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.fc_pose = make_linear_layers([512, 16*6], relu_final=False) # hand joint orientation
        else:
            self.fc_pose = make_linear_layers([512, 24*6], relu_final=False) # body joint orientation
        self.fc_shape = make_linear_layers([512, 10], relu_final=False) # shape parameter

    def rot6d_to_rotmat(self,x):
        x = x.view(-1,3,2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2)
        return torch.stack((b1, b2, b3), dim=-1)

    def forward(self, pose_3d):
        pose_3d = pose_3d.view(-1,self.joint_num*3)
        feat = self.fc(pose_3d)

        pose = self.fc_pose(feat)
        pose = self.rot6d_to_rotmat(pose)
        pose = torch.cat([pose,torch.zeros((pose.shape[0],3,1)).cuda().float()],2)
        pose = tgm.rotation_matrix_to_angle_axis(pose).reshape(-1,72)
        
        shape = self.fc_shape(feat)

        return pose, shape
