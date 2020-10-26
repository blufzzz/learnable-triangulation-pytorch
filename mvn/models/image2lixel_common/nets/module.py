import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from mvn.models.image2lixel_common.config import cfg
import torchgeometry as tgm
from mvn.models.image2lixel_common.nets.layer import make_conv_layers, make_deconv_layers, make_conv1d_layers, make_linear_layers
from mvn.utils.op import get_kernels
from IPython.core.debugger import set_trace

class PoseNet(nn.Module):
    def __init__(self, 
                joint_num, 
                volume_size,
                return_coords, 
                joint_independent=True, 
                rank=1, 
                input_features=256, 
                normalization_type='group_norm'):

        super(PoseNet, self).__init__()
        self.joint_num = joint_num
        self.normalization_type = normalization_type
        self.return_coords = return_coords
        self.rank = rank
        self.joint_independent = joint_independent
        self.volume_size = volume_size
        if self.volume_size == 32:
            deconv_layers_channels = [2048,256,256]
        elif self.volume_size == 64:
            deconv_layers_channels = [2048,256,256, 256]
        else:
            raise RuntimeError('wrong `volume_size`')

        if not return_coords:
            x_channels = self.joint_num*self.rank if joint_independent else self.rank
            y_channels = self.joint_num*(self.rank**2) if joint_independent else self.rank**2
            z_channels = self.joint_num*self.rank if joint_independent else self.rank
        else:
            x_channels, y_channels, z_channels = self.joint_num, self.joint_num, self.joint_num

        self.deconv = make_deconv_layers(deconv_layers_channels)
        self.conv_x = make_conv1d_layers([256,x_channels], kernel=1, stride=1, padding=0, bnrelu_final=False, normalization_type=normalization_type)
        self.conv_y = make_conv1d_layers([256,y_channels], kernel=1, stride=1, padding=0, bnrelu_final=False, normalization_type=normalization_type)
        self.conv_z_1 = make_conv1d_layers([2048,256*self.volume_size], kernel=1, stride=1, padding=0, normalization_type=normalization_type) # k=3, p=1
        self.conv_z_2 = make_conv1d_layers([256,z_channels], kernel=1, stride=1, padding=0, bnrelu_final=False, normalization_type=normalization_type)

        # if not joint_independent:
        #     self.conv_j = ...

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

        batch_size = img_feat.shape[0]
        x,y,z = None, None, None
        if coordinates is not None:
            x,y,z = coordinates

        img_feat_xy = self.deconv(img_feat)

        # x axis
        img_feat_x = img_feat_xy.mean((2))
        heatmap_x = self.conv_x(img_feat_x)
        
        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y)
        
        # z axis
        img_feat_z = img_feat.mean((2,3))[:,:,None]
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1,256,self.volume_size)
        heatmap_z = self.conv_z_2(img_feat_z)

        if self.return_coords:
            coord_x = self.soft_argmax_1d(heatmap_x, x)
            coord_y = self.soft_argmax_1d(heatmap_y, y)
            coord_z = self.soft_argmax_1d(heatmap_z ,z)
            joint_coord = torch.cat((coord_x, coord_y, coord_z),2)
            return joint_coord

        else:
            if self.joint_independent:
                heatmap_x = heatmap_x.view(batch_size, self.joint_num, -1, self.rank)
                heatmap_y = heatmap_y.view(batch_size, self.joint_num, self.rank, -1, self.rank)
                heatmap_z = heatmap_z.view(batch_size, self.joint_num, self.rank, -1)
            else:
                pass

            return heatmap_x, heatmap_y, heatmap_z


class PoseNetTT(nn.Module):
    def __init__(self, rank, volume_size, joint_num, input_features=256, intermediate_features=256, kernel_size=2, normalization_type='group_norm', joint_independent=False):
        super(PoseNetTT, self).__init__()
        self.joint_num = joint_num
        self.rank = rank
        self.volume_size = volume_size
        self.joint_independent = joint_independent
        self.intermediate_features = intermediate_features
        self.input_features = input_features
        self.normalization_type = normalization_type



        x_rank = 1 if self.joint_independent else rank
        out_channels = joint_num if self.joint_independent else 1
        self.x_branch = self.make_branch(input_features, intermediate_features, out_channels, [volume_size, volume_size, volume_size], [x_rank, volume_size, rank], 
                                        kernel_size=2, bnrelu_final=False, normalization_type=normalization_type)
        self.y_branch = self.make_branch(input_features, intermediate_features, out_channels, [volume_size, volume_size, volume_size], [rank, volume_size, rank], 
                                        kernel_size=2, bnrelu_final=False, normalization_type=normalization_type)
        self.z_branch = self.make_branch(input_features, intermediate_features, out_channels, [volume_size, volume_size, volume_size], [rank, volume_size, 1], 
                                        kernel_size=2, bnrelu_final=False, normalization_type=normalization_type)

        if not self.joint_independent:
            raise RuntimeError()
            # self.j_branch = self.make_branch(input_features, intermediate_features, intermediate_features, rank=1, kernel_size=2 bnrelu_final=False, normalization_type=normalization_type)
            # self.G_j_layer = nn.Linear(intermediate_features, joint_num*rank)

    def make_branch(self, input_features, intermediate_features, output_features, input_size, output_size, kernel_size=2, bnrelu_final=True, normalization_type='group_norm'):
        

        kernels, strides = get_kernels(input_size, output_size, kernel_size)
        layers = []
        n_layers = len(kernels)

        for i,(k,s) in enumerate(zip(kernels, strides)):

            out_channel = intermediate_features if i < n_layers-1 else output_features
            in_channel = intermediate_features if i > 0 else input_features

            layers.append(nn.Conv3d(in_channel, in_channel, kernel_size=k, stride=s, padding=0))
            layers.append(nn.LeakyReLU())
            layers.append(nn.GroupNorm(32, in_channel) \
                    if normalization_type=='group_norm' else \
                    nn.BatchNorm3d(in_channel)
                    )

            layers.append(nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0))
            layers.append(nn.LeakyReLU())

            if (i < n_layers-1) or bnrelu_final:
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

        img_feat_x = self.x_branch(img_feat_xyz) # squeeze channel dim
        img_feat_y = self.y_branch(img_feat_xyz)
        img_feat_z = self.z_branch(img_feat_xyz) # squeeze channel and first dim
        
        if not self.joint_independent:
            img_feat_j = self.j_branch(img_feat_xyz).squeeze(1)
            img_feat_j = self.G_j_layer(img_feat_j.squeeze(-1).squeeze(-1).squeeze(-1))
            img_feat_j = img_feat_j.view(batch_size, self.joint_num, self.rank)

        else:
            return img_feat_x.squeeze(1), img_feat_y, img_feat_z.squeeze(-1)



class Pose2Feat(nn.Module):
    def __init__(self, joint_num):
        super(Pose2Feat, self).__init__()
        self.joint_num = joint_num
        self.conv = make_conv_layers([64+joint_num*64,64])

    def forward(self, img_feat, joint_heatmap_3d):
        batch_size = joint_heatmap_3d.shape[0] 
        joint_heatmap_3d = joint_heatmap_3d.view(batch_size,-1, *joint_heatmap_3d.shape[-2:])
        feat = torch.cat((img_feat, joint_heatmap_3d),1)
        feat = self.conv(feat)
        return feat

class MeshNet(nn.Module):
    def __init__(self, volume_size, vertex_num, return_coords, rank=1, joint_independent=True, normalization_type='group_norm'):
        super(MeshNet, self).__init__()
        self.vertex_num = vertex_num
        self.volume_size = volume_size
        self.return_coords = return_coords
        if self.volume_size == 32:
            deconv_layers_channels = [2048,256,256]
        elif self.volume_size == 64:
            deconv_layers_channels = [2048,256,256,256]
        else:
            raise RuntimeError('wrong `volume_size`')

        self.deconv = make_deconv_layers(deconv_layers_channels)
        self.conv_x = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False, normalization_type=normalization_type)
        self.conv_y = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False, normalization_type=normalization_type)
        self.conv_z_1 = make_conv1d_layers([2048,256*self.volume_size], kernel=1, stride=1, padding=0, normalization_type=normalization_type) 
        self.conv_z_2 = make_conv1d_layers([256,self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False, normalization_type=normalization_type) 

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

        if self.return_coords:
            coord_x = self.soft_argmax_1d(heatmap_x, x) # x: 1d coord grid [bs,32]; heatmap_x: [bs,17,32]
            coord_y = self.soft_argmax_1d(heatmap_y, y) # y: 1d coord grid [bs,32]; heatmap_y: [bs,17,32]
            coord_z = self.soft_argmax_1d(heatmap_z ,z) # z: 1d coord grid [bs,32]; heatmap_z: [bs,17,32]
            joint_coord = torch.cat((coord_x, coord_y, coord_z),2)
            return joint_coord

        else:
            return heatmap_x, heatmap_y, heatmap_z

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
