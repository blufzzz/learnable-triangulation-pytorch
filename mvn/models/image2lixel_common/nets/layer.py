import torch
import torch.nn as nn
from mvn.models.image2lixel_common.config import cfg

def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv1d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True, normalization_type='batch_norm'):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv1d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            if normalization_type == 'batch_norm':
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            elif normalization_type == 'group_norm':
                layers.append(nn.GroupNorm(32, feat_dims[i+1]))
            else:
                raise RuntimeError()
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_deconv_layers(feat_dims, bnrelu_final=True, normalization_type='batch_norm', dim=2):
    layers = []
    for i in range(len(feat_dims)-1):
        if d == 2:
            layer = nn.ConvTranspose2d(
                        in_channels=feat_dims[i],
                        out_channels=feat_dims[i+1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        output_padding=0,
                        bias=False)
        elif d == 3:
            layer = nn.ConvTranspose3d(
                        in_channels=feat_dims[i],
                        out_channels=feat_dims[i+1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        output_padding=0,
                        bias=False)
        else:
            raise RuntimeError('wrong `d`')
        layers.append(layer)

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            if normalization_type == 'batch_norm':
                if d == 2:
                    bn = nn.BatchNorm2d(feat_dims[i+1])
                elif d == 3:
                    bn = nn.BatchNorm3d(feat_dims[i+1])
                else:
                    raise RuntimeError('wrong `d`')
                layers.append(bn)
            elif normalization_type == 'group_norm':
                layers.append(nn.GroupNorm(32, feat_dims[i+1]))
            else:
                raise RuntimeError()
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

