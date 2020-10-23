# Reference: https://github.com/dragonbook/V2V-PoseNet-pytorch
from IPython.core.debugger import set_trace
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import inspect
import torch
import sys 
import numpy as np
from mvn.utils.misc import get_capacity

ACTIVATION_TYPE = 'LeakyReLU'
MODULES_REQUIRES_NORMALIZAION = ['Res3DBlock', 'Upsample3DBlock', 'Basic3DBlock']
ORDINARY_NORMALIZATIONS = ['group_norm', 'batch_norm']

class SqueezeLayer(nn.Module):
    def __init__(self, squeeze_dim):
        super(SqueezeLayer, self).__init__()
        super().__init__()
        self.squeeze_dim = squeeze_dim
    def forward(self, x):
        return x.squeeze(self.squeeze_dim)    

def get_activation(activation_type):
    return {'LeakyReLU':nn.LeakyReLU,'ReLU':nn.ReLU}[activation_type]

def get_normalization(normalization_type, features_channels, n_groups=32):

        if normalization_type ==  'batch_norm':
            return nn.BatchNorm3d(features_channels)
        elif normalization_type == 'group_norm':
            return nn.GroupNorm(n_groups, features_channels)
        else:
            raise RuntimeError('{} is unknown normalization_type'.format(normalization_type))          


class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, normalization_type, n_groups=32):
        super(Basic3DBlock, self).__init__()
        self.normalization_type = normalization_type
        self.conv =  nn.Conv3d(in_planes, out_planes, 
                                kernel_size=kernel_size, 
                                stride=1, 
                                padding=((kernel_size-1)//2))

        self.normalization = get_normalization(normalization_type, out_planes, n_groups)
        self.activation = get_activation(ACTIVATION_TYPE)(True)

    def forward(self, x):

        x = self.conv(x)
        x = self.normalization(x)
        x = self.activation(x)

        return x 


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, normalization_type, kernel_size=3, n_groups=32, padding=1):
        super(Res3DBlock, self).__init__()

        self.normalization_type = normalization_type
        
        self.res_norm1 = get_normalization(normalization_type, out_planes, n_groups)
        self.res_norm2 = get_normalization(normalization_type, out_planes, n_groups)

        self.res_conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding)
        self.res_conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding)
        
        self.activation = get_activation(ACTIVATION_TYPE)(True)

        self.use_skip_con = (in_planes != out_planes)
        if self.use_skip_con:
            self.skip_con_conv = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
            self.skip_con_norm = get_normalization(normalization_type, out_planes, n_groups)    

    def forward(self, x):
        if self.use_skip_con:
            skip = self.skip_con_conv(x)
            skip  = self.skip_con_norm(skip)
        else:
            skip = x

        x = self.res_conv1(x)
        x = self.res_norm1(x)
        x = self.activation(x)

        x = self.res_conv2(x)
        x = self.res_norm2(x)

        x = F.relu(x + skip, True)

        return x       


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        x = F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size) 
        return x  


class Upsample3DBlock(nn.Module):
    def __init__(self, 
                in_planes, 
                out_planes, 
                kernel_size, 
                stride, 
                normalization_type, 
                n_groups=32):
        super().__init__()
        # assert(kernel_size == 2)
        # assert(stride == 2)
        self.normalization_type = normalization_type
        self.transpose = nn.ConvTranspose3d(in_planes, 
                                            out_planes, 
                                            kernel_size=kernel_size, 
                                            stride=stride, 
                                            padding=0, 
                                            output_padding=0)

        self.normalization = get_normalization(normalization_type, out_planes, n_groups)
        self.activation = get_activation(ACTIVATION_TYPE)(True)

    def forward(self, x):
        x = self.transpose(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x  

class TuckerBasisNet(nn.Module):
    """docstring for TuckerBasisNet"""
    def __init__(self, input_dim, volume_size, n_joints, normalization_type):
        super(TuckerBasisNet, self).__init__()
        self.input_dim = input_dim
        self.volume_size = volume_size
        self.n_joints = n_joints

        self.resblock_1 = Res3DBlock(input_dim, 256, normalization_type, kernel_size=3, n_groups=32, padding=1)
        self.pool_block = Pool3DBlock(2)
        self.resblock_2 = Res3DBlock(256, 1024, normalization_type, kernel_size=3, n_groups=32, padding=1)
        self.avg_pool = torch.nn.AdaptiveMaxPool3d((1,1,1))

        self.lj = nn.Linear(1024, n_joints**2)
        self.lx = nn.Linear(1024, volume_size**2)
        self.ly = nn.Linear(1024, volume_size**2)
        self.lz = nn.Linear(1024, volume_size**2)


    def forward(self, x):

        batch_size = x.shape[0]
        x = self.resblock_1(x)
        x = self.pool_block(x)
        x = self.resblock_2(x)
        x = self.avg_pool(x).view(batch_size, -1)
        # x is a vector

        j_ = self.lj(x).view(batch_size, self.n_joints, self.n_joints)

        x_ = self.lx(x).view(batch_size, self.volume_size, self.volume_size)
        y_ = self.ly(x).view(batch_size, self.volume_size, self.volume_size)
        z_ = self.lz(x).view(batch_size, self.volume_size, self.volume_size)

        return [j_, x_, y_, z_]




class EncoderDecorder(nn.Module):

    def __init__(self, config, normalization_type, return_bottleneck=False):
        super().__init__()

        '''
        default_normalization_type - used where is no adaptive normalization
        '''
        self.return_bottleneck = return_bottleneck
        self.normalization_type = normalization_type
        self.use_skip_connection = config.use_skip_connection if hasattr(config, 'use_skip_connection') else True
        if self.use_skip_connection:
            skip_block_type = config.skip_block_type
            upsampling_config = config.upsampling

        downsampling_blocks_numerator = {'Res3DBlock':0, 'Pool3DBlock':0, 'Upsample3DBlock':0}
        upsampling_blocks_numerator = {'Res3DBlock':0, 'Pool3DBlock':0, 'Upsample3DBlock':0}
        bottleneck_blocks_numerator = {'Res3DBlock':0, 'Pool3DBlock':0, 'Upsample3DBlock':0}

        if hasattr(config, 'downsampling'):
            self.downsampling_dict = nn.ModuleDict()
            downsampling_config = config.downsampling
            for i, module_config in enumerate(downsampling_config):
                
                self._append_module(self.downsampling_dict, module_config, downsampling_blocks_numerator)
                module_type = module_config['module_type']
                module_number = str(downsampling_blocks_numerator[module_type])

                # add skip-connection
                if module_type == 'Res3DBlock' and self.use_skip_connection:
                    in_planes = module_config['params'][-1]
                    out_planes = upsampling_config[-i-1]['params'][0]

                    skip_con_block = getattr(sys.modules[__name__], skip_block_type)(in_planes, 
                                                                                     out_planes, 
                                                                                     normalization_type=self.normalization_type)
                    
                    name = ''.join(('skip', '_', module_type, '_', module_number))
                    self.downsampling_dict[name] = skip_con_block
        
        if hasattr(config, 'bottleneck'):
            self.bottleneck_dict = nn.ModuleDict()
            bottleneck_config = config.bottleneck
            for i, module_config in enumerate(bottleneck_config):  
                self._append_module(self.bottleneck_dict, module_config, bottleneck_blocks_numerator)
        
        if hasattr(config, 'upsampling'):
            self.upsampling_dict = nn.ModuleDict()
            for i, module_config in enumerate(upsampling_config):
                self._append_module(self.upsampling_dict, module_config, upsampling_blocks_numerator)

    def _append_module(self, modules_dict, module_config, blocks_numerator):
            
            module_type = module_config['module_type']
            module_number = str(blocks_numerator[module_type])
            blocks_numerator[module_type] += 1
            constructor = getattr(sys.modules[__name__], module_config['module_type'])
            
            # construct params dict                         
            model_params = {}
            # get arguments and omitting `self`
            arguments = inspect.getargspec(constructor.__init__).args[1:]
            for i, param in enumerate(module_config['params']): 
                # get arguments for __init__ function
                param_name = arguments[i]                            
                model_params[param_name] = param

            # add some optional arguments for __init__    
            if module_type in MODULES_REQUIRES_NORMALIZAION:
                model_params['normalization_type'] = self.normalization_type

            module = constructor(**model_params)
            name = ''.join((module_type, '_', module_number))
            modules_dict[name] = module

    def forward(self, x):

        if hasattr(self, 'downsampling_dict'):
            skip_connections = []
            for name, module in self.downsampling_dict.items():
                if name.split('_')[0] == 'skip' and self.use_skip_connection:
                    skip_connections.append(module(x))
                else:
                    x = module(x)

        bottleneck = None
        if hasattr(self, 'bottleneck_dict'):                
            for name, module in self.bottleneck_dict.items():
                x = module(x)
            if self.return_bottleneck:
                bottleneck = x
        
        if hasattr(self, 'upsampling_dict'):
            skip_number = -1
            n_ups_blocks = len(self.upsampling_dict)
            for i, (name, module)in enumerate(self.upsampling_dict.items()):
                if name.split('_')[0] == 'Res3DBlock' and self.use_skip_connection:
                    skip_connection = skip_connections[skip_number]
                    if x.shape[-3:] != skip_connection.shape[-3:]:
                        skip_connection = F.interpolate(skip_connection,
                                                        size=(x.shape[-3:]), 
                                                        mode='trilinear')
                    x = x + skip_connection
                    skip_number -= 1
                x = module(x)
        return x, bottleneck                  


class V2VModel(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels, 
                 v2v_normalization_type, 
                 config,
                 back_layer_output_channels=64,
                 return_bottleneck=False, 
                 output_vector=False
                 ):

        super().__init__()

        self.normalization_type = v2v_normalization_type
        self.return_bottleneck = return_bottleneck
        self.output_vector = output_vector

        encoder_input_channels = config.downsampling[0]['params'][0] if hasattr(config, 'downsampling') else config.encoder_input_channels
        encoder_output_channels = config.upsampling[-1]['params'][1] if hasattr(config, 'upsampling') else config.encoder_output_channels

        self.front_layer = Basic3DBlock(input_channels, 
                                        encoder_input_channels, 
                                        3, # kernel size
                                        self.normalization_type)

        self.encoder_decoder = EncoderDecorder(config,
                                               normalization_type=self.normalization_type,
                                               return_bottleneck=self.return_bottleneck)

        if self.output_vector:
            self.amp = torch.nn.AdaptiveMaxPool3d((1,1,1))

        self.back_layer = Basic3DBlock(encoder_output_channels, 
                                       back_layer_output_channels, 
                                       1, # kernel size
                                       self.normalization_type)

        self.output_layer = nn.Conv3d(back_layer_output_channels, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):

        x = self.front_layer(x)
        x, bottleneck = self.encoder_decoder(x)    

        x = self.back_layer(x)
        x = self.output_layer(x)
        
        if self.output_vector:
            x = self.amp(x).squeeze(-1).squeeze(-1).squeeze(-1)

        return x, bottleneck

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)

