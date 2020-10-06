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
ADAPTIVE_NORMALIZATION = ['adain', 'adain_mlp', 'spade']
SPADE_PARAMS_EVOLUTION = False
STYLE_FORWARD = False


def registed_forward_hook(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Conv3d) and (np.array(child.stride) > 1).any():
            child.stride = (1,1,1)
        else:
            change_stride(child)


def change_stride(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Conv3d) and (np.array(child.stride) > 1).any():
            child.stride = (1,1,1)
        else:
            change_stride(child)

def convert_relu_to_leakyrelu(model): 
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.LeakyReLU())
        else:
            convert_relu_to_leakyrelu(child)                 

def convert_bn_to_gn(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm3d):
            C = child.num_features
            setattr(model, child_name, nn.GroupNorm(C,C))
        else:
            convert_bn_to_gn(child)


class SqueezeLayer(nn.Module):
    def __init__(self, squeeze_dim):
        super(SqueezeLayer, self).__init__()
        super().__init__()
        self.squeeze_dim = squeeze_dim
    def forward(self, x):
        return x.squeeze(self.squeeze_dim)    
        


class SPADE(nn.Module):
    def __init__(self, 
                style_vector_channels, 
                features_channels, 
                hidden=64, # hidden=128
                ks=3, 
                params_evolution=False):

        super().__init__()

        self.params_evolution = SPADE_PARAMS_EVOLUTION
        pw = ks // 2
        self.shared = nn.Sequential(
            nn.Conv3d(features_channels if STYLE_FORWARD else style_vector_channels, hidden, kernel_size=ks, padding=pw),
            get_activation(ACTIVATION_TYPE)()
        )
        self.gamma = nn.Conv3d(hidden, features_channels, kernel_size=ks, padding=pw)
        self.beta = nn.Conv3d(hidden, features_channels, kernel_size=ks, padding=pw)
        self.bn = nn.InstanceNorm3d(features_channels, affine=False)
        if self.params_evolution:
            print ('params_evolution inited')
            self.params_evolution_block = nn.Sequential(
                                            nn.Conv3d(style_vector_channels,
                                                      style_vector_channels, 
                                                      kernel_size=ks, 
                                                      padding=pw),
                                            get_activation(ACTIVATION_TYPE)(),
                                            nn.GroupNorm(32, style_vector_channels)
                                            )

    def forward(self, x, params, params_evolution=False):

        if params is None:
            return x

        batch_size = x.shape[0]
        if params_evolution:
            assert self.params_evolution
        if STYLE_FORWARD:
            assert not self.params_evolution
            params = x[batch_size//2:]
            x = x[:batch_size//2] 

        params = F.interpolate(params, size=x.size()[2:], mode='trilinear')
        actv = self.shared(params)
        gamma = self.gamma(actv)
        beta = self.beta(actv)
        x = self.bn(x) * (1 + gamma) + beta

        if STYLE_FORWARD:
            x = torch.cat([x,params],0)

        return x if not params_evolution else (x, self.params_evolution_block(params))


class AdaIN(nn.Module):
    def __init__(self, 
                style_vector_channels, 
                features_channels, 
                use_affine_mlp=False, 
                hidden_mlp=512):

        super(AdaIN, self).__init__()
        self.use_affine_mlp = use_affine_mlp
        if self.use_affine_mlp:
            self.affine = nn.Sequential(nn.Linear(style_vector_channels, hidden_mlp),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_mlp, 2*features_channels))
        else:
            self.affine = nn.Linear(style_vector_channels, 2*features_channels)

    def forward(self, features, params, eps = 1e-15, params_evolution=False):
        # features: [batch_size, C, D1, D2, D3]
        # params: [batch_size, 2*С]
        adain_params = self.affine(params)
        batch_size, C = features.shape[:2]

        unbiased = features.view(batch_size, C, -1).shape[-1] == 1

        adain_mean = adain_params[:,:C].view(batch_size, C,1,1,1)
        adain_std = adain_params[:,C:].view(batch_size, C,1,1,1)
    
        features_mean = features.view(batch_size, C, -1).mean(-1).view(batch_size, C,1,1,1)
        features_std = features.view(batch_size, C, -1).std(-1, unbiased=unbiased).view(batch_size, C,1,1,1)

        out = ((features - features_mean) / (features_std + eps)) * adain_std + adain_mean

        return out if not params_evolution else (out, params)


class CompoundNorm(nn.Module):
    def __init__(self, normalization_types, out_planes, n_groups, style_vector_channels):
        super().__init__()
        norm_type, adaptive_norm_type = normalization_types
        assert norm_type in ORDINARY_NORMALIZATIONS and adaptive_norm_type in ADAPTIVE_NORMALIZATION
        self.norm = get_normalization(norm_type, out_planes, n_groups, style_vector_channels)
        self.adaptive_norm = get_normalization(adaptive_norm_type, out_planes, n_groups, style_vector_channels)
    def forward(self, x, params, params_evolution=False):
        
        x = self.norm(x)
        if params_evolution:
            x, params = self.adaptive_norm(x, params, params_evolution)
        else:    
            x = self.adaptive_norm(x, params, params_evolution)
        return x if not params_evolution else (x, params)       

class GroupNorm(nn.Module):
    def __init__(self, n_groups, features_channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(n_groups, features_channels)
    def forward(self, x, params=None, params_evolution=False):
        x = self.group_norm(x)
        return x if not params_evolution else (x, params)      

class BatchNorm3d(nn.Module):
    def __init__(self, features_channels):
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(features_channels, affine=False)
    def forward(self, x, params=None, params_evolution=False):
        x = self.batch_norm(x)
        return x if not params_evolution else (x, params)           
        

def get_activation(activation_type):
    return {'LeakyReLU':nn.LeakyReLU,'ReLU':nn.ReLU}[activation_type]

def get_normalization(normalization_type, features_channels, n_groups=32, style_vector_channels=None):

    if type(normalization_type) is list:
        return CompoundNorm(normalization_type, features_channels, n_groups, style_vector_channels)
    else:    
        if normalization_type == 'adain':
            return AdaIN(style_vector_channels, features_channels, use_affine_mlp=False)
        if normalization_type == 'adain_mlp':
            return AdaIN(style_vector_channels, features_channels, use_affine_mlp=True)    
        elif normalization_type ==  'batch_norm':
            return BatchNorm3d(features_channels)
        elif normalization_type ==  'spade':
            return SPADE(style_vector_channels, features_channels)    
        elif normalization_type == 'group_norm':
            return GroupNorm(n_groups, features_channels)
        else:
            raise RuntimeError('{} is unknown normalization_type'.format(normalization_type))          


class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, normalization_type, n_groups=32, style_vector_channels=None):
        super(Basic3DBlock, self).__init__()
        self.normalization_type = normalization_type
        self.conv =  nn.Conv3d(in_planes, out_planes, 
                                kernel_size=kernel_size, 
                                stride=1, 
                                padding=((kernel_size-1)//2))

        self.normalization = get_normalization(normalization_type, out_planes, n_groups, style_vector_channels)

        self.activation = get_activation(ACTIVATION_TYPE)(True)

    def forward(self, x, params=None, params_evolution=False):

        x = self.conv(x)
        if params_evolution:
            x, params = self.normalization(x, params, params_evolution)
        else:
            x = self.normalization(x, params)
        x = self.activation(x)

        return  x if not params_evolution else (x, params)  


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, normalization_type, kernel_size=3, n_groups=32, padding=1, style_vector_channels=None):
        super(Res3DBlock, self).__init__()

        self.normalization_type = normalization_type
        
        self.res_norm1 = get_normalization(normalization_type, out_planes, n_groups, style_vector_channels)
        self.res_norm2 = get_normalization(normalization_type, out_planes, n_groups, style_vector_channels)

        self.res_conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding)
        self.res_conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding)
        
        self.activation = get_activation(ACTIVATION_TYPE)(True)

        self.use_skip_con = (in_planes != out_planes)
        if self.use_skip_con:
            self.skip_con_conv = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
            self.skip_con_norm = get_normalization(normalization_type, out_planes, n_groups, style_vector_channels)    

    def forward(self, x, params=None, params_evolution=False):
        if self.use_skip_con:
            skip = self.skip_con_conv(x)
            if params_evolution:
                skip,params  = self.skip_con_norm(skip, params, params_evolution)
            else:
                skip  = self.skip_con_norm(skip, params, params_evolution)
        else:
            skip = x

        x = self.res_conv1(x)
        if params_evolution:
            x,params = self.res_norm1(x, params, params_evolution)
        else:
            x = self.res_norm1(x, params)
        x = self.activation(x)

        x = self.res_conv2(x)
        if params_evolution:
            x,params = self.res_norm2(x, params, params_evolution)
        else:
            x = self.res_norm2(x, params)

        x = F.relu(x + skip, True)

        return x if not params_evolution else (x, params)        


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x, params=None, params_evolution=False):
        x = F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size) 
        return x if not params_evolution else (x, params) 


class Upsample3DBlock(nn.Module):
    def __init__(self, 
                in_planes, 
                out_planes, 
                kernel_size, 
                stride, 
                normalization_type, 
                n_groups=32, 
                style_vector_channels=None):
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

        self.normalization = get_normalization(normalization_type, out_planes, n_groups, style_vector_channels)
        self.activation = get_activation(ACTIVATION_TYPE)(True)

    def forward(self, x, params=None, params_evolution=False):
        x = self.transpose(x)
        if params_evolution:
            x, params = self.normalization(x, params, params_evolution)
        else:
            x = self.normalization(x, params)
        x = self.activation(x)
        return x if not params_evolution else (x, params)     


class EncoderDecorder(nn.Module):

    def __init__(self, config, normalization_type, nonadaptive_normalization_type, style_vector_channels):
        super().__init__()

        '''
        default_normalization_type - used where is no adaptive normalization
        '''

        self.style_vector_channels = style_vector_channels
        self.nonadaptive_normalization_type = nonadaptive_normalization_type 
        self.normalization_type = normalization_type
        self.use_skip_connection = config.use_skip_connection if hasattr(config, 'use_skip_connection') else True
        if self.use_skip_connection:
            self.skip_block_adain = config.skip_block_adain
            skip_block_type = config.skip_block_type

        adain_dict = {}

        downsampling_config = config.downsampling
        bottleneck_config = config.bottleneck
        upsampling_config = config.upsampling

        self.downsampling_dict = nn.ModuleDict()
        self.upsampling_dict = nn.ModuleDict()
        self.bottleneck_dict = nn.ModuleDict()

        downsampling_blocks_numerator = {'Res3DBlock':0, 'Pool3DBlock':0, 'Upsample3DBlock':0}
        upsampling_blocks_numerator = {'Res3DBlock':0, 'Pool3DBlock':0, 'Upsample3DBlock':0}
        bottleneck_blocks_numerator = {'Res3DBlock':0, 'Pool3DBlock':0, 'Upsample3DBlock':0}

        # assert len(upsampling_config) == len(downsampling_config)

        for i, module_config in enumerate(downsampling_config):
            
            self._append_module(self.downsampling_dict, module_config, downsampling_blocks_numerator)
            module_type = module_config['module_type']
            module_number = str(downsampling_blocks_numerator[module_type])

            # add skip-connection
            global SPADE_PARAMS_EVOLUTION
            if module_type == 'Res3DBlock' and self.use_skip_connection:
                in_planes = module_config['params'][-1]
                out_planes = upsampling_config[-i-1]['params'][0]
                skip_normalization_type = self.normalization_type if self.skip_block_adain else \
                                          self.nonadaptive_normalization_type 

                spade_params_evol = SPADE_PARAMS_EVOLUTION
                SPADE_PARAMS_EVOLUTION = False
                skip_con_block = getattr(sys.modules[__name__], skip_block_type)(in_planes, 
                                                                                 out_planes, 
                                                                                 normalization_type=skip_normalization_type,
                                                                                 style_vector_channels=self.style_vector_channels)
                
                name = ''.join(('skip', '_', module_type, '_', module_number))
                self.downsampling_dict[name] = skip_con_block
                SPADE_PARAMS_EVOLUTION = spade_params_evol
                
        for i, module_config in enumerate(bottleneck_config):  
            self._append_module(self.bottleneck_dict, module_config, bottleneck_blocks_numerator)
            
        for i, module_config in enumerate(upsampling_config):

            if i == len(upsampling_config) - 1:
                spade_params_evol = SPADE_PARAMS_EVOLUTION
                SPADE_PARAMS_EVOLUTION = False

            self._append_module(self.upsampling_dict, module_config, upsampling_blocks_numerator)

            if i == len(upsampling_config) - 1:
                SPADE_PARAMS_EVOLUTION=  spade_params_evol

    def _append_module(self, modules_dict, module_config, blocks_numerator):
            
            module_type = module_config['module_type']
            module_adain = module_config['adain']
            module_number = str(blocks_numerator[module_type])
            blocks_numerator[module_type] += 1
            constructor = getattr(sys.modules[__name__], module_config['module_type'])

            module_normalization = self.normalization_type if module_adain else \
                                     self.nonadaptive_normalization_type 
            
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
                model_params['normalization_type'] = module_normalization
                model_params['style_vector_channels'] = self.style_vector_channels

            module = constructor(**model_params)
            name = ''.join((module_type, '_', module_number))
            modules_dict[name] = module

    def forward(self, x, params=None, params_evolution=False):

        skip_connections = []

        for name, module in self.downsampling_dict.items():
            if name.split('_')[0] == 'skip' and self.use_skip_connection:
                skip_connections.append(module(x, params=params))
            else:
                if params_evolution:
                    x, params = module(x, params=params, params_evolution=params_evolution)
                else:
                    x = module(x, params=params)
                        

        for name, module in self.bottleneck_dict.items():
            if params_evolution:
                x, params = module(x, params=params, params_evolution=params_evolution)
            else:
                x = module(x, params=params)
            
        skip_number = -1
        n_ups_blocks = len(self.upsampling_dict)
        for i, (name, module )in enumerate(self.upsampling_dict.items()):
            if name.split('_')[0] == 'Res3DBlock' and self.use_skip_connection:
                skip_connection = skip_connections[skip_number]
                if x.shape[-3:] != skip_connection.shape[-3:]:
                    skip_connection = F.interpolate(skip_connection,
                                                    size=(x.shape[-3:]), 
                                                    mode='trilinear')
                x = x + skip_connection
                skip_number -= 1
            if params_evolution and i < n_ups_blocks-1:
                x, params = module(x, params=params, params_evolution=params_evolution)
            else:
                x = module(x, params=params)

        return x                  


class V2VModel(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels, 
                 v2v_normalization_type, 
                 config, 
                 params_evolution=False,
                 style_forward=False,
                 style_vector_dim=None, 
                 temporal_condition_type=None,
                 use_compound_norm = True,
                 back_layer_output_channels = 64):

        super().__init__()

        self.params_evolution = params_evolution
        self.style_forward = style_forward

        if self.params_evolution:
            global SPADE_PARAMS_EVOLUTION
            SPADE_PARAMS_EVOLUTION = True
            print ('PARAMS EVOLUTION:', SPADE_PARAMS_EVOLUTION)

        if self.style_forward:
            global STYLE_FORWARD
            STYLE_FORWARD = True 
            print ('STYLE_FORWARD:', STYLE_FORWARD)    


        self.style_vector_channels = style_vector_dim
        self.use_compound_norm = use_compound_norm
        self.temporal_condition_type = temporal_condition_type

        if self.temporal_condition_type is None:
            self.normalization_type = v2v_normalization_type
        else:
            self.normalization_type = [v2v_normalization_type, temporal_condition_type] if \
                                        self.use_compound_norm else temporal_condition_type
                                        
        self.nonadaptive_normalization_type = v2v_normalization_type                     

        encoder_input_channels = config.downsampling[0]['params'][0]
        encoder_output_channels = config.upsampling[-1]['params'][1]

        self.front_layer = Basic3DBlock(input_channels, 
                                        encoder_input_channels, 
                                        3, 
                                        self.nonadaptive_normalization_type,
                                        style_vector_channels=self.style_vector_channels)

        self.encoder_decoder = EncoderDecorder(config,
                                               normalization_type=self.normalization_type,
                                               nonadaptive_normalization_type =self.nonadaptive_normalization_type,  
                                               style_vector_channels=self.style_vector_channels)

        self.back_layer = Basic3DBlock(encoder_output_channels, 
                                       back_layer_output_channels, 
                                       1, 
                                       self.nonadaptive_normalization_type,
                                       style_vector_channels=self.style_vector_channels)

        self.output_layer = nn.Conv3d(back_layer_output_channels, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x, params=None):

        if self.style_forward:
            global STYLE_FORWARD
            STYLE_FORWARD = True
            batch_size = x.shape[0]
            x = torch.cat([x,params],0)
            params=None

        x = self.front_layer(x)
        x = self.encoder_decoder(x, 
                                params, 
                                params_evolution=self.params_evolution)    
        x = self.back_layer(x)
        x = self.output_layer(x)

        if self.style_forward:
            params = x[batch_size:]
            x = x[:batch_size]

        return (x, params) if self.style_forward else x

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


class Conv1Plus2D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, time_padding, normalization_type='batch_norm'):
        super().__init__()
        t, h, w = kernel_size
        self.block = nn.Sequential(        
            nn.ConvTranspose3d(in_planes, 
                    out_planes, 
                    kernel_size=(1, h, w), 
                    stride=(1, 1, 1), 
                    bias=False),
            get_normalization(normalization_type, out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, 
                    out_planes, 
                    kernel_size=(t, 1, 1), 
                    stride=(1, 1, 1), 
                    padding=(time_padding,0,0),
                    bias=False),
            get_normalization(normalization_type, out_planes),
            nn.ReLU(True)
            )

    def forward(self, x):
        return self.block(x)    



class R2D(nn.Module):

    def __init__(self, 
                device, 
                style_vector_dim, 
                normalization_type, 
                n_r2d_layers, 
                output_volume_dim, 
                time=None,
                upscale_heatmap = False,
                n_upscale_layers = 3,
                output_heatmap_shape=[96,96],
                use_time_avg_pooling = False,
                change_stride_layers = None,
                load_weights=True,
                ME_SHAPE = [5,28,28]
                ):

        super().__init__()

        assert n_r2d_layers >= 1
        assert output_volume_dim in [1,2,3]
        from torchvision.models.video.resnet import VideoResNet,\
                                                     BasicBlock, \
                                                     R2Plus1dStem, \
                                                     Conv2Plus1D

        self.use_time_avg_pooling = use_time_avg_pooling
        self.time = time                            
        self.output_volume_dim = output_volume_dim
        self.n_r2d_layers = n_r2d_layers
        self.style_vector_dim = style_vector_dim
        self.load_weights = load_weights
        self.normalization_type = normalization_type
        self.output_features = {2:128,
                                3:256,
                                4:512}[n_r2d_layers]

        if upscale_heatmap:
            # we get upscaled hetmap as output
            assert self.output_volume_dim == 2 
            layers = []
            if not self.use_time_avg_pooling:
                assert self.output_volume_dim == 2
                out_shape = np.array([*output_heatmap_shape])
                r2d_shape = np.array(ME_SHAPE[-2:])
                diff_shape = (out_shape - r2d_shape) 
                assert (diff_shape >= 0).all()
                spatial_kernel_size = 1 + (diff_shape//n_upscale_layers)
                spatial_kernel_size_residual = 1 + diff_shape%n_upscale_layers
                hidden_dim = self.output_features

                # time dim
                me_time = ME_SHAPE[0]
                time_kernel = me_time//n_upscale_layers if n_upscale_layers < me_time else 2

                for i in range(n_upscale_layers):
                    in_planes = self.output_features if i == 0 else hidden_dim
                    out_planes = hidden_dim

                    # time dim
                    current_time_dim = me_time + i*(-time_kernel + 1) # suppose stride = 1 
                    if i == n_upscale_layers - 1:
                        padding = 0
                        current_time_kernel = current_time_dim
                    else:
                        padding = time_kernel//2 if current_time_dim < time_kernel else 0     
                        current_time_kernel = time_kernel

                    kernel_size = [current_time_kernel] + spatial_kernel_size.tolist()
                    layers.append(Conv1Plus2D(in_planes, 
                                              out_planes, 
                                              kernel_size,
                                              normalization_type = normalization_type,
                                              time_padding=padding))
                    
                # last block 
                kernel_size_residual = [1] + spatial_kernel_size_residual.tolist()
                layers.append(nn.ConvTranspose3d(hidden_dim,
                                                 self.style_vector_dim,
                                                 kernel_size=kernel_size_residual)) 
            # use time-average pooling
            else:
                layers.append(nn.ConvTranspose3d(self.output_features,
                                                 self.style_vector_dim,
                                                 kernel_size=[1,2,2],
                                                 stride=[1,1,1])) #[1,2,2]
                layers.append(nn.AdaptiveAvgPool3d([1, *output_heatmap_shape]))

            self.final_layer = nn.Sequential(*layers)                    

        else:
            if self.output_volume_dim == 2:
                final_layers = []
                x_kernel_size = 1
                if time is not None:
                    if change_stride_layers is None:
                        x_kernel_size = {2:{3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:5},
                                         3:{3:1, 4:1, 5:2, 6:2, 7:2, 8:2, 9:3}}[n_r2d_layers][time]
                    else:
                        x_kernel_size = time                  
                final_layers.append(nn.Conv3d(self.output_features, 
                                                self.style_vector_dim, 
                                                kernel_size=(x_kernel_size, 1,1)))
                if self.use_time_avg_pooling:
                    final_layers.append(nn.AdaptiveAvgPool3d(output_size=(1, *output_heatmap_shape)))

                self.final_layer = nn.Sequential(*final_layers)            

            elif self.output_volume_dim == 1:
                self.final_layer = nn.Sequential(nn.Conv3d(self.output_features, 
                                                self.style_vector_dim, 
                                                kernel_size=(1,1,1)),
                                                nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
                                                )
            else:
                # return as-is, just change channels number
                self.final_layer = nn.Sequential(nn.Conv3d(self.output_features, 
                                                            self.style_vector_dim, 
                                                            kernel_size=(1,1,1)))

        model = VideoResNet(block=BasicBlock,
                            conv_makers=[Conv2Plus1D] * 4,
                            layers=[3, 4, 6, 3],
                            stem=R2Plus1dStem)

        model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288) # 128
        model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576) # 256
        model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152) # 512

        self.motion_extractor = nn.Sequential(OrderedDict([
                                  ('stem', model.stem), 
                                  *[(f'layer{i}', model._modules[f'layer{i}']) for i in range(1,n_r2d_layers+1)]
                                  ]))

        if self.load_weights:
            weights_path = './data/r2plus1d_34_clip8_ig65m_from_scratch-9bae36ae.pth'
            print (f'R2D inited by {weights_path}')
            weights_dict = torch.load(weights_path) # , map_location=device
            model_dict = self.motion_extractor.state_dict()
            new_pretrained_state_dict = {}

            for k, v in weights_dict.items():
                if k in model_dict:
                    new_pretrained_state_dict[k] = weights_dict[k]

            self.motion_extractor.load_state_dict(new_pretrained_state_dict)

            # We need exact Caffe2 momentum for BatchNorm scaling
            if normalization_type == 'batch_norm':
                for m in self.motion_extractor.modules():
                    if isinstance(m, nn.BatchNorm3d):
                        m.eps = 1e-3
                        m.momentum = 0.9
            elif normalization_type == 'group_norm':
                convert_bn_to_gn(self.motion_extractor)            

        if change_stride_layers is not None:
            for i in change_stride_layers:
                change_stride(self.motion_extractor._modules[f'layer{i}'])

    def forward(self, x, return_me_vector=False):

        # x: (batch,3,time,112,112)
        # output: torch.Size([batch, 256, f(time), 14, 14])
        
        x = self.motion_extractor(x)
        if return_me_vector:
            x_me = x
        if hasattr(self, 'final_layer'):
            x = self.final_layer(x)
        if self.output_volume_dim == 2:
            x = x.squeeze(-3)
        elif self.output_volume_dim == 1:
            x = x.squeeze(-1).squeeze(-1).squeeze(-1)    

        return (x, x_me) if return_me_vector else x  