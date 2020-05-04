# Reference: https://github.com/dragonbook/V2V-PoseNet-pytorch
from IPython.core.debugger import set_trace
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import inspect
import torch
import sys 
import numpy as np

ACTIVATION_TYPE = 'LeakyReLU'
MODULES_REQUIRES_NORMALIZAION = ['Res3DBlock', 'Upsample3DBlock', 'Basic3DBlock']
ORDINARY_NORMALIZATIONS = ['group_norm', 'batch_norm']
ADAPTIVE_NORMALIZATION = ['adain', 'spade']


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

class SPADE(nn.Module):
    def __init__(self, style_vector_channels, features_channels, hidden=128, ks=3): #  hidden=128
        super().__init__()

        pw = ks // 2
        self.shared = nn.Sequential(
            nn.Conv3d(style_vector_channels, hidden, kernel_size=ks, padding=pw),
            get_activation(ACTIVATION_TYPE)()
        )
        self.gamma = nn.Conv3d(hidden, features_channels, kernel_size=ks, padding=pw)
        self.beta = nn.Conv3d(hidden, features_channels, kernel_size=ks, padding=pw)
        self.bn = nn.BatchNorm3d(features_channels, affine=False, track_running_stats=False)

    def forward(self, x, params, return_all=False):

        params = F.interpolate(params, size=x.size()[2:], mode='nearest')
        actv = self.shared(params)
        gamma = self.gamma(actv)
        beta = self.beta(actv)
        out = self.bn(x) * (1 + gamma) + beta
        return (out, (1+gamma), beta) if return_all else out


class AdaIN(nn.Module):
    def __init__(self, style_vector_channels, features_channels):
        super(AdaIN, self).__init__()
        self.affine = nn.Linear(style_vector_channels, 2*features_channels)

    def forward(self, features, params, eps = 1e-4):
        # features: [batch_size, C, D1, D2, D3]
        # params: [batch_size, 2*С]
        adain_params = self.affine(params)
        size = features.size()
        batch_size, C = features.shape[:2]
        unbiased = features.view(batch_size, C, -1).shape[-1] == 1

        adain_mean = adain_params[:,:C].view(batch_size, C,1,1,1)
        adain_std = adain_params[:,C:].view(batch_size, C,1,1,1)
    
        features_mean = features.view(batch_size, C, -1).mean(-1).view(batch_size, C,1,1,1)
        features_std = features.view(batch_size, C, -1).std(-1, unbiased=unbiased).view(batch_size, C,1,1,1)

        return ((features - features_mean) / (features_std + eps)) * adain_std + adain_mean


class CompoundNorm(nn.Module):
    def __init__(self, normalization_types, out_planes, n_groups, style_vector_channels):
        super().__init__()
        norm_type, adaptive_norm_type = normalization_types
        assert norm_type in ORDINARY_NORMALIZATIONS and adaptive_norm_type in ADAPTIVE_NORMALIZATION
        self.norm = get_normalization(norm_type, out_planes, n_groups, style_vector_channels)
        self.adaptive_norm = get_normalization(adaptive_norm_type, out_planes, n_groups, style_vector_channels)
    def forward(self, x, params):
        x = self.norm(x)
        x = self.adaptive_norm(x, params)
        return x        

class GroupNorm(nn.Module):
    def __init__(self, n_groups, features_channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(n_groups, features_channels)
    def forward(self, x, params=None):
        return self.group_norm(x)

class BatchNorm3d(nn.Module):
    def __init__(self, features_channels):
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(features_channels, affine=False)
    def forward(self, x, params=None):
        return self.batch_norm(x)        
        

def get_activation(activation_type):
    return {'LeakyReLU':nn.LeakyReLU,'ReLU':nn.ReLU}[activation_type]

def get_normalization(normalization_type, features_channels, n_groups=32, style_vector_channels=None):

    if type(normalization_type) is list:
        return CompoundNorm(normalization_type, features_channels, n_groups, style_vector_channels)
    else:    
        if normalization_type == 'adain':
            return AdaIN(style_vector_channels, features_channels)
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

    def forward(self, x, params=None):
        x = self.conv(x)
        x = self.normalization(x, params)
        x = self.activation(x)
        return x


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, normalization_type, kernel_size=3, n_groups=32, padding=1, style_vector_channels=None):
        super(Res3DBlock, self).__init__()

        self.normalization_type = normalization_type
        
        self.res_norm1 = get_normalization(normalization_type, out_planes, n_groups,style_vector_channels)
        self.res_norm2 = get_normalization(normalization_type, out_planes, n_groups,style_vector_channels)

        self.res_conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding)
        self.res_conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding)
        
        self.activation = get_activation(ACTIVATION_TYPE)(True)

        self.use_skip_con = (in_planes != out_planes)
        if self.use_skip_con:
            self.skip_con_conv = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
            self.skip_con_norm = get_normalization(normalization_type, out_planes, n_groups, style_vector_channels)    

    def forward(self, x, params=None):
        if self.use_skip_con:
            skip = self.skip_con_conv(x)
            skip = self.skip_con_norm(skip, params)
        else:
            skip = x

        x = self.res_conv1(x)
        x = self.res_norm1(x, params) 
        x = self.activation(x)
        x = self.res_conv2(x)
        x = self.res_norm2(x, params) 

        return F.relu(x + skip, True)        


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x, params=None):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, normalization_type, n_groups=32, style_vector_channels=None):
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

    def forward(self, x, params=None):
        x = self.transpose(x)
        x = self.normalization(x,params)
        x = self.activation(x)
        return x       



class EncoderDecorder(nn.Module):

    def __init__(self, config, normalization_type, temporal_condition_type, style_vector_channels):
        super().__init__()

        '''
        default_normalization_type - used where is no adaptive normalization
        '''

        self.style_vector_channels = style_vector_channels
        self.normalization_type = [normalization_type, temporal_condition_type]
        self.nonadaptive_normalization_type = normalization_type
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
            if module_type == 'Res3DBlock' and self.use_skip_connection:
                in_planes = module_config['params'][-1]
                out_planes = upsampling_config[-i-1]['params'][0]
                skip_normalization_type = self.normalization_type if self.skip_block_adain else \
                                          self.nonadaptive_normalization_type 

                skip_con_block = getattr(sys.modules[__name__], skip_block_type)(in_planes, 
                                                                                 out_planes, 
                                                                                 normalization_type=skip_normalization_type,
                                                                                 style_vector_channels=self.style_vector_channels)

                name = ''.join(('skip', '_', module_type, '_', module_number))
                self.downsampling_dict[name] = skip_con_block
                
        for i, module_config in enumerate(bottleneck_config):  
            self._append_module(self.bottleneck_dict, module_config, bottleneck_blocks_numerator)
            
        for i, module_config in enumerate(upsampling_config):  
            self._append_module(self.upsampling_dict, module_config, upsampling_blocks_numerator)

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

    def forward(self, x, params=None, return_intermediate=False):

        skip_connections = []
        intermediate_dict = []

        for name, module in self.downsampling_dict.items():
            if name.split('_')[0] == 'skip' and self.use_skip_connection:
                skip_connections.append(module(x, params=params))
            else:
                x = module(x, params=params)
                if return_intermediate:
                    intermediate_dict['downsampling_' + name] = x

        for name, module in self.bottleneck_dict.items():
            x = module(x, params=params)
            if return_intermediate:
                intermediate_dict['bottleneck_' + name] = x
            
        skip_number = -1    
        for name, module in self.upsampling_dict.items():
            if name.split('_')[0] == 'Res3DBlock' and self.use_skip_connection:
                x = x + skip_connections[skip_number]
                skip_number -= 1
            x = module(x, params=params)
            if return_intermediate:
                intermediate_dict['upsampling_' + name] = x      

        if return_intermediate:
            return x, intermediate_dict
        else:    
            return x                  


class V2VModel(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels, 
                 v2v_normalization_type, 
                 config, 
                 style_vector_dim=None, 
                 temporal_condition_type=None):

        super().__init__()

        # self.style_vector = 

        self.style_vector_channels = style_vector_dim #config.style_vector_dim if hasattr(config, 'style_vector_dim') else None
        
        self.normalization_type = v2v_normalization_type #config.v2v_normalization_type
        self.temporal_condition_type = temporal_condition_type #config.temporal_condition_type if hasattr(config, 'temporal_condition_type') else None

        encoder_input_channels = config.downsampling[0]['params'][0]
        encoder_output_channels = config.upsampling[-1]['params'][1]

        self.front_layer = Basic3DBlock(input_channels, 
                                        encoder_input_channels, 
                                        3, 
                                        self.normalization_type,
                                        style_vector_channels=self.style_vector_channels)

        self.encoder_decoder = EncoderDecorder(config,
                                               self.normalization_type, 
                                               self.temporal_condition_type,
                                               style_vector_channels=self.style_vector_channels)

        self.back_layer = Basic3DBlock(encoder_output_channels, 
                                       32, 
                                       1, 
                                       self.normalization_type,
                                       style_vector_channels=self.style_vector_channels)

        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x, params=None, device=None, return_intermediate=False):
        # intermediate_dict = []
        x = self.front_layer(x)
        # if return_intermediate:
        #     intermediate_dict['front_layer'] = x

        # if return_intermediate:
        #     x, intermediate = self.encoder_decoder(x, params, return_intermediate=return_intermediate)
        #     intermediate_dict.update(intermediate)
        # else:
        x = self.encoder_decoder(x, params, return_intermediate=return_intermediate)    

        x = self.back_layer(x)
        # if return_intermediate:
        #     intermediate_dict['back_layer'] = x

        x = self.output_layer(x)

        # if return_intermediate:
        #     return x, intermediate_dict
        # else:
        return x

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



class C3D(nn.Module):
    def __init__(self, device, output_channels, poolings=5, n_layers=5, weights_path='./data/c3d.pickle'):
        super(C3D, self).__init__()

        self.poolings = poolings
        self.n_layers = n_layers
        self.output_channels = output_channels

        assert self.n_layers >= 1

        if self.n_layers >= 1:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            if self.poolings >= 5:
                self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        if self.n_layers >= 2:    
            self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            if self.poolings >= 4:
                self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        if self.n_layers >= 3:    
            self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            if self.poolings >= 3:
                self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        if self.n_layers >= 4:    
            self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            if self.poolings >= 2:
                self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        if self.n_layers >= 5:    
            self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            if self.poolings >= 1:
                self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.relu = get_activation(ACTIVATION_TYPE)()

        last_layer_channels = {1:64, 2:128, 3:256, 4:512, 5:512}[self.n_layers]

        self.final_layer = nn.Conv3d(last_layer_channels, self.output_channels)

        weights_dict = torch.load(weights_path, map_location=device)
        model_dict = self.state_dict()
        new_pretrained_state_dict = {}
        for k, v in weights_dict.items():
            if k in model_dict:
                new_pretrained_state_dict[k] = weights_dict[k]
            else:    
                print (k, 'hasnt been loaded in C3D')
        self.load_state_dict(new_pretrained_state_dict)

    def forward(self, x):

        if self.n_layers >= 1:
            h = self.relu(self.conv1(x))
            if self.poolings >= 5:
                h = self.pool1(h)

        if self.n_layers >= 2:         
            h = self.relu(self.conv2(h))
            if self.poolings >= 4:
                h = self.pool2(h)

        if self.n_layers >= 3:         
            h = self.relu(self.conv3a(h))
            h = self.relu(self.conv3b(h))
            if self.poolings >= 3:
                h = self.pool3(h)

        if self.n_layers >= 4:         
            h = self.relu(self.conv4a(h))
            h = self.relu(self.conv4b(h))
            if self.poolings >= 2:    
                h = self.pool4(h)

        if self.n_layers >= 5:         
            h = self.relu(self.conv5a(h))
            h = self.relu(self.conv5b(h))
            if self.poolings >= 1:
                h = self.pool5(h)

        return h    


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
                time,
                upscale_heatmap = False,
                n_upscale_layers = 3,
                upscale_heatmap_shape=[96,96],
                use_time_avg_pooling = False,
                change_stride_layers = None,
                load_weights=False
                ):

        super().__init__()

        assert n_r2d_layers >= 1
        assert output_volume_dim in [1,2,3]
        from torchvision.models.video.resnet import VideoResNet,\
                                                     BasicBlock, \
                                                     R2Plus1dStem, \
                                                     Conv2Plus1D

        self.use_time_avg_pooling = use_time_avg_pooling                                             
        self.output_volume_dim = output_volume_dim
        self.n_r2d_layers = n_r2d_layers
        self.style_vector_dim = style_vector_dim
        self.load_weights = load_weights
        self.normalization_type = normalization_type
        self.output_features = {2:128,
                                3:256,
                                4:512}[n_r2d_layers]

        if upscale_heatmap: 
            layers = []
            if not self.use_time_avg_pooling:
                ME_SHAPE = [5,28,28]
                assert self.output_volume_dim == 2 and n_r2d_layers == 3
                out_shape = np.array([*upscale_heatmap_shape])
                r2d_shape = np.array(ME_SHAPE[-2:])
                me_time = ME_SHAPE[0]
                diff_shape = (out_shape - r2d_shape) 
                assert (diff_shape >= 0).all()
                spatial_kernel_size = 1 + (diff_shape//n_upscale_layers)
                spatial_kernel_size_residual = 1 + diff_shape%n_upscale_layers
                hidden_dim = self.output_features // 2

                for i in range(n_upscale_layers):
                    in_planes = self.output_features if i == 0 else hidden_dim
                    out_planes = hidden_dim
                    padding = 1 if i < (n_upscale_layers-1) else 0
                    time_kernel = [me_time] if not padding else [1] 
                    kernel_size = time_kernel + spatial_kernel_size.tolist()
                    layers.append(Conv1Plus2D(in_planes, 
                                              out_planes, 
                                              kernel_size,
                                              time_padding=padding))
                    
                # last block 
                kernel_size_residual = [1] + spatial_kernel_size_residual.tolist()
                layers.append(nn.ConvTranspose3d(hidden_dim,
                                             self.style_vector_dim,
                                             kernel_size=kernel_size_residual)) 
            else:
                layers.append(nn.ConvTranspose3d(self.output_features,
                                                 self.style_vector_dim,
                                                 kernel_size=[1,2,2],
                                                 stride=[1,2,2]))
                layers.append(nn.AdaptiveAvgPool3d([1, *upscale_heatmap_shape]))


            self.final_layer = nn.Sequential(*layers)                    

        else:
            if self.output_volume_dim == 2:
                if self.use_time_avg_pooling:
                    self.final_layer = nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(1, 28, 28)),
                                                    nn.Conv3d(self.output_features, 
                                                    self.style_vector_dim, 
                                                    kernel_size=(1,1,1))
                                                    )
                else:
                    x_kernel_size = {2:{3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:5},
                                     3:{3:1, 4:1, 5:2, 6:2, 7:2, 8:2, 9:3}}[n_r2d_layers][time]

                    self.final_layer = nn.Sequential(nn.Conv3d(self.output_features, 
                                            self.style_vector_dim, 
                                            kernel_size=(x_kernel_size, 1,1)))    

            elif self.output_volume_dim == 1:
                self.final_layer = nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
                                                nn.Conv3d(self.output_features, 
                                                self.style_vector_dim, 
                                                kernel_size=(1,1,1))
                                                )
            else:
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