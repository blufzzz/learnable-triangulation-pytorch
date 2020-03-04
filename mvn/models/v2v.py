# Reference: https://github.com/dragonbook/V2V-PoseNet-pytorch
from IPython.core.debugger import set_trace
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys 

MODULES_REQUIRES_NORMALIZAION = ['Res3DBlock', 'Upsample3DBlock', 'Basic3DBlock']
STYLE_VECTOR_CHANNELS = None
ORDINARY_NORMALIZATIONS = ['group_norm', 'batch_norm']
ADAPTIVE_NORMALIZATION = ['adain', 'spade']

class SPADE(nn.Module):
    def __init__(self, style_vector_channels, features_channels, hidden=128, ks=3):
        super().__init__()

        pw = ks // 2
        self.shared = nn.Sequential(
            nn.Conv3d(style_vector_channels, hidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.gamma = nn.Conv3d(hidden, features_channels, kernel_size=ks, padding=pw)
        self.beta = nn.Conv3d(hidden, features_channels, kernel_size=ks, padding=pw)

    def forward(self, x, params):
        params = F.interpolate(params, size=x.size()[2:], mode='nearest')
        actv = self.shared(params)
        gamma = self.gamma(actv)
        beta = self.beta(actv)
        out = x * (1 + gamma) + beta
        return out


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

        features = ((features - features_mean) / (features_std + eps)) * adain_std + adain_mean
        return features


class CompoundNorm(nn.Module):
    def __init__(self, normalization_types, out_planes, n_groups=None):
        super().__init__()
        norm_type, adaptive_norm_type = normalization_types
        assert norm_type in ORDINARY_NORMALIZATIONS and adaptive_norm_type in ADAPTIVE_NORMALIZATION
        self.norm = get_normalization(norm_type, out_planes, n_groups=32)
        self.adaptive_norm = get_normalization(adaptive_norm_type, out_planes, n_groups=32)
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
        self.batch_norm = nn.BatchNorm3d(features_channels)
    def forward(self, x, params=None):
        return self.batch_norm(x)        
        

def get_normalization(normalization_type, features_channels, n_groups=32):

    style_vector_channels = STYLE_VECTOR_CHANNELS

    if type(normalization_type) is list:
        return CompoundNorm(normalization_type, features_channels, n_groups)
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
    def __init__(self, in_planes, out_planes, kernel_size, normalization_type, n_groups=32):
        super(Basic3DBlock, self).__init__()
        self.normalization_type = normalization_type
        self.conv =  nn.Conv3d(in_planes, out_planes, 
                                kernel_size=kernel_size, 
                                stride=1, 
                                padding=((kernel_size-1)//2))

        self.normalization = get_normalization(normalization_type, out_planes, n_groups)

        self.activation = nn.ReLU(True)

    def forward(self, x, params=None):
        x = self.conv(x)
        x = self.normalization(x, params)
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
        
        self.activation = nn.ReLU(True)

        self.use_skip_con = ( in_planes != out_planes)
        if self.use_skip_con:
            self.skip_con_conv = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
            self.skip_con_norm = get_normalization(normalization_type, out_planes, n_groups)    

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
    def __init__(self, in_planes, out_planes, kernel_size, stride, normalization_type, n_groups=32):
        super().__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.normalization_type = normalization_type
        self.transpose = nn.ConvTranspose3d(in_planes, 
                                            out_planes, 
                                            kernel_size=kernel_size, 
                                            stride=stride, 
                                            padding=0, 
                                            output_padding=0)

        self.normalization = get_normalization(normalization_type, out_planes, n_groups)
        self.activation = nn.ReLU(True)

    def forward(self, x, params=None):
        x = self.transpose(x)
        x = self.normalization(x,params)
        x = self.activation(x)
        return x       



class EncoderDecorder(nn.Module):

    def __init__(self, config, normalization_type, temporal_condition_type):
        super().__init__()

        '''
        default_normalization_type - used where is no adain normalization
        '''

        self.normalization_type = [normalization_type, temporal_condition_type]
        self.nonadaptive_normalization_type = normalization_type
        self.skip_block_adain = config.skip_block_adain

        adain_dict = {}
        skip_block_type = config.skip_block_type

        downsampling_config = config.downsampling
        bottleneck_config = config.bottleneck
        upsampling_config = config.upsampling

        self.downsampling_dict = nn.ModuleDict()
        self.upsampling_dict = nn.ModuleDict()
        self.bottleneck_dict = nn.ModuleDict()

        downsampling_blocks_numerator = {'Res3DBlock':0, 'Pool3DBlock':0, 'Upsample3DBlock':0}
        upsampling_blocks_numerator = {'Res3DBlock':0, 'Pool3DBlock':0, 'Upsample3DBlock':0}
        bottleneck_blocks_numerator = {'Res3DBlock':0, 'Pool3DBlock':0, 'Upsample3DBlock':0}

        assert len(upsampling_config) == len(downsampling_config)

        for i, module_config in enumerate(downsampling_config):
            
            self._append_module(self.downsampling_dict, module_config, downsampling_blocks_numerator)
            module_type = module_config['module_type']
            module_number = str(downsampling_blocks_numerator[module_type])
            if module_type == 'Res3DBlock':
                # add skip-connection
                in_planes = module_config['params'][-1]
                out_planes = upsampling_config[-i-1]['params'][0]
                skip_normalization_type = self.normalization_type if self.skip_block_adain else \
                                            self.nonadaptive_normalization_type 
                skip_con_block = getattr(sys.modules[__name__], skip_block_type)(in_planes, 
                                                               out_planes, 
                                                               normalization_type=skip_normalization_type)
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
            
            module_normalization = self.normalization_type if module_adain else \
                                     self.nonadaptive_normalization_type 
                                     
            model_params = module_config['params'] + [module_normalization] if \
                           module_type in MODULES_REQUIRES_NORMALIZAION else module_config['params']

            module = getattr(sys.modules[__name__], module_config['module_type'])(*model_params)
            name = ''.join((module_type, '_', module_number))
            modules_dict[name] = module

    def forward(self, x, params=None):

        skip_connections = []

        for name, module in self.downsampling_dict.items():
            if name.split('_')[0] == 'skip':
                skip_connections.append(module(x, params=params))
            else:
                x = module(x, params=params)

        for name, module in self.bottleneck_dict.items():
            x = module(x, params=params)
            
        skip_number = -1    
        for name, module in self.upsampling_dict.items():
            if name.split('_')[0] == 'Res3DBlock':
                x = x + skip_connections[skip_number]
                skip_number -= 1
            x = module(x, params=params)      

        return x                  


class V2VModel(nn.Module):
    def __init__(self, input_channels, output_channels, config):
        super().__init__()

        global STYLE_VECTOR_CHANNELS
        STYLE_VECTOR_CHANNELS = config.style_vector_dim if hasattr(config, 'style_vector_dim') else None
        
        self.normalization_type = config.v2v_normalization_type
        self.temporal_condition_type = config.temporal_condition_type if hasattr(config, 'temporal_condition_type') else None

        encoder_input_channels = config.v2v_configuration.downsampling[0]['params'][0]
        encoder_output_channels = config.v2v_configuration.upsampling[-1]['params'][-1]

        self.front_layer = Basic3DBlock(input_channels, encoder_input_channels, 3, self.normalization_type)

        self.encoder_decoder = EncoderDecorder(config.v2v_configuration,
                                               self.normalization_type, 
                                               self.temporal_condition_type)

        self.back_layer = Basic3DBlock(encoder_output_channels, 32, 1, self.normalization_type)

        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x, params=None):
        x = self.front_layer(x)
        x = self.encoder_decoder(x, params) 
        x = self.back_layer(x)
        x = self.output_layer(x)
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



