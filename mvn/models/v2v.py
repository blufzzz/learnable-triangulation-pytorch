# Reference: https://github.com/dragonbook/V2V-PoseNet-pytorch
from IPython.core.debugger import set_trace
import torch.nn as nn
import torch.nn.functional as F
import torch
 

MODULES_REQUIRES_NORMALIZAION = ['Res3DBlock', 'Upsample3DBlock', 'Basic3DBlock']
STYLE_VECTOR_CHANNELS = None

class SPADE(nn.Module):
    def __init__(self, style_vector_channels, features_channels, hidden=128, ks=3):
        super().__init__()

        pw = ks // 2
        self.shared = nn.Sequential(
            nn.Conv2d(style_vector_channels, hidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.gamma = nn.Conv2d(hidden, features_channels, kernel_size=ks, padding=pw)
        self.beta = nn.Conv2d(hidden, features_channels, kernel_size=ks, padding=pw)

    def forward(self, x, params):

        params = F.interpolate(params, size=x.size()[2:], mode='nearest')
        actv = self.shared(params)
        gamma = self.gamma(actv)
        beta = self.beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class AdaIN(nn.Module):
    def __init__(self, style_vector_channels, features_channels):
        super(AdaIN, self).__init__()
        self.affine = nn.Linear(style_vector_channels, 2*features_channels)

    def forward(self, features, params, eps = 1e-4 ,debug=False):
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
        self.dict = nn.ModuleDict()
        self.adaptive = True
        for i, norm_type in enumerate(normalization_types):
            self.dict[f'{norm_type}-{i}'] = get_normalization(normalization_type, out_planes, n_groups=32)
    def forward(self, x, params):
        for name, block in self.dict.items():
            x = self.block(x, params) if name.split('-')[0] in ['spade', 'adain'] else self.block(x)
        return x        


def get_normalization(normalization_type, features_channels, n_groups=32):

    style_vector_channels = STYLE_VECTOR_CHANNELS

    if type(normalization_type) is list:
        return CompoundNorm(normalization_type, features_channels, n_groups)
    else:    
        if normalization_type == 'adain':
            return AdaIN(style_vector_channels, features_channels)
        elif normalization_type ==  'batch_norm':
            return nn.BatchNorm3d(features_channels)
        elif normalization_type ==  'spade':
            return SPADE(style_vector_channels, features_channels)    
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

        self.activation = nn.ReLU(True)

    def forward(self, x, params=None):
        x = self.conv(x)
        x = self.normalization(x, params) if self.normalization_type in \
             ['adain', 'ada-group_norm', 'group-ada_norm'] else self.normalization(x)
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

    def forward(self, x, params=[None,None,None], debug=False):
        if self.use_skip_con:
            skip = self.skip_con_conv(x)
            skip = self.skip_con_norm(skip, params[2]) if self.normalization_type in \
                     ['adain', 'ada-group_norm', 'group-ada_norm'] else self.skip_con_norm(skip)
        else:
            skip = x

        x = self.res_conv1(x)
        x = self.res_norm1(x, params[0], debug=debug) if self.normalization_type in \
                ['adain', 'ada-group_norm', 'group-ada_norm'] else self.res_norm1(x)
        x = self.activation(x)
        x = self.res_conv2(x)
        x = self.res_norm2(x, params[1]) if self.normalization_type in \
            ['adain', 'ada-group_norm', 'group-ada_norm'] else self.res_norm2(x)

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
        x = self.normalization(x,params) if self.normalization_type in ['adain', 'ada-group_norm', 'group-ada_norm'] else self.normalization(x)
        x = self.activation(x)
        return x       


class EncoderDecorder(nn.Module):
    def __init__(self, normalization_type, volume_size=32):
        super().__init__()
        self.normalization_type = normalization_type
        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64, normalization_type)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128, normalization_type)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res3 = Res3DBlock(128, 128, normalization_type)
        self.volume_size = volume_size
        
        # 1
        if volume_size > 16:
            self.encoder_pool4 = Pool3DBlock(2)
        
        self.encoder_res4 = Res3DBlock(128, 128, normalization_type)
        
        # 2
        if volume_size > 32:
            self.encoder_pool5 = Pool3DBlock(2) 
        
        self.encoder_res5 = Res3DBlock(128, 128, normalization_type)

        self.mid_res = Res3DBlock(128, 128, normalization_type)

        self.decoder_res5 = Res3DBlock(128, 128, normalization_type)
        
        # 2
        if volume_size > 32:
            self.decoder_upsample5 = Upsample3DBlock(128, 128, 2, 2, normalization_type)
        
        self.decoder_res4 = Res3DBlock(128, 128, normalization_type)
        
        # 1
        if volume_size > 16:
            self.decoder_upsample4 = Upsample3DBlock(128, 128, 2, 2, normalization_type)
        
        self.decoder_res3 = Res3DBlock(128, 128, normalization_type)
        self.decoder_upsample3 = Upsample3DBlock(128, 128, 2, 2, normalization_type)
        self.decoder_res2 = Res3DBlock(128, 128, normalization_type)
        self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2, normalization_type)
        self.decoder_res1 = Res3DBlock(64, 64, normalization_type)
        self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2, normalization_type)

        self.skip_res1 = Res3DBlock(32, 32, normalization_type)
        self.skip_res2 = Res3DBlock(64, 64, normalization_type)
        self.skip_res3 = Res3DBlock(128, 128, normalization_type)
        self.skip_res4 = Res3DBlock(128, 128, normalization_type)
        self.skip_res5 = Res3DBlock(128, 128, normalization_type)

    def forward(self, x, params):
        skip_x1 = self.skip_res1(x, params[:2])
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x, params[2:5])
        skip_x2 = self.skip_res2(x, params[5:7])
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x, params[7:10])
        skip_x3 = self.skip_res3(x, params[10:12])
        x = self.encoder_pool3(x)
        x = self.encoder_res3(x, params[12:14])
        skip_x4 = self.skip_res4(x, params[14:16])
        x = self.encoder_pool4(x) if self.volume_size > 16 else x

        # torch.Size([1, 128, 1, 1, 1]), no NAN
        x = self.encoder_res4(x, params[16:18], debug=True) # NAN
        
        # torch.Size([1, 128, 1, 1, 1]), NAN
        skip_x5 = self.skip_res5(x, params[18:20]) # NAN
        x = self.encoder_pool5(x) if self.volume_size > 32 else x
        x = self.encoder_res5(x, params[20:22]) 

        x = self.mid_res(x, params[22:24])

        x = self.decoder_res5(x, params[24:26])
        x = self.decoder_upsample5(x, params[26]) if self.volume_size > 32 else x
        x = x + skip_x5
        x = self.decoder_res4(x, params[27:29])
        x = self.decoder_upsample4(x, params[29]) if self.volume_size > 16 else x
        x = x + skip_x4
        x = self.decoder_res3(x, params[30:32])
        x = self.decoder_upsample3(x, params[32])
        x = x + skip_x3
        x = self.decoder_res2(x, params[33:35])
        x = self.decoder_upsample2(x, params[35])
        x = x + skip_x2
        x = self.decoder_res1(x, params[36:38])
        x = self.decoder_upsample1(x, params[38])
        x = x + skip_x1

        return x


class V2VModel(nn.Module):
    def __init__(self, input_channels, output_channels, volume_size, normalization_type='batch_norm'):
            super().__init__()
            self.normalization_type=normalization_type
            self.front_layer1 = Basic3DBlock(input_channels, 32, 7, normalization_type)
            self.front_layer2 = Res3DBlock(32, 32, normalization_type)
            self.front_layer3 = Res3DBlock(32, 32, normalization_type)
            self.front_layer4 = Res3DBlock(32, 32, normalization_type)
            # [32, 32,32, 32,32, 32,32]

            self.encoder_decoder = EncoderDecorder(normalization_type, volume_size=volume_size)
            # [32,32, 64,64,64, 64,64, 128,128,128, 128,128, 128,128,
            # 128,128, 128,128, 128,128, 128,128, 128,128, 128,128, 128,
            # 128,128, 128, 128,128, 128, 128,128, 64, 64,64, 32]

            self.back_layer1 = Res3DBlock(32, 32, normalization_type)
            self.back_layer2 = Basic3DBlock(32, 32, 1, normalization_type)
            self.back_layer3 = Basic3DBlock(32, 32, 1, normalization_type)
            # [32,32, 32, 32]

            self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

            self._initialize_weights()

    def forward(self, x, params=[None]*50):
        x = self.front_layer1(x, params[0])
        x = self.front_layer2(x, params[1:3])
        x = self.front_layer3(x, params[3:5])
        x = self.front_layer4(x, params[5:7])

        x = self.encoder_decoder(x, params[7:46]) 

        x = self.back_layer1(x, params[46:48])
        x = self.back_layer2(x, params[48])
        x = self.back_layer3(x, params[49])

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


class EncoderDecorder_conf(nn.Module):

    def __init__(self, config, normalization_type):
        super().__init__()

        '''
        default_normalization_type - used where is no adain normalization
        '''

        self.normalization_type = config.normalization_type
        self.nonadaptive_normalization_type = config.nonadaptive_normalization_type

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
                skip_con_block = getattr(v2v, skip_block_type)(in_planes, 
                                                               out_planes, 
                                                               normalization_type=normalization_type)
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

            module = getattr(v2v, module_config['module_type'])(*model_params)
            name = ''.join((module_type, '_', module_number))
            modules_dict[name] = module

    def forward(self, x, adain_params):

        skip_connections = []

        for name, module in self.downsampling_dict.items():
            name_list = name.split('_')
            if name_list[0] == 'skip':
                skip_connections.append(module(x, params=adain_params))
            else:
                x = module(x, params=adain_params)

        for name, module in self.bottleneck_dict.items():
            x = module(x, params=adain_params)

        skip_number = -1    
        for name, module in self.upsampling_dict.items():
            x = module(x, params=adain_params)
            if name == 'Res3DBlock':
                x = x + skip_connections[skip_number]
                skip_number -= 1  

        return x                  


class V2VModel_conf(nn.Module):
    def __init__(self, input_channels, output_channels, config, normalization_type):
        super().__init__()

        global STYLE_VECTOR_CHANNELS
        STYLE_VECTOR_CHANNELS = config.style_vector_dim
        config = config.v2v_configuration

        self.nonadaptive_normalization_type = config.nonadaptive_normalization_type
        self.normalization_type=normalization_type

        encoder_input_channels = config.downsampling[0]['params'][0]
        encoder_output_channels = config.upsampling[-1]['params'][-1]

        self.front_layer = Basic3DBlock(input_channels, encoder_input_channels, 3, nonadaptive_normalization_type)

        self.encoder_decoder = EncoderDecorder_conf(config, normalization_type)

        self.back_layer = Basic3DBlock(encoder_output_channels, 32, 1, nonadaptive_normalization_type)

        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x, adain_params):
        x = self.front_layer(x)
        x = self.encoder_decoder(x, adain_params) 
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



class EncoderDecorder_v2(nn.Module):
    def __init__(self, normalization_type, n_poolings=4):
        super().__init__()
        assert n_poolings <= 5
        self.normalization_type = normalization_type
        self.encoder_pool1 = Pool3DBlock(2) 
        self.encoder_res1 = Res3DBlock(128, 128, normalization_type)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(128, 128, normalization_type)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res3 = Res3DBlock(128, 128, normalization_type)
        self.encoder_pool4 = Pool3DBlock(2) 
        self.encoder_res4 = Res3DBlock(128, 128, normalization_type)
        # self.encoder_pool5 = Pool3DBlock(2)
        self.encoder_res5 = Res3DBlock(128, 256, normalization_type)

        self.mid_res = Res3DBlock(256, 256, normalization_type) 

        self.decoder_res5 = Res3DBlock(256, 128, normalization_type)
        # self.decoder_upsample5 = Upsample3DBlock(128, 128, 2, 2, normalization_type)

        self.decoder_res4 = Res3DBlock(128, 128, normalization_type)
        self.decoder_upsample4 = Upsample3DBlock(128, 128, 2, 2, normalization_type)

        self.decoder_res3 = Res3DBlock(128, 64, normalization_type)
        self.decoder_upsample3 = Upsample3DBlock(64, 64, 2, 2, normalization_type)

        self.decoder_res2 = Res3DBlock(64, 32, normalization_type)
        self.decoder_upsample2 = Upsample3DBlock(32, 32, 2, 2, normalization_type)

        self.decoder_res1 = Res3DBlock(32, 32, normalization_type)
        self.decoder_upsample1 = Upsample3DBlock(32, 32, 2, 2, normalization_type)

        self.skip_res1 = Res3DBlock(128, 32, normalization_type)
        self.skip_res2 = Res3DBlock(128, 32, normalization_type)
        self.skip_res3 = Res3DBlock(128, 64, normalization_type)
        self.skip_res4 = Res3DBlock(128, 128, normalization_type)
        self.skip_res5 = Res3DBlock(128, 128, normalization_type)

    def forward(self, x, params):
        skip_x1 = self.skip_res1(x, params[:3])
        x = self.encoder_pool1(x)
        #print ('After pool1', x.shape)

        x = self.encoder_res1(x, params[3:5])
        skip_x2 = self.skip_res2(x, params[5:8])
        x = self.encoder_pool2(x)
        #print ('After pool2', x.shape)

        x = self.encoder_res2(x, params[8:10])
        skip_x3 = self.skip_res3(x, params[10:13])
        x = self.encoder_pool3(x)
        #print ('After pool3', x.shape)

        x = self.encoder_res3(x, params[13:15])
        skip_x4 = self.skip_res4(x, params[15:17])
        x = self.encoder_pool4(x)
        #print ('After pool4', x.shape)

        x = self.encoder_res4(x, params[17:19])
        skip_x5 = self.skip_res5(x, params[19:21])
        # x = self.encoder_pool5(x)
        x = self.encoder_res5(x, params[21:24]) 

        x = self.mid_res(x, params[24:26]) #([1, 256, 2, 2, 2]) for vs=32
        #print ('Mid res', x.shape) 

        x = self.decoder_res5(x, params[26:29])
        # x = self.decoder_upsample5(x, params[29])
        x = x + skip_x5
        x = self.decoder_res4(x, params[30:32])
        x = self.decoder_upsample4(x, params[32])
        x = x + skip_x4
        x = self.decoder_res3(x, params[33:36])
        x = self.decoder_upsample3(x, params[36])
        x = x + skip_x3
        x = self.decoder_res2(x, params[37:40])
        x = self.decoder_upsample2(x, params[40])
        x = x + skip_x2
        x = self.decoder_res1(x, params[41:43])
        x = self.decoder_upsample1(x, params[43])
        x = x + skip_x1

        return x



class V2VModel_v2(nn.Module):
    def __init__(self, input_channels, output_channels, volume_size, n_poolings=4, normalization_type='batch_norm'):
            super().__init__()
            self.normalization_type=normalization_type
            self.front_layer1 = Res3DBlock(input_channels, 128, normalization_type)
            self.front_layer2 = Res3DBlock(128, 128, normalization_type)
            # [128,128  128,128]

            self.encoder_decoder = EncoderDecorder_v2(normalization_type, n_poolings)
            # [16,16,16  128,128,128,  32,32,32, 128,128,  64,64,64,  128,128,
            # 128,128,  128,128,  128,128  256,256,256,  256,256,
            # 128,128,128,  128,  128,128,  128,  64,64,64,  64,  32,32,32,  32,  32,32,  32]

            self.back_layer1 = Res3DBlock(32, 32, normalization_type)
            # self.back_layer2 = Res3DBlock(32, 32, normalization_type)
            # [32,32,  32,32]

            self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

            self._initialize_weights()

    def forward(self, x, params=[None]*50):
        x = self.front_layer1(x, params[:2])
        x = self.front_layer2(x, params[2:4])
        x = self.encoder_decoder(x, params[4:48])  # 29
        x = self.back_layer1(x, params[48:50])
        # x = self.back_layer2(x, params[50:52])
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