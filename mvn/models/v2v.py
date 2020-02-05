# Reference: https://github.com/dragonbook/V2V-PoseNet-pytorch
from IPython.core.debugger import set_trace
import torch.nn as nn
import torch.nn.functional as F
import torch
 

class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()
    def forward(self, features, params, eps = 1e-4):
        # features: [batch_size, C, D1, D2, D3]
        # params: [batch_size, 2]
        use_adain_params = params is not None
        size = features.size()
        batch_size, C = features.shape[:2]

        if use_adain_params:
            # set_trace()
            adain_mean = params[:,:C].view(batch_size, C,1,1,1)
            adain_std = params[:,C:].view(batch_size, C,1,1,1)
        else:
            if batch_size % 2 != 0:
                print ('batch should contain concatenated features and style batches')
            batch_size = batch_size // 2
           
            # dont change the order!
            style = features[batch_size:] 
            features = features[:batch_size]

            adain_mean = style.view(batch_size, C, -1).mean(-1).view(batch_size, C,1,1,1)
            adain_std = style.view(batch_size, C, -1).std(-1).view(batch_size, C,1,1,1) 
        
        features_mean = features.view(batch_size, C, -1).mean(-1).view(batch_size, C,1,1,1)
        features_std = features.view(batch_size, C, -1).std(-1).view(batch_size, C,1,1,1)

        if use_adain_params:
            features = ((features - features_mean) / (features_std + eps)) * adain_std + adain_mean
            return features
        else:    
            return torch.cat([((features - features_mean) / (features_std + eps)) * adain_std + adain_mean, style])


class AdaGroupNorm(nn.Module):
    def __init__(self, n_groups, out_planes):
        super(AdaGroupNorm, self).__init__()
        self.adain = AdaIN()
        self.group_norm = nn.GroupNorm(n_groups, out_planes)
    def forward(self, x, params):
        x = self.adain(x, params)
        x = self.group_norm(x)
        return x

class GroupAdaNorm(nn.Module):
    def __init__(self, n_groups, out_planes):
        super(GroupAdaNorm, self).__init__()
        self.adain = AdaIN()
        self.group_norm = nn.GroupNorm(n_groups, out_planes)
    def forward(self, x, params):
        x = self.group_norm(x)
        x = self.adain(x, params)
        return x        


def get_normalization(normalization_type, out_planes, n_groups):
    if normalization_type == 'adain':
        return AdaIN()
    elif normalization_type ==  'batch_norm':
        return nn.BatchNorm3d(out_planes)
    elif normalization_type == 'group_norm':
        return nn.GroupNorm(n_groups, out_planes)
    elif normalization_type == 'ada-group_norm':
        return AdaGroupNorm(n_groups, out_planes)
    elif normalization_type == 'group-ada_norm':
        return GroupAdaNorm(n_groups, out_planes)            
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
        x = self.normalization(x, params) if self.normalization_type in ['adain', 'ada-group_norm', 'group-ada_norm'] else self.normalization(x)
        x = self.activation(x)
        return x


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, normalization_type, n_groups=32):
        super(Res3DBlock, self).__init__()

        self.normalization_type = normalization_type
        
        self.res_norm1 = get_normalization(normalization_type, out_planes, n_groups)
        self.res_norm2 = get_normalization(normalization_type, out_planes, n_groups)

        self.res_conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.res_conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)
        
        self.activation = nn.ReLU(True)

        self.use_skip_con = ( in_planes != out_planes)
        if self.use_skip_con:
            self.skip_con_conv = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
            self.skip_con_norm = get_normalization(normalization_type, out_planes, n_groups)    

    def forward(self, x, params=[None,None,None]):
        if self.use_skip_con:
            skip = self.skip_con_conv(x)
            skip = self.skip_con_norm(skip, params[2]) if self.normalization_type in ['adain', 'ada-group_norm', 'group-ada_norm'] else self.skip_con_norm(skip)
        else:
            skip = x

        x = self.res_conv1(x)
        x = self.res_norm1(x, params[0]) if self.normalization_type in ['adain', 'ada-group_norm', 'group-ada_norm'] else self.res_norm1(x)
        x = self.activation(x)
        x = self.res_conv2(x)
        x = self.res_norm2(x, params[1]) if self.normalization_type in ['adain', 'ada-group_norm', 'group-ada_norm'] else self.res_norm2(x)

        return F.relu(x + skip, True)        



class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)



class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, normalization_type, n_groups=32):
        super(Upsample3DBlock, self).__init__()
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
    def __init__(self, normalization_type):
        super().__init__()
        self.normalization_type = normalization_type
        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64, normalization_type)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128, normalization_type)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res3 = Res3DBlock(128, 128, normalization_type)
        self.encoder_pool4 = Pool3DBlock(2)
        self.encoder_res4 = Res3DBlock(128, 128, normalization_type)
        self.encoder_pool5 = Pool3DBlock(2)
        self.encoder_res5 = Res3DBlock(128, 128, normalization_type)

        self.mid_res = Res3DBlock(128, 128, normalization_type)

        self.decoder_res5 = Res3DBlock(128, 128, normalization_type)
        self.decoder_upsample5 = Upsample3DBlock(128, 128, 2, 2, normalization_type)
        self.decoder_res4 = Res3DBlock(128, 128, normalization_type)
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
        x = self.encoder_pool4(x)
        x = self.encoder_res4(x, params[16:18])
        skip_x5 = self.skip_res5(x, params[18:20])
        x = self.encoder_pool5(x)
        x = self.encoder_res5(x, params[20:22]) 

        x = self.mid_res(x, params[22:24])

        x = self.decoder_res5(x, params[24:26])
        x = self.decoder_upsample5(x, params[26])
        x = x + skip_x5
        x = self.decoder_res4(x, params[27:29])
        x = self.decoder_upsample4(x, params[29])
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
    def __init__(self, input_channels, output_channels, normalization_type='batch_norm'):
            super().__init__()
            self.normalization_type=normalization_type
            self.front_layer1 = Basic3DBlock(input_channels, 32, 7, normalization_type)
            self.front_layer2 = Res3DBlock(32, 32, normalization_type)
            self.front_layer3 = Res3DBlock(32, 32, normalization_type)
            self.front_layer4 = Res3DBlock(32, 32, normalization_type)
            # [32, 32,32, 32,32, 32,32]

            self.encoder_decoder = EncoderDecorder(normalization_type)
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


class EncoderDecorderAdaIN_MiddleVector(EncoderDecorder):
    """docstring for EncoderDecorderAdaIN_Middle"""
    def __init__(self, normalization_type):
        super().__init__(normalization_type)
        self.mid_res = Res3DBlock(128, 128, normalization_type='adain')
    def forward(self, x, params=None):

        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        skip_x3 = self.skip_res3(x)
        x = self.encoder_pool3(x)
        x = self.encoder_res3(x)
        skip_x4 = self.skip_res4(x)
        x = self.encoder_pool4(x)
        x = self.encoder_res4(x)
        skip_x5 = self.skip_res5(x)
        x = self.encoder_pool5(x)
        x = self.encoder_res5(x)

        if params is not None:
            x = self.mid_res(x, params)
        else:
            x = self.mid_res(x)    

        x = self.decoder_res5(x)
        x = self.decoder_upsample5(x)
        x = x + skip_x5
        x = self.decoder_res4(x)
        x = self.decoder_upsample4(x)
        x = x + skip_x4
        x = self.decoder_res3(x)
        x = self.decoder_upsample3(x)
        x = x + skip_x3
        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1 
        return x    
        
class V2VModelAdaIN_MiddleVector(V2VModel):
    def __init__(self, input_channels, output_channels, normalization_type='batch_norm'):
            super().__init__(input_channels, output_channels, normalization_type)
            self.normalization_type=normalization_type
            assert normalization_type != 'adain'
            self.encoder_decoder = EncoderDecorderAdaIN_MiddleVector(normalization_type)
    def forward(self, x, params):
        x = self.front_layer1(x)
        x = self.front_layer2(x)
        x = self.front_layer3(x)
        x = self.front_layer4(x)
        x = self.encoder_decoder(x,params) 
        x = self.back_layer1(x)
        x = self.back_layer2(x)
        x = self.back_layer3(x)
        x = self.output_layer(x)
        return x                         



class EncoderDecorder_v2(nn.Module):
    def __init__(self, normalization_type):
        super().__init__()
        self.normalization_type = normalization_type
        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(128, 128, normalization_type)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(128, 128, normalization_type)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res3 = Res3DBlock(128, 128, normalization_type)
        self.encoder_pool4 = Pool3DBlock(2)
        self.encoder_res4 = Res3DBlock(128, 128, normalization_type)
        self.encoder_pool5 = Pool3DBlock(2)
        self.encoder_res5 = Res3DBlock(128, 256, normalization_type)

        self.mid_res = Res3DBlock(256, 256, normalization_type) 

        self.decoder_res5 = Res3DBlock(256, 128, normalization_type)
        self.decoder_upsample5 = Upsample3DBlock(128, 128, 2, 2, normalization_type)

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
        x = self.encoder_res1(x, params[3:5])
        skip_x2 = self.skip_res2(x, params[5:8])
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x, params[8:10])
        skip_x3 = self.skip_res3(x, params[10:13])
        x = self.encoder_pool3(x)
        x = self.encoder_res3(x, params[13:15])
        skip_x4 = self.skip_res4(x, params[15:17])
        x = self.encoder_pool4(x)
        x = self.encoder_res4(x, params[17:19])
        skip_x5 = self.skip_res5(x, params[19:21])
        x = self.encoder_pool5(x)
        x = self.encoder_res5(x, params[21:24]) 

        x = self.mid_res(x, params[24:26])

        x = self.decoder_res5(x, params[26:29])
        x = self.decoder_upsample5(x, params[29])
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
    def __init__(self, input_channels, output_channels, normalization_type='batch_norm'):
            super().__init__()
            self.normalization_type=normalization_type
            self.front_layer1 = Res3DBlock(input_channels, 128, normalization_type)
            self.front_layer2 = Res3DBlock(128, 128, normalization_type)
            # [128,128  128,128]

            self.encoder_decoder = EncoderDecorder_v2(normalization_type)
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
        x = self.encoder_decoder(x, params[4:48]) 
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