# Reference: https://github.com/dragonbook/V2V-PoseNet-pytorch
from IPython.core.debugger import set_trace

import torch.nn as nn
import torch.nn.functional as F
 
class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()
    def forward(self, features, params):
        # features: [batch_size, C, D1, D2, D3]
        # params: [batch_size, 2]
        use_adain_params = params is not None
        size = features.size()
        batch_size, C = features.shape[:2]

        if use_adain_params:
            adain_mean = params[:,:C].view(batch_size, C,1,1,1)
            adain_std = params[:,C:].view(batch_size, C,1,1,1)
        else:
           assert batch_size % 2 == 0, \
           'batch should contain concatenated features and style batches'
           batch_size = batch_size / 2
           
           style = features[batch_size:] 
           features = features[:batch_size]

           adain_mean = style.view(batch_size, C, -1).mean(-1).view(batch_size, C,1,1,1)
           adain_std = style.view(batch_size, C, -1).std(-1).view(batch_size, C,1,1,1) 
        
        features_mean = features.view(batch_size, C, -1).mean(-1).view(batch_size, C,1,1,1)
        features_std = features.view(batch_size, C, -1).std(-1).view(batch_size, C,1,1,1)
        norm_features = (features - features_mean) / features_std
                
        return norm_features * adain_std + adain_mean


class Basic3DBlockAdaIN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlockAdaIN, self).__init__()
        self.conv =  nn.Conv3d(in_planes, out_planes, 
                                kernel_size=kernel_size, 
                                stride=1, 
                                padding=((kernel_size-1)//2))
        self.adain = AdaIN()
        self.activation = nn.ReLU(True)

    def forward(self, x, params=None):
        x = self.conv(x)
        x = self.adain(x, params)
        x = self.activation(x)
        return x


class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Res3DBlockAdaIN(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlockAdaIN, self).__init__()
        self.res_conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.res_adain1 = AdaIN()
        self.res_conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.res_adain2 = AdaIN()
        self.activation = nn.ReLU(True)

        self.use_skip_con = ( in_planes != out_planes)
        if self.use_skip_con:
            self.skip_con_conv = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
            self.skip_con_adain = AdaIN()

    def forward(self, x, params=None):
        use_adain_params=params is not None

        if self.use_skip_con:
            skip = self.skip_con_conv(x)
            skip = self.skip_con_adain(skip, params[2] if use_adain_params else None)
        else:
            skip = x

        x = self.res_conv1(x)
        x = self.res_adain1(x, params[0])
        x = self.activation(x)
        x = self.res_conv2(x)
        x = self.res_adain2(x, params[1])

        return F.relu(x + skip, True)        


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)

class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Upsample3DBlockAdaIN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlockAdaIN, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.transpose = nn.ConvTranspose3d(in_planes, 
                                            out_planes, 
                                            kernel_size=kernel_size, 
                                            stride=stride, 
                                            padding=0, 
                                            output_padding=0)
        self.adain = AdaIN()
        self.activation = nn.ReLU(True)

    def forward(self, x, params=None):
        x = self.transpose(x)
        x = self.adain(x,params)
        x = self.activation(x)
        return x       


class EncoderDecorder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res3 = Res3DBlock(128, 128)
        self.encoder_pool4 = Pool3DBlock(2)
        self.encoder_res4 = Res3DBlock(128, 128)
        self.encoder_pool5 = Pool3DBlock(2)
        self.encoder_res5 = Res3DBlock(128, 128)

        self.mid_res = Res3DBlock(128, 128)

        self.decoder_res5 = Res3DBlock(128, 128)
        self.decoder_upsample5 = Upsample3DBlock(128, 128, 2, 2)
        self.decoder_res4 = Res3DBlock(128, 128)
        self.decoder_upsample4 = Upsample3DBlock(128, 128, 2, 2)
        self.decoder_res3 = Res3DBlock(128, 128)
        self.decoder_upsample3 = Upsample3DBlock(128, 128, 2, 2)
        self.decoder_res2 = Res3DBlock(128, 128)
        self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2)
        self.decoder_res1 = Res3DBlock(64, 64)
        self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2)

        self.skip_res1 = Res3DBlock(32, 32)
        self.skip_res2 = Res3DBlock(64, 64)
        self.skip_res3 = Res3DBlock(128, 128)
        self.skip_res4 = Res3DBlock(128, 128)
        self.skip_res5 = Res3DBlock(128, 128)

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


class EncoderDecorderAdaIN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlockAdaIN(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlockAdaIN(64, 128)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res3 = Res3DBlockAdaIN(128, 128)
        self.encoder_pool4 = Pool3DBlock(2)
        self.encoder_res4 = Res3DBlockAdaIN(128, 128)
        self.encoder_pool5 = Pool3DBlock(2)
        self.encoder_res5 = Res3DBlockAdaIN(128, 128)

        self.mid_res = Res3DBlockAdaIN(128, 128)

        self.decoder_res5 = Res3DBlockAdaIN(128, 128)
        self.decoder_upsample5 = Upsample3DBlockAdaIN(128, 128, 2, 2)
        self.decoder_res4 = Res3DBlockAdaIN(128, 128)
        self.decoder_upsample4 = Upsample3DBlockAdaIN(128, 128, 2, 2)
        self.decoder_res3 = Res3DBlockAdaIN(128, 128)
        self.decoder_upsample3 = Upsample3DBlockAdaIN(128, 128, 2, 2)
        self.decoder_res2 = Res3DBlockAdaIN(128, 128)
        self.decoder_upsample2 = Upsample3DBlockAdaIN(128, 64, 2, 2)
        self.decoder_res1 = Res3DBlockAdaIN(64, 64)
        self.decoder_upsample1 = Upsample3DBlockAdaIN(64, 32, 2, 2)

        self.skip_res1 = Res3DBlockAdaIN(32, 32)
        self.skip_res2 = Res3DBlockAdaIN(64, 64)
        self.skip_res3 = Res3DBlockAdaIN(128, 128)
        self.skip_res4 = Res3DBlockAdaIN(128, 128)
        self.skip_res5 = Res3DBlockAdaIN(128, 128)

    def forward(self, x, params):
        use_adain_params = params is not None
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
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 16, 7),
            Res3DBlock(16, 32),
            Res3DBlock(32, 32),
            Res3DBlock(32, 32)
        )
        
        self.encoder_decoder = EncoderDecorder()

        self.back_layers = nn.Sequential(
            Res3DBlock(32, 32),
            Basic3DBlock(32, 32, 1),
            Basic3DBlock(32, 32, 1),
        )

        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.back_layers(x)
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


class V2VModelAdaIN(nn.Module):
    def __init__(self, input_channels, output_channels):
            super().__init__()

            self.front_layer1 = Basic3DBlockAdaIN(input_channels, 16, 7)
            self.front_layer2 = Res3DBlockAdaIN(16, 32)
            self.front_layer3 = Res3DBlockAdaIN(32, 32)
            self.front_layer4 = Res3DBlockAdaIN(32, 32)
            # [16, 32,32,32, 32,32, 32,32]

            self.encoder_decoder = EncoderDecorderAdaIN()
            # [32,32, 64,64,64, 64,64, 128,128,128, 128,128, 128,128,
            # 128,128, 128,128, 128,128, 128,128, 128,128, 128,128, 128,
            # 128,128, 128, 128,128, 128, 128,128, 64, 64,64, 32]

            self.back_layer1 =Res3DBlockAdaIN(32, 32)
            self.back_layer2 =Basic3DBlockAdaIN(32, 32, 1)
            self.back_layer3 =Basic3DBlockAdaIN(32, 32, 1)
            # [32,32, 32, 32]

            self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

            self._initialize_weights()

    def forward(self, x, adain_params):
        use_adain_params = adain_params is not None
        x = self.front_layer1(x, adain_params[0])
        x = self.front_layer2(x, adain_params[1:4])
        x = self.front_layer3(x, adain_params[4:6])
        x = self.front_layer4(x, adain_params[6:8])
        x = self.encoder_decoder(x, adain_params[8:47]) 
        x = self.back_layer1(x, adain_params[47:49])
        x = self.back_layer2(x, adain_params[49])
        x = self.back_layer3(x, adain_params[50])
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
    def __init__(self, arg):
        super().__init__()
        self.mid_res = Res3DBlockAdaIN(128, 128)
        
class V2VModelAdaIN_MiddleVector(V2VModel):
    def __init__(self, input_channels, output_channels):
            super().__init__()
            self.encoder_decoder = EncoderDecorderAdaIN_MiddleVector()                 