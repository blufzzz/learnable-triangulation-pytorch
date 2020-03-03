# Reference: https://github.com/dragonbook/V2V-PoseNet-pytorch
from IPython.core.debugger import set_trace
import torch.nn as nn
import torch.nn.functional as F
import torch
from mvn.models.v2v import SPADE, \
                            AdaIN, \
                            CompoundNorm, \
                            get_normalization, \
                            Basic3DBlock, \
                            Res3DBlock, \
                            Pool3DBlock, \
                            Upsample3DBlock, \
                            MODULES_REQUIRES_NORMALIZAION, \
                            STYLE_VECTOR_CHANNELS


class EncoderDecorder_v1(nn.Module):
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
        x = self.encoder_res4(x, params[16:18]) # NAN
        
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


class V2VModel_v1(nn.Module):
    def __init__(self, input_channels, output_channels, volume_size, normalization_type='batch_norm'):
            super().__init__()
            self.normalization_type=normalization_type
            self.front_layer1 = Basic3DBlock(input_channels, 32, 7, normalization_type)
            self.front_layer2 = Res3DBlock(32, 32, normalization_type)
            self.front_layer3 = Res3DBlock(32, 32, normalization_type)
            self.front_layer4 = Res3DBlock(32, 32, normalization_type)
            # [32, 32,32, 32,32, 32,32]

            self.encoder_decoder = EncoderDecorder_v1(normalization_type, volume_size=volume_size)
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