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
                            ADAPTIVE_NORMALIZATION, \
                            MODULES_REQUIRES_NORMALIZAION, \
                            ORDINARY_NORMALIZATIONS


class EncoderDecorder_v1(nn.Module):
    def __init__(self, normalization_type, volume_size, style_vector_channels):
        super().__init__()
        self.normalization_type = normalization_type
        self.style_vector_channels = style_vector_channels
        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64, normalization_type, style_vector_channels=self.style_vector_channels)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128, normalization_type, style_vector_channels=self.style_vector_channels)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res3 = Res3DBlock(128, 128, normalization_type, style_vector_channels=self.style_vector_channels)
        self.volume_size = volume_size
        
        # 1
        if volume_size > 16:
            self.encoder_pool4 = Pool3DBlock(2)
        
        self.encoder_res4 = Res3DBlock(128, 128, normalization_type, style_vector_channels=self.style_vector_channels)
        
        # 2
        if volume_size > 32:
            self.encoder_pool5 = Pool3DBlock(2) 
        
        self.encoder_res5 = Res3DBlock(128, 128, normalization_type, style_vector_channels=self.style_vector_channels)

        self.mid_res = Res3DBlock(128, 128, normalization_type, style_vector_channels=self.style_vector_channels)

        self.decoder_res5 = Res3DBlock(128, 128, normalization_type, style_vector_channels=self.style_vector_channels)
        
        # 2
        if volume_size > 32:
            self.decoder_upsample5 = Upsample3DBlock(128, 128, 2, 2, normalization_type, style_vector_channels=self.style_vector_channels)
        
        self.decoder_res4 = Res3DBlock(128, 128, normalization_type, style_vector_channels=self.style_vector_channels)
        
        # 1
        if volume_size > 16:
            self.decoder_upsample4 = Upsample3DBlock(128, 128, 2, 2, normalization_type, style_vector_channels=self.style_vector_channels)
        
        self.decoder_res3 = Res3DBlock(128, 128, normalization_type, style_vector_channels=self.style_vector_channels)
        self.decoder_upsample3 = Upsample3DBlock(128, 128, 2, 2, normalization_type, style_vector_channels=self.style_vector_channels)
        self.decoder_res2 = Res3DBlock(128, 128, normalization_type, style_vector_channels=self.style_vector_channels)
        self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2, normalization_type, style_vector_channels=self.style_vector_channels)
        self.decoder_res1 = Res3DBlock(64, 64, normalization_type, style_vector_channels=self.style_vector_channels)
        self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2, normalization_type, style_vector_channels=self.style_vector_channels)

        self.skip_res1 = Res3DBlock(32, 32, normalization_type, style_vector_channels=self.style_vector_channels)
        self.skip_res2 = Res3DBlock(64, 64, normalization_type, style_vector_channels=self.style_vector_channels)
        self.skip_res3 = Res3DBlock(128, 128, normalization_type, style_vector_channels=self.style_vector_channels)
        self.skip_res4 = Res3DBlock(128, 128, normalization_type, style_vector_channels=self.style_vector_channels)
        self.skip_res5 = Res3DBlock(128, 128, normalization_type, style_vector_channels=self.style_vector_channels)

    def forward(self, x, params=None):
        skip_x1 = self.skip_res1(x, params)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x, params)
        skip_x2 = self.skip_res2(x, params)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x, params)
        skip_x3 = self.skip_res3(x, params)
        x = self.encoder_pool3(x)
        x = self.encoder_res3(x, params)
        skip_x4 = self.skip_res4(x, params)
        x = self.encoder_pool4(x) if self.volume_size > 16 else x

        x = self.encoder_res4(x, params) 
        
        skip_x5 = self.skip_res5(x, params) 
        x = self.encoder_pool5(x) if self.volume_size > 32 else x
        x = self.encoder_res5(x, params) 

        x = self.mid_res(x, params)

        x = self.decoder_res5(x, params)
        x = self.decoder_upsample5(x, params) if self.volume_size > 32 else x
        x = x + skip_x5
        x = self.decoder_res4(x, params)
        x = self.decoder_upsample4(x, params) if self.volume_size > 16 else x
        x = x + skip_x4
        x = self.decoder_res3(x, params)
        x = self.decoder_upsample3(x, params)
        x = x + skip_x3
        x = self.decoder_res2(x, params)
        x = self.decoder_upsample2(x, params)
        x = x + skip_x2
        x = self.decoder_res1(x, params)
        x = self.decoder_upsample1(x, params)
        x = x + skip_x1

        return x


class V2VModel_v1(nn.Module):
    def __init__(self, input_channels,
                       output_channels, 
                       volume_size, 
                       normalization_type='batch_norm', 
                       temporal_condition_type=None, 
                       style_vector_parameter=False,
                       style_vector_shape=None,
                       style_vector_dim=None):

            super().__init__()

            self.style_vector_parameter = style_vector_parameter
            if self.style_vector_parameter:
                self.style_vector = nn.Parameter(torch.randn(*style_vector_shape)) 

            self.style_vector_channels = style_vector_dim 

            self.normalization_type = normalization_type if (temporal_condition_type is None or temporal_condition_type not in ADAPTIVE_NORMALIZATION) \
                                         else [normalization_type, temporal_condition_type]

            self.front_layer1 = Basic3DBlock(input_channels, 32, 7, self.normalization_type, style_vector_channels=self.style_vector_channels)
            self.front_layer2 = Res3DBlock(32, 32, self.normalization_type, style_vector_channels=self.style_vector_channels)
            self.front_layer3 = Res3DBlock(32, 32, self.normalization_type, style_vector_channels=self.style_vector_channels)
            self.front_layer4 = Res3DBlock(32, 32, self.normalization_type, style_vector_channels=self.style_vector_channels)
            # [32, 32,32, 32,32, 32,32]

            self.encoder_decoder = EncoderDecorder_v1(self.normalization_type, 
                                                        volume_size=volume_size, 
                                                        style_vector_channels=self.style_vector_channels)
            # [32,32, 64,64,64, 64,64, 128,128,128, 128,128, 128,128,
            # 128,128, 128,128, 128,128, 128,128, 128,128, 128,128, 128,
            # 128,128, 128, 128,128, 128, 128,128, 64, 64,64, 32]

            self.back_layer1 = Res3DBlock(32, 32, self.normalization_type, style_vector_channels=self.style_vector_channels)
            self.back_layer2 = Basic3DBlock(32, 32, 1, self.normalization_type, style_vector_channels=self.style_vector_channels)
            self.back_layer3 = Basic3DBlock(32, 32, 1, self.normalization_type, style_vector_channels=self.style_vector_channels)
            # [32,32, 32, 32]

            self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

            self._initialize_weights()

    def forward(self, x, params=None):

        if self.style_vector_parameter:
            params = self.style_vector

        x = self.front_layer1(x, params)
        x = self.front_layer2(x, params)
        x = self.front_layer3(x, params)
        x = self.front_layer4(x, params)

        x = self.encoder_decoder(x, params) 

        x = self.back_layer1(x, params)
        x = self.back_layer2(x, params)
        x = self.back_layer3(x, params)

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
