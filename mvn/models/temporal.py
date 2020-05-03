import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from IPython.core.debugger import set_trace
from torchvision import models
from pytorch_convolutional_rnn.convolutional_rnn import Conv2dLSTM, Conv2dLSTMCell, Conv3dLSTMCell
from mvn.models.v2v import Res3DBlock


class FeatureDecoderLSTM(nn.Module):
    def __init__(self, style_vector_dim, feature_space, hidden_dim=512, time=5):
        super().__init__()
        self.style_vector_dim = style_vector_dim
        self.feature_space = feature_space
        self.hidden_dim = hidden_dim
        
        self.style2cell_state = nn.Sequential(nn.ConvTranspose3d(style_vector_dim,
                                                                 hidden_dim//4,
                                                                 kernel_size=[1,2,2], 
                                                                 stride=[1,2,2]),
                                             nn.LeakyReLU(), 
                                             nn.GroupNorm(32, hidden_dim//4),
                                             nn.ConvTranspose3d(hidden_dim//4,
                                                                hidden_dim//2,
                                                                kernel_size=[1,2,2],
                                                                stride=[1,2,2]),
                                             nn.LeakyReLU(), 
                                             nn.GroupNorm(32, hidden_dim//2),
                                             nn.Conv3d(hidden_dim//2,
                                                        hidden_dim,
                                                        kernel_size=[time,1,1],
                                                        stride=1),
                                             nn.LeakyReLU()
                                             )
        
        self.lstm_cell = Conv2dLSTMCell(in_channels=feature_space,
                                        out_channels=hidden_dim,
                                        kernel_size=3,
                                        bias=True)
        
        self.hidden2feature = nn.Conv2d(hidden_dim, feature_space, kernel_size=1)

    def forward(self, style_vector, init_feature_map, time=5):
        
        hx_init = torch.zeros(1, self.hidden_dim, 96, 96).cuda()
        cx_init = self.style2cell_state(style_vector).squeeze(2)
        # set_trace()
        output = []
        for i in range(time):
            
            if i == 0:
                hx, cx = self.lstm_cell(init_feature_map, (hx_init, cx_init))
            else:
                hx, cx = self.lstm_cell(feature, (hx, cx))
            
            feature = self.hidden2feature(hx)
            output.append(feature)    
        
        output = torch.stack(output,1)
        return output
        


class StylePosesLSTM(nn.Module):

    '''
    use_style_pose_lstm_loss: true
    style_pose_lstm_loss_weight: 0.01
    style_pose_lstm_decoder_lr: 0.001
    '''

    def __init__(self, 
                style_vector_dim,
                style_shape, 
                pose_space=17, 
                hidden_dim=128, 
                volume_size=32, 
                time=5, 
                n_layers=3):

        super().__init__()
        self.style_vector_dim = style_vector_dim
        self.pose_space = pose_space
        self.hidden_dim = hidden_dim
        self.volume_size = volume_size
        self.n_layers = n_layers
        self.style_shape = style_shape

        hidden_shape = np.array([volume_size,volume_size,volume_size]) 
        diff_shape = (hidden_shape - style_shape) 
        kernel_size = 1 + (diff_shape//n_layers)
        kernel_size_residual = 1 + diff_shape%n_layers
        assert (diff_shape > 0).all()

        layers = []
        for i in range(n_layers):
            layers.append(nn.ConvTranspose3d(style_vector_dim if i==0 else hidden_dim//2,
                                             hidden_dim//2,
                                             kernel_size=kernel_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.GroupNorm(32, hidden_dim//2))
            
        # last block    
        layers.append(nn.ConvTranspose3d(hidden_dim//2,
                                         hidden_dim,
                                         kernel_size=kernel_size_residual))
        layers.append(nn.LeakyReLU())    

        self.style2cell_state = nn.Sequential(*layers)
        
        self.lstm_cell = Conv3dLSTMCell(in_channels=pose_space,
                                        out_channels=hidden_dim,
                                        kernel_size=3,
                                        bias=True)
        
        self.hidden2feature = nn.Conv3d(hidden_dim, pose_space, kernel_size=1)

    def forward(self, style_vector, init_pose, time=5):
        
        hx_init = torch.zeros(1, 
                                self.hidden_dim,
                                self.volume_size, 
                                self.volume_size, 
                                self.volume_size).cuda()

        cx_init = self.style2cell_state(style_vector)
        output = []
        for i in range(time):
            
            if i == 0:
                hx, cx = self.lstm_cell(init_pose, (hx_init, cx_init))
            else:
                hx, cx = self.lstm_cell(pose, (hx, cx))
            
            pose = self.hidden2feature(hx)
            output.append(pose)    
        
        output = torch.stack(output,1)
        return output


        

def get_encoder(encoder_type, 
                backbone_type,
                encoded_feature_space, 
                upscale_bottleneck,
                capacity=2, 
                spatial_dimension=1, 
                encoder_normalization_type='batch_norm'):
    
    assert spatial_dimension in [1,2], 'Wrong spatial_dimension! Only 1 and 2 are supported'
    encoder_input_channels = {'features':256,
                              'backbone':2048}[encoder_type]

    if encoder_type == "backbone":
        if spatial_dimension == 1:
            return  FeaturesEncoder_Bottleneck(encoded_feature_space,
                                               encoder_input_channels,     
                                               C = capacity, 
                                               normalization_type=encoder_normalization_type)
        else: #spatial_dimension == 2:

            input_size, target_size = {'resnet152':[12,96],
                                       'resnet50':[12,96]}[backbone_type] 

            return  FeaturesEncoder_Bottleneck2D(encoded_feature_space,
                                                 encoder_input_channels,
                                                 C=capacity, 
                                                 normalization_type=encoder_normalization_type, 
                                                 upscale=upscale_bottleneck, 
                                                 input_size=input_size, 
                                                 target_size=target_size,
                                                 upscale_kernel_size=2)
    elif encoder_type == "features":
        if spatial_dimension == 1:
            raise NotImplementedError()
        else: #spatial_dimension == 2:
            return  FeaturesEncoder_Features2D(encoder_input_channels,
                                               encoded_feature_space,
                                               C = capacity, 
                                               normalization_type=encoder_normalization_type)
    else:
        raise RuntimeError('Wrong encoder_type! Only `features` and `backbone` are supported')


def get_normalization(normalization_type, out_planes, n_groups, dimension=2):

    if normalization_type ==  'batch_norm':
        if dimension == 1:
            return nn.BatchNorm1d(out_planes)
        elif dimension == 2:
            return nn.BatchNorm2d(out_planes)
        else:
            raise RuntimeError('{} is unknown n_dimension')       
    elif normalization_type == 'group_norm':
        return nn.GroupNorm(n_groups, out_planes)
    else:
        raise RuntimeError('{} is unknown normalization_type for this model'.format(normalization_type))   

class Slice(nn.Module):
    def __init__(self, shift):
        super(Slice, self).__init__()
        self.shift = shift
    def forward(self, x):
        return x[:, :, self.shift : x.shape[2] - self.shift]

class Res1DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_groups = 32,
                 kernel_size=3, normalization_type='group_norm'):
        
        super(Res1DBlock, self).__init__()
        pad = kernel_size // 2 if kernel_size > 1 else 0
        self.res_branch = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, padding=pad),
            get_normalization(normalization_type, out_planes, n_groups=n_groups, dimension=1),
            nn.ReLU(True),
            nn.Conv1d(out_planes, out_planes, kernel_size=kernel_size),
            get_normalization(normalization_type, out_planes, n_groups=n_groups, dimension=1)
        )

        if in_planes == out_planes:
            self.skip_con = Slice(kernel_size // 2)
        else:
            self.skip_con = nn.Sequential(
                Slice(kernel_size // 2),
                nn.Conv1d(in_planes, out_planes, kernel_size=1),
                get_normalization(normalization_type, out_planes, n_groups=n_groups, dimension=1)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        
        return F.relu(res + skip, True)


class Seq2VecRNN2D(nn.Module):
    """Maps feature-map sequence [batch_size, dt, C1, H, W] 
    to feature-map [batch_size, C2, H, W]"""
    def __init__(self, input_features_dim, output_features_dim=64, hidden_dim=64):
        super(Seq2VecRNN2D, self).__init__()
        self.input_features_dim = input_features_dim
        self.output_features_dim = output_features_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = Conv2dLSTM(in_channels=input_features_dim,  # Corresponds to input size
                                   out_channels=hidden_dim,  # Corresponds to hidden size
                                   kernel_size=3,  # Int or List[int]
                                   num_layers=2,
                                   bidirectional=True,
                                   dilation=1, 
                                   stride=1,
                                   dropout=0.0,
                                   batch_first=True)

        self.output_layer = nn.Conv2d(2*self.hidden_dim, self.output_features_dim, kernel_size=1)
        self.activation = nn.ReLU()
        
    def forward(self, features, eps = 1e-3, device='cuda:0'):
        # [batch_size, dt, feature_shape]
        batch_size = features.shape[0]
        output, _ = self.lstm(features, None)
        output = output[:,-1,...]
        output = self.activation(self.output_layer(output))
        return output


class Seq2VecCNN2D(nn.Module):
    """Maps feature-map sequence [batch_size, dt, C1, H, W] 
    to feature-map [batch_size, C2, H, W]"""
    def __init__(self, 
                 input_features_dim, 
                 output_features_dim=32, 
                 intermediate_channels = 128,
                 normalization_type='group_norm',
                 kernel_size = 3,
                 dt = 8):
        
        super(Seq2VecCNN2D, self).__init__()
        self.input_features_dim = input_features_dim
        self.output_features_dim = output_features_dim
        self.normalization_type = normalization_type
        self.dt = dt
        
        self.first_block = nn.Conv3d(input_features_dim, 
                                      intermediate_channels,
                                      kernel_size=(1,kernel_size,kernel_size))

        l = dt
        blocks =  []
        while l >= kernel_size:
            l = l - kernel_size + 1
            
            blocks.append(nn.Conv3d(intermediate_channels, 
                                 intermediate_channels,
                                 kernel_size = kernel_size,
                                 padding = (0,1,1)))
            blocks.append(nn.GroupNorm(32, intermediate_channels))
            blocks.append(Res3DBlock(intermediate_channels, 
                                     intermediate_channels,
                                     normalization_type=normalization_type,
                                     kernel_size = kernel_size,
                                     padding = 1))
            
        self.blocks = nn.Sequential(*blocks)    
        self.final_block = nn.Conv3d(intermediate_channels, 
                                      output_features_dim,
                                      kernel_size=(l,1,1),
                                      padding=(0,1,1))
        
    def forward(self, x, device='cuda:0'):
        # [batch_size, dt, channels, 96, 96]
        x = x.transpose(1,2)
        # [batch_size, channels, dt, 96, 96]
        x  = self.first_block(x)
        x  = self.blocks(x)
        x  = self.final_block(x)
        
        return x[:,:,0,...] # squeeze time dimension



class Seq2VecCNN(nn.Module):
    """docstring for Seq2VecCNN"""
    def __init__(self, 
                 input_features_dim, 
                 output_features_dim=1024, 
                 intermediate_channels=512, 
                 normalization_type='group_norm',
                 dt = 8,
                 kernel_size = 3,
                 n_groups = 32):
        
        super(Seq2VecCNN, self).__init__()
        self.input_features_dim = input_features_dim
        self.output_features_dim = output_features_dim
        self.intermediate_channels = intermediate_channels
        self.normalization_type = normalization_type
        
        self.first_block = Res1DBlock(input_features_dim, 
                                      intermediate_channels,
                                      kernel_size=1,
                                      normalization_type=normalization_type)
        
        l = dt
        blocks =  []
        while l >= kernel_size:
            l = l - kernel_size + 1
            blocks.append(Res1DBlock(intermediate_channels, 
                                     intermediate_channels, 
                                     kernel_size=kernel_size,
                                     normalization_type=normalization_type))

            blocks.append(nn.Sequential(nn.Conv1d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1),
                                        get_normalization(normalization_type, intermediate_channels, n_groups=n_groups, dimension=1),
                                        nn.ReLU(True),
                                        nn.Conv1d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1),
                                        get_normalization(normalization_type, intermediate_channels, n_groups=n_groups, dimension=1),
                                        nn.ReLU(True)
                                        ))
        
        self.blocks = nn.Sequential(*blocks)    
        self.final_block = nn.Conv1d(intermediate_channels, 
                                      output_features_dim,
                                      kernel_size=l)
        
    def forward(self, x, device='cuda:0'):
        # [batch_size, dt, feature_shape]
        x = x.transpose(1,2) # [batch_size, dt, feature_shape] -> [batch_size, feature_shape, dt]
        x  = self.first_block(x)
        x  = self.blocks(x)
        x  = self.final_block(x)
        
        return x[...,0]


class Seq2VecRNN(nn.Module):
    """docstring for Seq2VecRNN"""
    def __init__(self, input_features_dim, output_features_dim=1024, hidden_dim = 1024):
        super(Seq2VecRNN, self).__init__()
        self.input_features_dim = input_features_dim
        self.output_features_dim = output_features_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(self.input_features_dim, self.hidden_dim, batch_first=True)
        
        if self.output_features_dim != self.hidden_dim :
            self.output_layer = nn.Linear(self.hidden_dim, self.output_features_dim)
            self.activation = nn.ReLU()
        
    def forward(self, features, eps = 1e-3, device='cuda:0'):
        # [bathc_size, dt, feature_shape]
        batch_size = features.shape[0]
        (h0, c0) = torch.randn(1, batch_size, self.hidden_dim, device=device)*eps,\
                   torch.randn(1, batch_size, self.hidden_dim, device=device)*eps
        output, (hn, cn) = self.lstm(features, (h0, c0))
        output = output[:,-1,...]
        if self.output_features_dim != self.hidden_dim:
            output = self.activation(self.output_layer(output))
        return output
        

class FeaturesEncoder_DenseNet(nn.Module):
    """docstring for FeaturesEncoder_DenseNet"""
    def __init__(self, input_features_dim, output_features_dim, pretrained=False):
        super().__init__()
        self.input_features_dim = input_features_dim
        self.output_features_dim = output_features_dim
        self.pretrained = pretrained
        self.backbone = models.densenet121(pretrained=pretrained).features
        self.backbone.conv0 = nn.Sequential(nn.Conv2d(input_features_dim,
                                                      input_features_dim//2, 
                                                      kernel_size=3,
                                                      stride = 2),
                                           nn.Conv2d(input_features_dim//2,
                                                     64,
                                                     kernel_size=3,
                                                     stride = 2))
        self.output = nn.Linear(1024, output_features_dim) if output_features_dim != 1024 else nn.Sequential()
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.backbone(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.output(x)
        x = self.activation(x)
        return x   


class FeaturesEncoder_Features2D(nn.Module):
    """docstring for FeaturesEncoder_Features2D"""
    def __init__(self, input_features_dim, output_features_dim, C = 2, multiplier=128, n_groups=32, normalization_type='batch_norm'):
        super().__init__()
        self.output_features_dim = output_features_dim
        self.input_features_dim = input_features_dim
        self.C = C
        self.multiplier = multiplier
        
        if self.C == 0:
            self.features = nn.Sequential()
        else:    
            self.features=nn.Sequential(nn.Conv2d(input_features_dim, 
                                                  self.C * self.multiplier, 
                                                  kernel_size=3, 
                                                  padding=1),
                                          get_normalization(normalization_type,
                                                               self.C * self.multiplier, n_groups=n_groups, dimension=2),
                                          nn.ReLU(),
                                          nn.Conv2d(self.C * self.multiplier, 
                                                    self.C * self.multiplier//2, 
                                                    kernel_size=3, 
                                                    padding=1),
                                          get_normalization(normalization_type,
                                                               self.C * self.multiplier//2, n_groups=n_groups, dimension=2),
                                          nn.ReLU(),
                                          nn.Conv2d(self.C * self.multiplier//2,
                                                    self.C * self.multiplier//4, 
                                                    kernel_size=3, 
                                                    padding=1),
                                          get_normalization(normalization_type,
                                                               self.C * self.multiplier//4, n_groups=n_groups, dimension=2),
                                          nn.ReLU(),
                                          nn.Conv2d(self.C * self.multiplier//4,
                                                    self.C * self.multiplier//4, kernel_size=1),
                                          get_normalization(normalization_type,
                                                               self.C * self.multiplier//4, n_groups=n_groups, dimension=2),
                                          nn.ReLU(),
                                          nn.Conv2d(self.C * self.multiplier//4, output_features_dim, kernel_size=1)
                                        )
        
    def forward(self, x):
        # [2048, 12, 12] x size for backbone
        batch_size = x.shape[0]
        x = self.features(x)
        return x     


class FeaturesEncoder_Bottleneck2D(nn.Module):
    """docstring for FeaturesEncoder_Bottleneck2D"""
    def __init__(self,
                 output_features_dim,
                 input_channels,
                 C=2, 
                 multiplier=128, 
                 n_groups=32, 
                 normalization_type='batch_norm', 
                 upscale=False, 
                 input_size=None, 
                 target_size=None,
                 upscale_kernel_size=2):
        
        super().__init__()
        self.output_features_dim = output_features_dim
        self.C = C
        self.multiplier = multiplier
        self.upscale = upscale
        self.upscale_kernel_size = upscale_kernel_size
        self.normalization_type = normalization_type
        self.n_groups = n_groups
        modules=[]
            
        size = input_size 
        in_channels = input_channels
        out_channels = input_channels
        i = 1
        while True:
            size *= upscale_kernel_size
            if size > target_size:
                break
            out_channels = self.C * self.multiplier//i
            modules.append(self._get_upscale_block(in_channels, out_channels))
            if i < 4:
                i*=2
            in_channels = out_channels
        
        self.features = nn.Sequential(*modules)
        self.final_upscale = nn.UpsamplingBilinear2d(size=target_size)
        self.final_layer = nn.Conv2d(out_channels, output_features_dim, kernel_size=1)
    
    def _get_upscale_block(self, in_channels, out_channels):
        
        block = nn.Sequential(nn.Conv2d(in_channels, 
                                        out_channels, 
                                        kernel_size=3, 
                                        padding=1),
                                        get_normalization(self.normalization_type, 
                                                             out_channels, 
                                                             n_groups=self.n_groups, 
                                                             dimension=2))
        if self.upscale:
            block.add_module('upscale',nn.ConvTranspose2d(out_channels, 
                                                          out_channels, 
                                                          kernel_size=self.upscale_kernel_size, 
                                                          stride=self.upscale_kernel_size))
        return block
    
    def forward(self, x):
        # [2048, 12, 12]
        batch_size = x.shape[0]
        x = self.features(x)
        x =  self.final_upscale(x)
        x = self.final_layer(x)
        return x          


class FeaturesEncoder_Bottleneck(nn.Module):
    """docstring for FeaturesEncoder_Bottleneck"""
    def __init__(self, output_features_dim, input_channels, C = 2, multiplier=128, n_groups=32, normalization_type='batch_norm'):
        super().__init__()
        self.output_features_dim = output_features_dim
        self.C = C
        self.multiplier = multiplier
        self.features=nn.Sequential(nn.Conv2d(input_channels, 
                                              self.C * self.multiplier, 
                                              kernel_size=3, 
                                              stride=2),
                                      get_normalization(normalization_type,
                                                           self.C * self.multiplier, n_groups=n_groups, dimension=2),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(self.C * self.multiplier, 
                                                self.C * self.multiplier//2, 
                                                kernel_size=3, 
                                                stride=1),
                                      get_normalization(normalization_type,
                                                           self.C * self.multiplier//2, n_groups=n_groups, dimension=2),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(self.C * self.multiplier//2,
                                                self.C * self.multiplier//4, 
                                                kernel_size=3, 
                                                stride=1),
                                      get_normalization(normalization_type,
                                                           self.C * self.multiplier//4, n_groups=n_groups, dimension=2),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(self.C * self.multiplier//4,
                                                self.C * self.multiplier//4, kernel_size=1),
                                      get_normalization(normalization_type,
                                                           self.C * self.multiplier//4, n_groups=n_groups, dimension=2),
                                      nn.LeakyReLU()
                                    )
        
        self.linear = nn.Linear(self.C * self.multiplier//4, output_features_dim)
        self.activation = nn.LeakyReLU()
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x          
        
