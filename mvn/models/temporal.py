import torch.nn as nn
import torch.nn.functional as F
import torch
from IPython.core.debugger import set_trace
from torchvision import models


class Slice(nn.Module):
    def __init__(self, shift):
        super(Slice, self).__init__()
        self.shift = shift
    def forward(self,x):
        return x[:, :, self.shift : x.shape[2] - self.shift]

class Res1DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_groups = 32, kernel_size=3):
        
        super(Res1DBlock, self).__init__()
        pad = kernel_size // 2 if kernel_size > 1 else 0
        self.res_branch = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, padding=pad),
            nn.GroupNorm(n_groups, out_planes),
            nn.ReLU(True),
            nn.Conv1d(out_planes, out_planes, kernel_size=kernel_size),
            nn.GroupNorm(n_groups, out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = Slice(kernel_size // 2)
        else:
            self.skip_con = nn.Sequential(
                Slice(kernel_size // 2),
                nn.Conv1d(in_planes, out_planes, kernel_size=1),
                nn.GroupNorm(n_groups, out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        
        return F.relu(res + skip, True)


class Basic2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic2DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Pool2DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool2DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)

class Res2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res2DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Upsample2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample2DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

        
class TemporalDiscriminator(nn.Module):
    def __init__(self, n_blocks=10, in_features=187, out_features=256):
        super().__init__()
        
        self.features=nn.Sequential(
            nn.Conv1d(in_features, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256,256,3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256,out_features,3),
            nn.BatchNorm1d(256),
            nn.ReLU(True))
        self.linear=nn.Linear(out_features, 2)
        self.softmax=nn.Softmax()
        
    def forward(self, x):
        '''
        x - input with shape [1,dt,3*J]
        '''
        x=self.features(x)
        return self.softmax(self.linear(x.squeeze(-1)))


class Seq2VecCNN(nn.Module):
    """docstring for Seq2VecCNN"""
    def __init__(self, 
                 input_features_dim, 
                 output_features_dim=1024, 
                 intermediate_channels=512, 
                 dt = 8,
                 kernel_size = 3):
        
        super(Seq2VecCNN, self).__init__()
        self.input_features_dim = input_features_dim
        self.output_features_dim = output_features_dim
        self.intermediate_channels = intermediate_channels
        
        self.first_block = Res1DBlock(input_features_dim, 
                                      intermediate_channels,
                                      kernel_size=1)
        
        
        l = dt
        blocks =  []
        while l >= kernel_size:
            l = l - kernel_size + 1
            blocks.append(Res1DBlock(intermediate_channels, 
                                     intermediate_channels, 
                                     kernel_size = kernel_size))
        
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
        

class FeaturesDecoder(nn.Module):
    """docstring for FeaturesDecoder"""
    def __init__(self, input_features_dim, output_features_dim):
        super().__init__()
        self.input_features_dim = input_features_dim
        self.output_features_dim = output_features_dim
    def forward(self, x):
        return x    


class FeaturesEncoder_DenseNet(nn.Module):
    """docstring for FeaturesEncoder_DenseNet"""
    def __init__(self, input_features_dim, output_features_dim, pretrained=False, normalization_type='batch_norm'):
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



class FeaturesEncoder_Bottleneck(nn.Module):
    """docstring for FeaturesEncoder_Bottleneck"""
    def __init__(self, output_features_dim, C = 4, multiplier=128):
        super().__init__()
        self.output_features_dim = output_features_dim
        self.C = C
        self.multiplier = multiplier
        self.features=nn.Sequential(nn.Conv2d(2048, 
                                              self.C * self.multiplier, 
                                              kernel_size=3, 
                                              stride=2),
                                      nn.BatchNorm2d(self.C * self.multiplier),
                                      nn.ReLU(),
                                      nn.Conv2d(self.C * self.multiplier, 
                                                self.C * self.multiplier//2, 
                                                kernel_size=3, 
                                                stride=1),
                                      nn.BatchNorm2d(self.C * self.multiplier//2),
                                      nn.ReLU(),
                                      nn.Conv2d(self.C * self.multiplier//2,
                                                self.C * self.multiplier//4, 
                                                kernel_size=3, 
                                                stride=1),
                                      nn.BatchNorm2d(self.C * self.multiplier//4),
                                      nn.ReLU(),
                                      nn.Conv2d(self.C * self.multiplier//4,
                                                self.C * self.multiplier//4, kernel_size=1),
                                      nn.BatchNorm2d(self.C * self.multiplier//4),
                                      nn.ReLU()
                                    )
        
        self.linear = nn.Linear(self.C * self.multiplier//4, output_features_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        x = self.activation(x)
        return x          
        
class FeaturesAR_CNN1D(nn.Module):
    """docstring for FeaturesAR_CNN1D"""
    def __init__(self, input_features_dim, output_features_dim, intermediate_features=400):
        super().__init__()
        self.input_features_dim = input_features_dim
        self.intermediate_features = intermediate_features
        self.encoder = FeaturesEncoder(input_features_dim, output_features_dim=intermediate_features)
        self.decoder = FeaturesDecoder(intermediate_features, output_features_dim=output_features_dim)

        self.cnn1d = nn.Sequential(nn.Conv1d(400,128,2),
                               nn.BatchNorm2d(128),
                               nn.ReLU(),
                               nn.MaxPool1d(2),
                               nn.Conv1d(128,64,2),
                               nn.BatchNorm1d(64),
                               nn.ReLU(),
                               nn.MaxPool1d(2),
                               nn.Conv2d(64,32,1),
                               nn.BatchNorm1d(32),
                               nn.ReLU(),
                               nn.MaxPool1d(2),
                               nn.Conv1d(32,16,1),
                               nn.BatchNorm1d(16),
                               nn.ReLU(),
                               nn.MaxPool1d(2))
        
    def forward(self, features, eps = 1e-3):
        # [batch size, dt, 256, 96, 96]

        device = torch.cuda.current_device()
        features_shape = features.shape[2:]
        batch_size, dt = features.shape[:2]
        encoded_features = self.encoder(features.view(-1, *features_shape))
        encoded_features = encoded_features.view(batch_size, dt, -1)
        predicted_encoded_feature = self.cnn1d(encoded_features)
        decoded_feature = self.decoder(predicted_encoded_feature)
        return decoded_feature


class FeaturesAR_CNN2D_UNet(nn.Module):
    def __init__(self, input_features_dim, output_features_dim, C = 8):
        super().__init__()

        self.front_layer1 = Basic2DBlock(input_features_dim, C*2, 7)
        self.front_layer2 = Res2DBlock(C*2, C*2)
        self.front_layer3 = Res2DBlock(C*2, C*2)
        self.front_layer4 = Res2DBlock(C*2, C*2)

        self.encoder_pool1 = Pool2DBlock(2)
        self.encoder_res1 = Res2DBlock(C*2, C*4)
        self.encoder_pool2 = Pool2DBlock(2)
        self.encoder_res2 = Res2DBlock(C*4, C*8)
        self.encoder_pool3 = Pool2DBlock(2)
        self.encoder_res3 = Res2DBlock(C*8, C*8)
        self.encoder_pool4 = Pool2DBlock(2)
        self.encoder_res4 = Res2DBlock(C*8, C*8)
        self.encoder_pool5 = Pool2DBlock(2)
        self.encoder_res5 = Res2DBlock(C*8, C*8)

        self.mid_res = Res2DBlock(C*8, C*8)

        self.decoder_res5 = Res2DBlock(C*8, C*8)
        self.decoder_upsample5 = Upsample2DBlock(C*8, C*8, 2, 2)
        self.decoder_res4 = Res2DBlock(C*8, C*8)
        self.decoder_upsample4 = Upsample2DBlock(C*8, C*8, 2, 2)
        self.decoder_res3 = Res2DBlock(C*8, C*8)
        self.decoder_upsample3 = Upsample2DBlock(C*8, C*8, 2, 2)
        self.decoder_res2 = Res2DBlock(C*8, C*8)
        self.decoder_upsample2 = Upsample2DBlock(C*8, C*4, 2, 2)
        self.decoder_res1 = Res2DBlock(C*4, C*4)
        self.decoder_upsample1 = Upsample2DBlock(C*4, C*2, 2, 2)

        self.skip_res1 = Res2DBlock(C*2, C*2)
        self.skip_res2 = Res2DBlock(C*4, C*4)
        self.skip_res3 = Res2DBlock(C*8, C*8)
        self.skip_res4 = Res2DBlock(C*8, C*8)
        self.skip_res5 = Res2DBlock(C*8, C*8)

        self.back_layer1 = Res2DBlock(C*2, C*2)
        self.back_layer2 = Basic2DBlock(C*2, C*2, 1)
        self.back_layer3 = Basic2DBlock(C*2, C*2, 1)

        self.output_layer = nn.Conv2d(C*2, output_features_dim, kernel_size=1, stride=1, padding=0)


    def forward(self, x, params=None):
        
        x = self.front_layer1(x)
        x = self.front_layer2(x)
        x = self.front_layer3(x)
        x = self.front_layer4(x)

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

        x = self.back_layer1(x)
        x = self.back_layer2(x)
        x = self.back_layer3(x)

        x = self.output_layer(x)

        return x
       
class FeaturesAR_RNN(object):
  """docstring for FeaturesAR_RNN"""
  def __init__(self, input_features_dim, output_features_dim=None, hidden_dim = 512):
      super().__init__()
      self.seq2vec = Seq2VecRNN(input_features_dim, output_features_dim, hidden_dim = hidden_dim)
      self.decoder = FeaturesDecoder(input_features_dim=hidden_dim, output_features_dim = output_features_dim if output_features_dim is not None else hidden_dim)
  def forward(self, features):
      batch_size = features.shape[0]
      last_hidden_state = self.seq2vec(features)
      last_hidden_state = last_hidden_state.view(batch_size,-1,1,1)
      result = self.decoder(last_hidden_state)
      return result


class FeaturesAR_CNN2D_ResNet(object):
    """docstring for FeaturesAR_CNN2D_ResNet"""
    def __init__(self, arg):
        super(FeaturesAR_CNN2D_ResNet, self).__init__()
        self.arg = arg
        