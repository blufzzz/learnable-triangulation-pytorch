import torch.nn as nn
import torch.nn.functional as F
import torch
from IPython.core.debugger import set_trace
from torchvision import models
from pytorch_convolutional_rnn.convolutional_rnn import Conv2dLSTM, Conv2dPeepholeLSTM
from mvn.models.v2v import Res3DBlock

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


class Seq2VecRNN2D(nn.Module):
    """docstring for Seq2VecRNN"""
    def __init__(self, input_features_dim, output_features_dim=64, hidden_dim = 64):
        super(Seq2VecRNN2D, self).__init__()
        self.input_features_dim = input_features_dim
        self.output_features_dim = output_features_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = True
        
        self.lstm = Conv2dLSTM(in_channels=input_features_dim,  # Corresponds to input size
                                   out_channels=hidden_dim,  # Corresponds to hidden size
                                   kernel_size=3,  # Int or List[int]
                                   num_layers=2,
                                   bidirectional=self.bidirectional,
                                   dilation=1, 
                                   stride=1,
                                   dropout=0.2,
                                   batch_first=True)

        self.output_layer = nn.Conv2d(2*self.hidden_dim if self.bidirectional else self.hidden_dim,
                                      self.output_features_dim, kernel_size=1)
        self.activation = nn.ReLU()
        
    def forward(self, features, eps = 1e-3, device='cuda:0'):
        # [bathc_size, dt, feature_shape]
        batch_size = features.shape[0]
        output, _ = self.lstm(features, None)
        output = output[:,-1,...]
        output = self.activation(self.output_layer(output))
        return output


class Seq2VecCNN2D(nn.Module):
    """docstring for Seq2VecCNN2D"""
    def __init__(self, 
                 input_features_dim, 
                 output_features_dim=32, 
                 intermediate_channel = 128,
                 normalization_type='group_norm',
                 kernel_size = 3,
                 dt = 8):
        
        super(Seq2VecCNN2D, self).__init__()
        self.input_features_dim = input_features_dim
        self.output_features_dim = output_features_dim
        self.normalization_type = normalization_type
        self.dt = dt
        
        self.first_block = nn.Conv3d(input_features_dim, 
                                      intermediate_channel,
                                      kernel_size=(1,kernel_size,kernel_size))

        l = dt
        blocks =  []
        while l >= kernel_size:
            l = l - kernel_size + 1
            
            blocks.append(nn.Conv3d(intermediate_channel, 
                                 intermediate_channel,
                                 kernel_size = kernel_size,
                                 padding = (0,1,1)))
            blocks.append(nn.GroupNorm(32, intermediate_channel))
            blocks.append(Res3DBlock(intermediate_channel, 
                                     intermediate_channel,
                                     normalization_type=normalization_type,
                                     kernel_size = kernel_size,
                                     padding = 1))
            
        self.blocks = nn.Sequential(*blocks)    
        self.final_block = nn.Conv3d(intermediate_channel, 
                                      output_features_dim,
                                      kernel_size=(l,1,1),
                                      padding=(0,1,1))
        
    def forward(self, x, device='cuda:0'):
        # [batch_size, dt,channels dt,96.96]
        x = x.transpose(1,2)
        x  = self.first_block(x)
        x  = self.blocks(x)
        x  = self.final_block(x)
        
        return x[:,:,0,...]



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



class FeaturesEncoder_Bottleneck(nn.Module):
    """docstring for FeaturesEncoder_Bottleneck"""
    def __init__(self, output_features_dim, C = 4, multiplier=128, n_groups=32):
        super().__init__()
        self.output_features_dim = output_features_dim
        self.C = C
        self.multiplier = multiplier
        self.features=nn.Sequential(nn.Conv2d(2048, 
                                              self.C * self.multiplier, 
                                              kernel_size=3, 
                                              stride=2),
                                      nn.GroupNorm(n_groups, self.C * self.multiplier),
                                      nn.ReLU(),
                                      nn.Conv2d(self.C * self.multiplier, 
                                                self.C * self.multiplier//2, 
                                                kernel_size=3, 
                                                stride=1),
                                      nn.GroupNorm(n_groups, self.C * self.multiplier//2),
                                      nn.ReLU(),
                                      nn.Conv2d(self.C * self.multiplier//2,
                                                self.C * self.multiplier//4, 
                                                kernel_size=3, 
                                                stride=1),
                                      nn.GroupNorm(n_groups, self.C * self.multiplier//4),
                                      nn.ReLU(),
                                      nn.Conv2d(self.C * self.multiplier//4,
                                                self.C * self.multiplier//4, kernel_size=1),
                                      nn.GroupNorm(n_groups, self.C * self.multiplier//4),
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
        
