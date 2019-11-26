import torch.nn as nn
import torch.nn.functional as F
import torch

from IPython.core.debugger import set_trace
        
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


class Seq2VecRNN(nn.Module):
    """docstring for Seq2VecModel"""
    def __init__(self, input_features_dim, output_features_dim, hidden_dim = 512):
        super(Seq2VecModel, self).__init__()
        self.input_features_dim = input_features_dim
        self.hidden_dim = hidden_dim
        self.output_features_dim = output_features_dim
        self.feature2vector = nn.Sequential(nn.Conv2d(self.input_features_dim,128,3),
                               nn.BatchNorm2d(128),
                               nn.ReLU(),
                               nn.MaxPool2d(2),
                               nn.Conv2d(128,64,3),
                               nn.BatchNorm2d(64),
                               nn.ReLU(),
                               nn.MaxPool2d(2),
                               nn.Conv2d(64,32,1),
                               nn.BatchNorm2d(32),
                               nn.ReLU(),
                               nn.MaxPool2d(2),
                               nn.Conv2d(32,16,1),
                               nn.BatchNorm2d(16),
                               nn.ReLU(),
                               nn.MaxPool2d(2))
        self.lstm = nn.LSTM(400, 512, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_features_dim)
        self.activation = nn.ReLU()
        
    def forward(self, features, eps = 1e-3):
        # [batch size, dt, 256, 96, 96]

        device = torch.cuda.current_device()
        features_shape = features.shape[2:]
        batch_size, dt = features.shape[:2]
        vectors = self.feature2vector(features.view(-1, *features_shape))
        vectors = vectors.view(batch_size, dt, -1)
        (h0, c0) = torch.randn(1, batch_size, self.hidden_dim, device=device)*eps,\
                   torch.randn(1, batch_size, self.hidden_dim, device=device)*eps
        output, (hn, cn) = self.lstm(vectors, (h0, c0))
        output = self.activation(self.output_layer(output[:,-1,...]))
        return output
        
        
class FeaturesAR_CNN1D(nn.Module):
    """docstring for FeaturesAR_CNN1D"""
    def __init__(self, input_features_dim, output_features_dim):
        super(FeaturesAR_CNN1D, self).__init__()
        self.input_features_dim = input_features_dim

        self.feature2vector = nn.Sequential(nn.Conv2d(self.features_dim,128,3),
                               nn.BatchNorm2d(128),
                               nn.ReLU(),
                               nn.MaxPool2d(2),
                               nn.Conv2d(128,64,3),
                               nn.BatchNorm2d(64),
                               nn.ReLU(),
                               nn.MaxPool2d(2),
                               nn.Conv2d(64,32,1),
                               nn.BatchNorm2d(32),
                               nn.ReLU(),
                               nn.MaxPool2d(2),
                               nn.Conv2d(32,16,1),
                               nn.BatchNorm2d(16),
                               nn.ReLU(),
                               nn.MaxPool2d(2))
        # 400
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
        vectors = self.feature2vector(features.view(-1, *features_shape))
        vectors = vectors.view(batch_size, dt, -1)
          

        return 


       
class FeaturesAR_CNN2D(nn.Module):
    """docstring for FeaturesAR_CNN2D"""
    def __init__(self, features_dim, hidden_dim = 512):
        super(FeaturesAR_CNN2D, self).__init__()
        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.feature2vector = nn.Sequential(nn.Conv2d(self.features_dim,128,3),
                               nn.BatchNorm2d(128),
                               nn.ReLU(),
                               nn.MaxPool2d(2),
                               nn.Conv2d(128,64,3),
                               nn.BatchNorm2d(64),
                               nn.ReLU(),
                               nn.MaxPool2d(2),
                               nn.Conv2d(64,32,1),
                               nn.BatchNorm2d(32),
                               nn.ReLU(),
                               nn.MaxPool2d(2),
                               nn.Conv2d(32,16,1),
                               nn.BatchNorm2d(16),
                               nn.ReLU(),
                               nn.MaxPool2d(2))
        self.lstm = nn.LSTM(400, 512, batch_first=True)
        
    def forward(self, features, eps = 1e-3):
        # [batch size, dt, 256, 96, 96]

        device = torch.cuda.current_device()
        features_shape = features.shape[2:]
        batch_size, dt = features.shape[:2]
        vectors = self.feature2vector(features.view(-1, *features_shape))
        vectors = vectors.view(batch_size, dt, -1)
        (h0, c0) = torch.randn(1, batch_size, self.hidden_dim, device=device)*eps,\
                   torch.randn(1, batch_size, self.hidden_dim, device=device)*eps
        output, (hn, cn) = self.lstm(vectors, (h0, c0))
        return output[:,-1,...]        
        
        
class FeaturesAR_RNN(object):
  """docstring for FeaturesAR_RNN"""
  def __init__(self, arg):
    super(FeaturesAR_RNN, self).__init__()
    self.arg = arg
