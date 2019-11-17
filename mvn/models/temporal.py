import torch.nn as nn
import torch.nn.functional as F

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


class Seq2VecModel(nn.Module):
    """docstring for Seq2VecModel"""
    def __init__(self, features_dim, hidden_dim = 512):
        super(Seq2VecModel, self).__init__()
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
        
    def forward(self, features):
        # [batch size, dt, 256, 96, 96]

        features_shape = features.shape[2:]
        batch_size, dt = features.shape[:2]
        vectors = self.feature2vector(features.view(-1, *features_shape))
        vectors = vectors.view(batch_size, dt, -1)
        (h0, c0) = torch.randn(1, batch_size, self.hidden_dim), torch.randn(1, batch_size, self.hidden_dim)
        output, (hn, cn) = self.lstm(vectors, (h0, c0))
        return output[:,-1,...]
        
        
        
        
        
        
        
        