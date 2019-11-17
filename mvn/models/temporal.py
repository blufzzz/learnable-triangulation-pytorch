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


class Seq2VecModel(object):
    """docstring for Seq2VecModel"""
    def __init__(self, features_dim, style_vector_dim):
        super(Seq2VecModel, self).__init__()
        self.features_dim = features_dim
        self.style_vector_dim = style_vector_dim
        
        self.feature2vector = nn.Sequential()
        
        
        
        
        
        