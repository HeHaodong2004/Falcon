import torch
from torch import nn
from .common import FEATURES
class GraphDeltaNet(nn.Module):
    def __init__(self,width=48,input_features=FEATURES):
        super().__init__();self.net=nn.Sequential(nn.Linear(input_features,width),nn.SiLU(),nn.Linear(width,32),nn.SiLU(),nn.Linear(32,4))
    def forward(self,x):
        y=self.net(x);return y[:,:3],y[:,3]
