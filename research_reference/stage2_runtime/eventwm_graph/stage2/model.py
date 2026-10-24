"""Small shared cell classifier: observed UNKNOWN/FREE/BLOCKED and height."""
import torch
from torch import nn
from .features import FEATURES


class GraphDeltaNet(nn.Module):
    def __init__(self,width=48):
        super().__init__()
        self.width=width
        self.net=nn.Sequential(nn.Linear(FEATURES,width),nn.SiLU(),nn.Linear(width,32),nn.SiLU(),nn.Linear(32,4))

    def forward(self,x):
        out=self.net(x)
        return out[:,:3],out[:,3]
