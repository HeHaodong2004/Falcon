import math
import torch
from torch import nn
from eventwm_graph.stage2_hf.time_model import TimeNet


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__();self.point=TimeNet(train_occupancy=True);self.correction=nn.Sequential(nn.Linear(111,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,2))
        nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)
        for p in self.point.parameters():p.requires_grad_(False)
    def initialize(self,checkpoint):self.point.load_state_dict(torch.load(checkpoint,map_location='cpu')['model'])
    def raw(self,x,extra):
        with torch.no_grad():r,b,h=self.point.raw(x,torch.zeros((len(x),32),device=x.device))
        delta=self.correction(torch.cat([x,extra],1));return r+delta[:,0],b+delta[:,1],h
    def probabilities(self,x,extra):
        r,b,h=self.raw(x,extra);r=(r-math.log(8.)).sigmoid();b=b.sigmoid();return torch.stack([1-r,r*(1-b),r*b],1),h
