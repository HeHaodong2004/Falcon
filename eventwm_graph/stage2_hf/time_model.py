import math
import torch
from torch import nn
from torch.nn import functional as F
from .common import CKPT
class TimeNet(nn.Module):
 def __init__(self,train_occupancy=False,width=64):
  super().__init__();self.train_occupancy=train_occupancy;self.prior=nn.Sequential(nn.Linear(88,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,4));self.correction=nn.Sequential(nn.Linear(120,width),nn.SiLU(),nn.Linear(width,32),nn.SiLU(),nn.Linear(32,2))
  for p in self.prior.parameters():p.requires_grad_(False)
 def initialize(self):
  old=torch.load(CKPT,map_location='cpu')['model']
  for i in (0,2,4):self.prior[i].weight.data.copy_(old['net.%d.weight'%i][:,16:]if i==0 else old['net.%d.weight'%i]);self.prior[i].bias.data.copy_(old['net.%d.bias'%i])
  nn.init.zeros_(self.correction[4].weight);nn.init.zeros_(self.correction[4].bias);self.correction[4].bias.data[0]=math.log(8.)
 def raw(self,x,extra=None,basis=None,projection=None):
  with torch.no_grad():base=self.prior(x)
  if extra is not None:delta=self.correction(torch.cat([x,extra],-1))
  else:
   h=F.linear(x,self.correction[0].weight[:,:88],self.correction[0].bias)+basis@projection;h=self.correction[1](h);h=self.correction[2](h);h=self.correction[3](h);delta=self.correction[4](h)
  r=base[:,1:3].logsumexp(-1)-base[:,0]+delta[:,0];b=base[:,2]-base[:,1]+(delta[:,1]if self.train_occupancy else 0)
  return r,b,base[:,3]
 def probabilities(self,x,extra=None,basis=None,projection=None,calibration=None):
  r,b,h=self.raw(x,extra,basis,projection);r=r-math.log(8.)
  if calibration:r=r*calibration['rev_scale']+calibration['rev_bias'];b=b*calibration['blocked_scale']+calibration['blocked_bias']
  r=r.sigmoid();b=b.sigmoid();return torch.stack([1-r,r*(1-b),r*b],-1),h
