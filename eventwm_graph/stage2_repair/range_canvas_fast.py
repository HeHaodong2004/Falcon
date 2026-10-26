"""Identical measured-context channels, avoiding unnecessary range square roots.

Later frames remain eligible until both binary observations are established.
This does not discard stationary later observations or change label targets.
"""
import torch
from .range_bev import RangeCanvas,SOURCE
from .cuda_module import CudaModule


class FastRangeCanvas(RangeCanvas):
    def __init__(self,device='cuda:0'):
        old='double distance=sqrt(dx*dx+dy*dy+dz*dz);if(forward<=1.e-6||fabs(a)>forward||fabs(b)>forward||distance>10.)continue;'
        new='if(forward<=1.e-6||fabs(a)>forward||fabs(b)>forward||fabs(dx)>10.01||fabs(dy)>10.01||fabs(dz)>10.01)continue;double distance=sqrt(dx*dx+dy*dy+dz*dz);if(distance>10.)continue;'
        if SOURCE.count(old)!=1:raise ValueError('measured context source changed')
        updated=SOURCE.replace(old,new)
        old_hit='if(!person)hit=true;}'
        if updated.count(old_hit)!=1:raise ValueError('measured hit source changed')
        updated=updated.replace(old_hit,old_hit+'if(clear&&hit)break;')
        self.device=torch.device(device);self.module=CudaModule(updated,device)
