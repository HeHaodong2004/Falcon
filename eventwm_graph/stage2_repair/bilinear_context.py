"""Sample learned context at each ORIGINAL cell's physical center.

The coarse map has one center per2x2original cells. Nearest indexing gives
all four cells identical learned context; bilinear sampling is a bounded
representation diagnostic. Targets and original candidate cells are unchanged.
"""
import torch
from torch.nn import functional as F
from .range_spatial import SpatialEvidenceNet
from .shared_sensor_api import SharedFastBuildPredictor


def sample(local,indices,mapids):
    _,channels,h,w=local.shape
    def one(field,ij):
        if not len(ij):return field.new_empty((0,channels),dtype=torch.float32)
        grid=torch.stack([(ij[:,1].float()+.5)/w-1.,(ij[:,0].float()+.5)/h-1.],-1)[None,None]
        values=F.grid_sample(field.float(),grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
        valid=(ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<2*h)&(ij[:,1]<2*w)
        return values*valid[:,None]
    if len(local)==1:return one(local,indices)
    output=torch.empty((len(indices),channels),device=local.device,dtype=torch.float32)
    for i in range(len(local)):
        take=mapids==i;output[take]=one(local[i:i+1],indices[take])
    return output


class BilinearSpatialNet(SpatialEvidenceNet):
    def raw(self,x,extra,indices,mapids,encoded):
        local,global_=encoded;context=sample(local,indices,mapids)
        with torch.set_grad_enabled(torch.is_grad_enabled()and self.train_range_head),torch.cuda.amp.autocast(enabled=False):r,b,height=self.point.raw(x.float(),extra.float())
        delta=self.head(torch.cat([x,extra,context,global_[mapids].float()],1));return r+delta[:,0],b+delta[:,1],height


class BilinearPredictor(SharedFastBuildPredictor):
    model_class=BilinearSpatialNet
    checkpoint_schema='legal_range_bilinear_1'
