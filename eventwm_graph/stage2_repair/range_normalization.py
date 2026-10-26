"""Guarded normalization: preserve the original rounded float ray directions.

Two double Newton refinements start from hardware float rsqrt. Before a
float conversion, values near a rounding midpoint use the original double
sqrt/division. This retains the direction/limit buffers consumed by tracing
while avoiding three full double divisions for every required probe.
"""
import torch
from .range_exact_fast import source as exact_source,PreparedModule
from .range_map_reuse import SupportReuse
from .range_fast import RangeFeatureKernel
from .camera_vectorized import parameters
from .range_batched_spatial import BatchedSpatialPredictor

HELPERS=r'''
__device__ bool near_float_midpoint(double value,float rounded,double error){
 unsigned bits=__float_as_uint(rounded);bool up=value>(double)rounded;
 unsigned adjacent=rounded==0.f?(up?1u:0x80000001u):bits+(((rounded>0.f)==up)?1u:0xffffffffu);
 double midpoint=((double)rounded+(double)__uint_as_float(adjacent))*.5;
 return fabs(value-midpoint)<=error;
}
__device__ double guarded_inverse_length(double squared){
 double inv=(double)rsqrtf((float)squared);
 inv=inv*(1.5-.5*squared*inv*inv);inv=inv*(1.5-.5*squared*inv*inv);return inv;
}
__device__ float rounded_direction(double numerator,double squared,double inverse){
 double q=numerator*inverse;float f=(float)q;double error=64.*2.2204460492503130808e-16*(fabs(q)+1.e-300);
 return near_float_midpoint(q,f,error)?(float)(numerator/sqrt(squared)):f;
}
__device__ float rounded_limit(double squared,double inverse){
 double length=squared*inverse,q=length-.015;float f=(float)q;double error=64.*2.2204460492503130808e-16*(length+.015);
 return near_float_midpoint(q,f,error)?(float)(sqrt(squared)-.015):f;
}
extern "C" __global__ void test_normalization(const double* vectors,int n,float* reference,float* fast){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const double* v=vectors+3*i;double s=v[0]*v[0]+v[1]*v[1]+v[2]*v[2],length=sqrt(s),inverse=guarded_inverse_length(s);
 for(int j=0;j<3;j++){reference[i*4+j]=v[j]/length;fast[i*4+j]=rounded_direction(v[j],s,inverse);}
 reference[i*4+3]=length-.015;fast[i*4+3]=rounded_limit(s,inverse);
}
'''


def source():
    original=exact_source();begin=original.index('extern "C" __global__ void generated_views');end=original.index('extern "C" __global__ void square_support');body=original[begin:end]
    old='length=sqrt(dx*dx+dy*dy+dz*dz);if(length<1.e-7)continue;float limit=(float)(length-.015);'
    new='squared=dx*dx+dy*dy+dz*dz;double inverse=guarded_inverse_length(squared);if(squared<1.e-14*(1.-64.*2.2204460492503130808e-16))continue;if(squared<=1.e-14*(1.+64.*2.2204460492503130808e-16)&&sqrt(squared)<1.e-7)continue;float limit=rounded_limit(squared,inverse);'
    if body.count(old)!=1:raise ValueError('reference normalization source changed')
    body=body.replace(old,new)
    for name in ['dx','dy','dz']:body=body.replace('(float)('+name+'/length)','rounded_direction('+name+',squared,inverse)')
    return original[:begin]+HELPERS+body+original[end:]


class NormalizedRangeKernel(SupportReuse,RangeFeatureKernel):
    camera_parameters=staticmethod(parameters)
    def __init__(self,device):self.device=torch.device(device);self.module=PreparedModule(source(),device)


class NormalizedSpatialPredictor(BatchedSpatialPredictor):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.range_kernel=NormalizedRangeKernel(self.device)
