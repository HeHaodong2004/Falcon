"""Exact63-probe evidence with shared constants and early shadow rejection.

Frustum membership of the square/halo prism is tested using its supporting
planes and discrete16-direction range maximum. Numerical boundary cases use
the original63individual checks. Every LOS probe remains required; testing a
far mid-height halo probe first changes only short-circuit order.
"""
import ctypes as ct
import torch
from .range_fast import RangeFeatureKernel,SOURCE
from .range_map_reuse import SupportReuse
from .camera_vectorized import parameters
from .cuda_module import CudaModule

HELPERS=r'''
extern "C" __global__ void prepare_certificate(const double* cameras,int nv,double resolution,double* planes,double* offsets){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nv)return;
 double radius=.25+resolution/sqrt(2.);
 if(i==0)for(int j=0;j<63;j++){
  int at=j%21;double x=0.,z=0.;
  if(at>=5){double angle=(at-5)*6.2831853071795864769/16.;x=cos(angle)*radius;z=sin(angle)*radius;}
  else if(at){x=(at<=2?-1.:1.)*.49*resolution;z=(at==1||at==3?-1.:1.)*.49*resolution;}
  offsets[j*3]=x;offsets[j*3+1]=j/21==0?.025:j/21==1?.7:1.4;offsets[j*3+2]=z;
 }
 const double* c=cameras+i*13;
 for(int p=0;p<5;p++){
  double n[3];for(int a=0;a<3;a++)n[a]=-c[3+a*3+2]+(p==1?c[3+a*3]:p==2?-c[3+a*3]:p==3?c[3+a*3+1]:p==4?-c[3+a*3+1]:0.);
  double smallest=1.e30;
  for(int at=0;at<16;at++){double angle=at*6.2831853071795864769/16.;double x=cos(angle)*radius,z=sin(angle)*radius;smallest=fmin(smallest,n[0]*x+n[2]*z);}
  double* o=planes+i*20+p*4;o[0]=n[0];o[1]=n[1];o[2]=n[2];o[3]=smallest+fmin(n[1]*.025,n[1]*1.4);
 }
}
__device__ bool certificate_frustum(double x,double y,double z,const double* c,const double* planes,const double* offsets,double radius){
 double dx=x-c[0],dy=y-c[1],dz=z-c[2];
 double error=64.*2.2204460492503130808e-16*(fabs(x)+fabs(y)+fabs(z)+fabs(c[0])+fabs(c[1])+fabs(c[2])+20.);bool boundary=false;
 for(int p=0;p<5;p++){
  const double* q=planes+p*4;double value=dx*q[0]+dy*q[1]+dz*q[2]+q[3]-(p==0?1.e-6:0.);
  double e=error*(1.+fabs(q[0])+fabs(q[1])+fabs(q[2]));
  if(value < -e)return false;if(fabs(value)<=e)boundary=true;
 }
 double a=fabs(dx),b=fabs(dz),ca=.9238795325112867561,sa=.3826834323650897717;
 double dot=fmax(fmax(a,b),fmax(fmax(ca*a+sa*b,sa*a+ca*b),.7071067811865475244*(a+b)));
 double vertical=fmax(fabs(dy+.025),fabs(dy+1.4));
 double maximum=dx*dx+dz*dz+radius*radius+2.*radius*dot+vertical*vertical;
 double e=error*(fabs(dx)+fabs(dy)+fabs(dz)+radius+3.)*8.;
 if(maximum>100.+e)return false;if(fabs(maximum-100.)<=e)boundary=true;
 if(boundary)for(int j=0;j<63;j++)if(!frustum(x+offsets[j*3],y+offsets[j*3+1],z+offsets[j*3+2],c))return false;
 return true;
}
extern "C" __global__ void test_certificate(const double* points,const int* ids,int n,const double* cameras,const double* planes,const double* offsets,double resolution,unsigned char* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const double* p=points+3*i;int id=ids[i];const double* c=cameras+id*13;bool old=true;
 for(int j=0;j<63;j++)if(!frustum(p[0]+offsets[j*3],p[1]+offsets[j*3+1],p[2]+offsets[j*3+2],c)){old=false;break;}
 out[2*i]=old;out[2*i+1]=certificate_frustum(p[0],p[1],p[2],c,planes+id*20,offsets,.25+resolution/sqrt(2.));
}
'''


def source(ordered=True,closed_form=True):
    begin=SOURCE.index('extern "C" __global__ void generated_views')
    end=SOURCE.index('extern "C" __global__ void square_support')
    body=SOURCE[begin:end]
    header='const unsigned char* unknown'
    assert body.count(header)==1
    body=body.replace(header,'const double* certificate_planes,const double* probe_offsets,'+header)
    first=body.index(' for(int j=0;j<63;j++){')
    last=body.index(' MESH_VALUE;',first)
    original=body[first:last]
    if closed_form:
        body=body[:first]+' if(!certificate_frustum(x,y,z,c,certificate_planes+cam*20,probe_offsets,radius))return;\n'+body[last:]
    else:
        body=body[:first]+' for(int j=0;j<63;j++)if(!frustum(x+probe_offsets[j*3],y+probe_offsets[j*3+1],z+probe_offsets[j*3+2],c))return;\n'+body[last:]
    target='for(int j=0;j<63;j++){\n  int at=j%21;double ax=0.,az=0.;'
    if body.count(target)!=1:raise ValueError('LOS source changed')
    start=body.index(target);stop=body.index('  double dx=x+ax-c[0]',start)
    loop=('double rx=x-c[0],rz=z-c[2];int far=26+(fabs(rx)>=fabs(rz)?(rx>=0?0:8):(rz>=0?4:12));\n for(int pass=0;pass<64;pass++){int j=pass==0?far:pass-1;if(pass>0&&j==far)continue;\n'if ordered else'for(int j=0;j<63;j++){\n')
    body=body[:start]+loop+'  double ax=probe_offsets[j*3],az=probe_offsets[j*3+2];\n'+body[stop:]
    body=body.replace('dy=y+(j/21==0?.025:j/21==1?.7:1.4)-c[1]','dy=y+probe_offsets[j*3+1]-c[1]')
    return SOURCE[:begin]+HELPERS+body+SOURCE[end:]


class PreparedModule(CudaModule):
    def launch(self,name,total,values,threads=128):
        if name=='generated_views':
            nv=values[4].value
            planes=torch.empty((nv,20),device=self.device,dtype=torch.float64)
            offsets=torch.empty((63,3),device=self.device,dtype=torch.float64)
            super().launch('prepare_certificate',nv,[values[3],values[4],values[10],planes,offsets])
            self.last_constants=(planes,offsets)
            values=[*values[:5],planes,offsets,*values[5:]]
        return super().launch(name,total,values,threads)


class FastExactKernel(SupportReuse,RangeFeatureKernel):
    camera_parameters=staticmethod(parameters)
    def __init__(self,device,ordered=True,closed_form=True):
        self.device=torch.device(device);self.module=PreparedModule(source(ordered,closed_form),device)
