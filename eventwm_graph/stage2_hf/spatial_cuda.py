"""Batched CUDA implementation of the declared legal trajectory covariates.

Torch owns every buffer and stream. No labels are kernel inputs; no independent
CUDA context/allocator is created. CPU implementation remains the reference.
"""
import ctypes as ct
import hashlib,os,sys,tempfile
from pathlib import Path
import numpy as np
import torch
from eventwm_graph.stage2_segment.cuda_kernel import checked

SOURCE=r'''
extern "C" __global__ void query_features(const double* path,const double* pre,const double* globals,
 const long long* spec,const float* maps,int nt,int nx,int nz,float* output) {
 int cell=blockIdx.x*blockDim.x+threadIdx.x,batch=blockIdx.y,area=nx*nz;
 if(cell>=area)return;int i=cell/nz,k=cell%nz;const long long* s=spec+batch*5;
 int n=(int)s[0],hx=(int)s[3],hz=(int)s[4];if(i>=hx||k>=hz)return;
 const double* p=path+batch*nt*5;const double* pp=pre+batch*nt*4;const double* g=globals+batch*9;
 double resolution=g[0],duration=g[1],x=((double)s[1]+2*i+1)*resolution,z=((double)s[2]+2*k+1)*resolution;
 double mind=100.,best=-1.,maxw=0.,f5=0.,f8=0.,count=0.,first=1.,last=0.,soft=0.;
 for(int j=0;j<n;j++) {
  double dx=x-p[j*5+1],dz=z-p[j*5+3],d=sqrt(dx*dx+dz*dz);mind=fmin(mind,d);if(j==0)continue;
  double facing=fmax(-1.,fmin(1.,(dx*pp[j*4]-dz*pp[j*4+1])/fmax(d,1.e-4)));
  if(d<=10.) {
   best=fmax(best,facing);double w=fmax(0.,facing)*fmax(0.,1.-d/11.);maxw=fmax(maxw,w);soft+=w*pp[j*4+2];
   if(facing>=.5){f5+=pp[j*4+2];count+=1.;first=fmin(first,pp[j*4+3]);last=fmax(last,pp[j*4+3]);}
   if(facing>=.866025403784)f8+=pp[j*4+2];
  }
 }
 float* o=output+batch*32*area;
 o[0*area+cell]=(float)(mind/10.);o[1*area+cell]=(float)best;o[2*area+cell]=(float)maxw;
 o[3*area+cell]=(float)(f5/duration);o[4*area+cell]=(float)(f8/duration);o[5*area+cell]=(float)(count/(n>1?n-1:1));
 o[6*area+cell]=(float)first;o[7*area+cell]=(float)last;o[8*area+cell]=(float)(soft/duration);
 o[9*area+cell]=(float)((x-p[1])/10.);o[10*area+cell]=(float)((z-p[3])/10.);
 o[11*area+cell]=(float)((x-p[(n-1)*5+1])/10.);o[12*area+cell]=(float)((z-p[(n-1)*5+3])/10.);
 o[13*area+cell]=(float)(duration/5.);o[14*area+cell]=(float)(g[2]/5.);o[15*area+cell]=(float)g[3];
 o[16*area+cell]=(float)g[4];o[17*area+cell]=(float)g[5];o[18*area+cell]=(float)g[6];o[19*area+cell]=(float)g[7];
 const float* free=maps+batch*6*area;const float* blocked=free+area;
 for(int r=0;r<3;r++) {
  int j=r==0?(n>1?1:0):(r==1?n/2:n-1);double sx=p[j*5+1],sz=p[j*5+3],fr=0.,bl=0.,fu=1.,fb=1.;
  for(int step=1;step<=16;step++){
   double t=(double)step/17.;int a=(int)floor(((sx+(x-sx)*t)/resolution-(double)s[1])/2.);
   int b=(int)floor(((sz+(z-sz)*t)/resolution-(double)s[2])/2.);
   if(a>=0&&a<hx&&b>=0&&b<hz){double f=(double)free[a*nz+b],v=(double)blocked[a*nz+b];fr+=f;bl+=v;if(f+v<.999)fu=fmin(fu,t);if(v>0.)fb=fmin(fb,t);}
   else fu=fmin(fu,t);
  }
  o[(20+4*r)*area+cell]=(float)(fr/16.);o[(21+4*r)*area+cell]=(float)(bl/16.);
  o[(22+4*r)*area+cell]=(float)fu;o[(23+4*r)*area+cell]=(float)fb;
 }
}
'''

def compile_source(device):
 cap=torch.cuda.get_device_capability(device);options=[('--gpu-architecture=compute_%d%d'%cap).encode(),b'--fmad=false',b'--prec-div=true',b'--prec-sqrt=true',b'--std=c++11'];lib=ct.CDLL(str(Path(sys.prefix)/'lib/libnvrtc.so'));identity=hashlib.sha256(SOURCE.encode()+b'\0'.join(options)).hexdigest();cache=Path(os.environ['NUMBA_CACHE_DIR'])/'hf_cuda';cache.mkdir(parents=True,exist_ok=True);target=cache/(identity+'.ptx')
 if target.exists():return target.read_bytes(),identity
 program=ct.c_void_p();checked(lib,'nvrtcCreateProgram',[ct.POINTER(ct.c_void_p),ct.c_char_p,ct.c_char_p,ct.c_int,ct.c_void_p,ct.c_void_p],ct.byref(program),SOURCE.encode(),b'spatial_query.cu',0,None,None)
 try:
  args=(ct.c_char_p*len(options))(*options)
  try:checked(lib,'nvrtcCompileProgram',[ct.c_void_p,ct.c_int,ct.POINTER(ct.c_char_p)],program,len(options),args)
  except RuntimeError:
   size=ct.c_size_t();checked(lib,'nvrtcGetProgramLogSize',[ct.c_void_p,ct.POINTER(ct.c_size_t)],program,ct.byref(size));log=ct.create_string_buffer(size.value);checked(lib,'nvrtcGetProgramLog',[ct.c_void_p,ct.c_void_p],program,log);raise RuntimeError(log.value.decode())
  size=ct.c_size_t();checked(lib,'nvrtcGetPTXSize',[ct.c_void_p,ct.POINTER(ct.c_size_t)],program,ct.byref(size));buf=ct.create_string_buffer(size.value);checked(lib,'nvrtcGetPTX',[ct.c_void_p,ct.c_void_p],program,buf)
  with tempfile.NamedTemporaryFile(dir=cache,delete=False)as f:f.write(buf.raw);temporary=f.name
  os.replace(temporary,target);return buf.raw,identity
 finally:checked(lib,'nvrtcDestroyProgram',[ct.POINTER(ct.c_void_p)],ct.byref(program))

class QueryKernel:
 def __init__(self,device):
  self.device=torch.device(device);self.driver=ct.CDLL('libcuda.so.1')
  with torch.cuda.device(self.device):
   torch.empty(1,device=self.device)
   torch.cuda.synchronize(self.device)
   ptx,self.identity=compile_source(self.device);self.module=ct.c_void_p();buf=ct.create_string_buffer(ptx);checked(self.driver,'cuModuleLoadData',[ct.POINTER(ct.c_void_p),ct.c_void_p],ct.byref(self.module),buf);self.function=ct.c_void_p();checked(self.driver,'cuModuleGetFunction',[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p],ct.byref(self.function),self.module,b'query_features')
 def __call__(self,maps,path,pre,globals,spec):
  b,_,nx,nz=maps.shape;nt=path.shape[1];out=torch.zeros((b,32,nx,nz),dtype=torch.float32,device=self.device)
  values=[ct.c_void_p(t.data_ptr())for t in (path,pre,globals,spec,maps)]+[ct.c_int(v)for v in (nt,nx,nz)]+[ct.c_void_p(out.data_ptr())];args=(ct.c_void_p*len(values))(*[ct.addressof(v)for v in values])
  with torch.cuda.device(self.device):
   stream=ct.c_void_p(torch.cuda.current_stream(self.device).cuda_stream);checked(self.driver,'cuLaunchKernel',[ct.c_void_p,*([ct.c_uint]*7),ct.c_void_p,ct.POINTER(ct.c_void_p),ct.c_void_p],self.function,(nx*nz+127)//128,b,1,128,1,1,0,stream,args,None)
  return out

def path_arrays(paths,los,shapes,resolutions):
 n=len(paths);nt=max(map(len,paths));p=np.zeros((n,nt,5),np.float64);pre=np.zeros((n,nt,4),np.float64);g=np.zeros((n,9),np.float64);spec=np.zeros((n,5),np.int64)
 for i,(path,lo,shape,resolution)in enumerate(zip(paths,los,shapes,resolutions)):
  count=len(path);p[i,:count]=path;duration=max(path[-1,0]-path[0,0],1e-4);dt=np.r_[0,np.diff(path[:,0])];dy=np.diff(path[:,4]);dist=np.linalg.norm(np.diff(path[:,[1,3]],axis=0),axis=1)
  pre[i,:count]=np.column_stack([np.cos(path[:,4]),np.sin(path[:,4]),dt,(path[:,0]-path[0,0])/duration]);rotation=np.abs(np.arctan2(np.sin(dy),np.cos(dy))).sum();stationary=dt[1:][dist<1e-4].sum();delta=path[-1,4]-path[0,4]
  g[i]=[resolution,duration,dist.sum(),np.cos(delta),np.sin(delta),rotation/(2*np.pi),stationary/duration,path[-1,2]-path[0,2],0.];spec[i]=[count,*lo,shape[0]//2,shape[1]//2]
 return dict(path=p,pre=pre,globals=g,spec=spec)
