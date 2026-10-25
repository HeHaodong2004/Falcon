"""One exact-domain CUDA feature kernel, compiled with the installed NVRTC.

The CUDA driver receives only owned Torch device buffers and the current Torch
stream. No independent CUDA context or allocator is created. CPU boundary
fallbacks remain in cuda.py; compilation is part of reported cold startup.
"""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import torch


SOURCE = r'''
extern "C" __global__ void features(
 const int* keys, const double* path, const double* pre, const double* globals,
 const unsigned char* grid, int n, int t, int nx, int nz, int lx, int lz,
 double resolution, double boundary, int end_only, float* view, float* lines,
 unsigned char* view_boundary, unsigned char* line_boundary) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n)return;
 int kx=keys[2*i],kz=keys[2*i+1];
 float x=__fmul_rn(__fadd_rn((float)kx,.5f),(float)resolution);
 float z=__fmul_rn(__fadd_rn((float)kz,.5f),(float)resolution);
 double mind=1.e30;
 for(int j=0;j<t;j++) {
  float dx=__fsub_rn(x,(float)path[j*5+1]);
  float dz=__fsub_rn(z,(float)path[j*5+3]);
  float d=__fadd_rn(__fmul_rn(dx,dx),__fmul_rn(dz,dz));
  mind=fmin(mind,(double)d);
 }
 double best=-1.,weighted=0.,min0=11.,min5=11.,min8=11.;
 double first=1.,last=0.,sum=0.,maxall=-1.;
 double c0=0.,c5=0.,c8=0.,t0=0.,t5=0.,t8=0.;
 double den=fmax(path[(t-1)*5]-path[0],1.e-4);int critical=0;
 for(int j=1;j<t;j++) {
  double dx=(double)x-path[j*5+1],dz=(double)z-path[j*5+3];
  double distance=sqrt(dx*dx+dz*dz);
  double facing=fmax(-1.,fmin(1.,(dx*pre[j*4]-dz*pre[j*4+1])/fmax(distance,1.e-4)));
  if(fabs(distance-10.)<=1.e-8||fabs(facing)<=1.e-8||fabs(facing-.5)<=1.e-8||fabs(facing-.866025403784)<=1.e-8)critical=1;
  maxall=fmax(maxall,facing);double w=fmax(0.,1.-distance/11.);
  double product=fmax(0.,facing)*w;sum+=product;weighted=fmax(weighted,product);
  if(distance<=10.) {
   best=fmax(best,facing);double dt=pre[j*4+2];
   if(facing>=0.){c0+=1.;t0+=dt;min0=fmin(min0,distance);}
   if(facing>=.5){c5+=1.;t5+=dt;min5=fmin(min5,distance);first=fmin(first,pre[j*4+3]);last=fmax(last,pre[j*4+3]);}
   if(facing>=.866025403784){c8+=1.;t8+=dt;min8=fmin(min8,distance);}
  }
 }
 double count=(double)(t>1?t-1:1);float* v=view+i*20;
 v[0]=(float)(sqrt(mind)/10.);v[1]=(float)best;v[2]=(float)weighted;
 v[3]=(float)(min0/10.);v[4]=(float)(min5/10.);v[5]=(float)(min8/10.);
 v[6]=(float)(c0/count);v[7]=(float)(c5/count);v[8]=(float)(c8/count);
 v[9]=(float)(t0/den);v[10]=(float)(t5/den);v[11]=(float)(t8/den);
 v[12]=(float)first;v[13]=(float)last;v[14]=(float)maxall;v[15]=(float)(sum/count);
 for(int j=0;j<4;j++)v[16+j]=(float)globals[j];view_boundary[i]=(unsigned char)critical;
 double tx=((double)kx+.5)*resolution,tz=((double)kz+.5)*resolution;
 int rays=end_only?1:3,width=rays*4;critical=0;
 for(int ray=0;ray<rays;ray++) {
  int j=end_only?t-1:(ray==0?(t>1?1:0):(ray==1?t/2:t-1));
  double sx=path[j*5+1],sz=path[j*5+3];
  int free_count=0,blocked_count=0;double fu=1.,fb=1.;
  for(int step=1;step<=16;step++) {
   double f=(double)step/17.;
   double px=(sx+(tx-sx)*f)/resolution,pz=(sz+(tz-sz)*f)/resolution;
   if(fabs(px-round(px))<=boundary||fabs(pz-round(pz))<=boundary)critical=1;
   long long ix=(long long)floor(px)-(long long)lx,iz=(long long)floor(pz)-(long long)lz;
   int state=(ix>=0&&ix<nx&&iz>=0&&iz<nz)?grid[ix*nz+iz]:0;
   if(state==1)free_count++;else if(state==2){blocked_count++;fb=fmin(fb,f);}else fu=fmin(fu,f);
  }
  float* out=lines+i*width+ray*4;
  out[0]=(float)((double)free_count/16.);out[1]=(float)((double)blocked_count/16.);out[2]=(float)fu;out[3]=(float)fb;
 }
 line_boundary[i]=(unsigned char)critical;
}

__device__ int clamp_index(int x,int maximum){return x<0?0:(x>maximum?maximum:x);}
__device__ double box(const long long* image,int stride,int a,int b,int c,int d) {
 return (double)(image[b*stride+d]-image[a*stride+d]-image[b*stride+c]+image[a*stride+c]);
}
extern "C" __global__ void assemble(
 const int* keys,const float* view,const float* lines,const unsigned char* grid,
 const float* heights,const float* d_free,const float* d_blocked,
 const long long* free_image,const long long* blocked_image,const float* constants,
 int n,int cx,int cz,int lx,int lz,int mx,int mz,int mlx,int mlz,int width,
 float resolution,float x0,float z0,float x1,float z1,float robot_height,
 float* output,float* baseline) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int kx=keys[2*i],kz=keys[2*i+1],x=kx-lx,z=kz-lz,col=0;
 float* out=output+i*(76+width);
 for(int dx=-3;dx<=3;dx+=3)for(int dz=-3;dz<=3;dz+=3){
  int a=clamp_index(x+dx,cx-1),b=clamp_index(z+dz,cz-1);unsigned char state=grid[a*cz+b];
  out[col]=(float)(state==1);out[col+9]=(float)(state==2);col++;
 }
 out[18]=d_free[x*cz+z]/10.f;out[19]=d_blocked[x*cz+z]/10.f;
 float px=__fmul_rn(__fadd_rn((float)kx,.5f),resolution),pz=__fmul_rn(__fadd_rn((float)kz,.5f),resolution);
 out[20]=__fsub_rn(px,x0)/10.f;out[21]=__fsub_rn(pz,z0)/10.f;
 out[22]=__fsub_rn(px,x1)/10.f;out[23]=__fsub_rn(pz,z1)/10.f;
 float height=heights[x*cz+z];baseline[i]=height;out[24]=__fsub_rn(height,robot_height);
 for(int j=0;j<6;j++)out[25+j]=constants[j];
 for(int j=0;j<20;j++)out[31+j]=view[i*20+j];
 x=kx-mlx;z=kz-mlz;col=51;
 for(int rindex=0;rindex<3;rindex++){
  int r=rindex==0?1:(rindex==1?4:12);
  int a=clamp_index(x-r,mx),b=clamp_index(x+r+1,mx),c=clamp_index(z-r,mz),d=clamp_index(z+r+1,mz);
  double area=(double)((2*r+1)*(2*r+1));double f=box(free_image,mz+1,a,b,c,d)/area,h=box(blocked_image,mz+1,a,b,c,d)/area;
  out[col]=(float)f;out[col+1]=(float)h;out[col+2]=(float)fmax(0.,1.-f-h);col+=3;
 }
 for(int rindex=0;rindex<2;rindex++){
  int r=rindex==0?4:12;
  for(int sx=-1;sx<=1;sx+=2)for(int sz=-1;sz<=1;sz+=2){
   int xa=sx<0?x-r:x,xb=sx<0?x+1:x+r+1,za=sz<0?z-r:z,zb=sz<0?z+1:z+r+1;
   int a=clamp_index(xa,mx),b=clamp_index(xb,mx),c=clamp_index(za,mz),d=clamp_index(zb,mz);
   double area=(double)((r+1)*(r+1));out[col]=(float)(box(free_image,mz+1,a,b,c,d)/area);
   out[col+1]=(float)(box(blocked_image,mz+1,a,b,c,d)/area);col+=2;
  }
 }
 for(int j=0;j<width;j++)out[76+j]=lines[i*width+j];
}
'''


def checked(library,name,types,*args):
    function=getattr(library,name);function.argtypes=types;function.restype=ct.c_int
    code=function(*args)
    if code:raise RuntimeError('%s returned CUDA status%d'%(name,code))


def compile_ptx(capability):
    options=[('--gpu-architecture=compute_%d%d'%capability).encode(),b'--fmad=false',b'--prec-div=true',b'--prec-sqrt=true',b'--std=c++11']
    compiler=ct.CDLL(str(Path(sys.prefix)/'lib/libnvrtc.so'))
    major=ct.c_int();minor=ct.c_int();checked(compiler,'nvrtcVersion',[ct.POINTER(ct.c_int),ct.POINTER(ct.c_int)],ct.byref(major),ct.byref(minor))
    identity=hashlib.sha256(SOURCE.encode()+b'\0'.join(options)+('%d.%d'%(major.value,minor.value)).encode()).hexdigest()
    cache=Path(os.environ.get('NUMBA_CACHE_DIR',str(Path(tempfile.gettempdir())/'stage2-jit')))/'cuda';cache.mkdir(parents=True,exist_ok=True)
    target=cache/(identity+'.ptx')
    if target.exists():return target.read_bytes(),identity
    program=ct.c_void_p();checked(compiler,'nvrtcCreateProgram',[ct.POINTER(ct.c_void_p),ct.c_char_p,ct.c_char_p,ct.c_int,ct.c_void_p,ct.c_void_p],ct.byref(program),SOURCE.encode(),b'segment_features.cu',0,None,None)
    try:
        arguments=(ct.c_char_p*len(options))(*options)
        try:checked(compiler,'nvrtcCompileProgram',[ct.c_void_p,ct.c_int,ct.POINTER(ct.c_char_p)],program,len(options),arguments)
        except RuntimeError:
            size=ct.c_size_t();checked(compiler,'nvrtcGetProgramLogSize',[ct.c_void_p,ct.POINTER(ct.c_size_t)],program,ct.byref(size))
            log=ct.create_string_buffer(size.value);checked(compiler,'nvrtcGetProgramLog',[ct.c_void_p,ct.c_void_p],program,log)
            raise RuntimeError(log.value.decode())
        size=ct.c_size_t();checked(compiler,'nvrtcGetPTXSize',[ct.c_void_p,ct.POINTER(ct.c_size_t)],program,ct.byref(size))
        ptx=ct.create_string_buffer(size.value);checked(compiler,'nvrtcGetPTX',[ct.c_void_p,ct.c_void_p],program,ptx)
        with tempfile.NamedTemporaryFile(dir=cache,delete=False) as f:f.write(ptx.raw);temporary=f.name
        os.replace(temporary,target)
        return ptx.raw,identity
    finally:checked(compiler,'nvrtcDestroyProgram',[ct.POINTER(ct.c_void_p)],ct.byref(program))


class FeatureKernel:
    def __init__(self,device):
        self.device=device;self.driver=ct.CDLL('libcuda.so.1')
        with torch.cuda.device(device):
            ptx,self.identity=compile_ptx(torch.cuda.get_device_capability(device));self.module=ct.c_void_p()
            buffer=ct.create_string_buffer(ptx)
            checked(self.driver,'cuModuleLoadData',[ct.POINTER(ct.c_void_p),ct.c_void_p],ct.byref(self.module),buffer)
            self.function=ct.c_void_p();checked(self.driver,'cuModuleGetFunction',[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p],ct.byref(self.function),self.module,b'features')
            self.assemble=ct.c_void_p();checked(self.driver,'cuModuleGetFunction',[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p],ct.byref(self.assemble),self.module,b'assemble')

    def launch(self,buffers,scalars):
        inputs=buffers[:5];outputs=buffers[5:]
        values=[ct.c_void_p(t.data_ptr()) for t in inputs]+[ct.c_int(x) for x in scalars[:6]]+[ct.c_double(x) for x in scalars[6:8]]+[ct.c_int(scalars[8])]+[ct.c_void_p(t.data_ptr()) for t in outputs]
        arguments=(ct.c_void_p*len(values))(*[ct.addressof(v) for v in values]);n=scalars[0]
        with torch.cuda.device(self.device):
            stream=ct.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
            checked(self.driver,'cuLaunchKernel',[ct.c_void_p,*([ct.c_uint]*7),ct.c_void_p,ct.POINTER(ct.c_void_p),ct.c_void_p],self.function,(n+127)//128,1,1,128,1,1,0,stream,arguments,None)

    def launch_assembly(self,buffers,integers,floats):
        values=[ct.c_void_p(t.data_ptr()) for t in buffers[:10]]+[ct.c_int(x) for x in integers]+[ct.c_float(x) for x in floats]+[ct.c_void_p(t.data_ptr()) for t in buffers[10:]]
        arguments=(ct.c_void_p*len(values))(*[ct.addressof(v) for v in values]);n=integers[0]
        with torch.cuda.device(self.device):
            stream=ct.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
            checked(self.driver,'cuLaunchKernel',[ct.c_void_p,*([ct.c_uint]*7),ct.c_void_p,ct.POINTER(ct.c_void_p),ct.c_void_p],self.assemble,(n+127)//128,1,1,128,1,1,0,stream,arguments,None)
