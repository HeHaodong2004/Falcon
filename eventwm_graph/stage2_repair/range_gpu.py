"""Array-only observed-range mesh and balanced GPU BVH construction.

No simulator, private asset path, support bank or label is an input. The only
surface triangles come from already measured range pixels. The numerical
approximation relative to the CPU prototype is checked on matching examples.
"""
import ctypes as ct
import time
import numpy as np
import torch
from .cuda_module import CudaModule
from .geometry_cuda import GeometryKernel
from .observed_range_mesh import ObservedRangeMesh
from .range_history import validate

SOURCE=r'''
extern "C" __global__ void measured_triangles(const float* ranges,const double* origins,const double* rotations,
 const double* tracks,const int* offsets,int height,int width,int stride,int per,int n,
 float* triangles,long long* keys,const int* low){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int frame=i/per,q=(i%per)%(per/2);
 int w=(width-1)/stride,h=(height-1)/stride,row=q/w,col=q%w;
 int uu[3],vv[3];
 if(i%per<per/2){uu[0]=col;vv[0]=row;uu[1]=col;vv[1]=row+1;uu[2]=col+1;vv[2]=row;}
 else{uu[0]=col+1;vv[0]=row+1;uu[1]=col+1;vv[1]=row;uu[2]=col;vv[2]=row+1;}
 float v[9];keys[i]=9223372036854775807LL;
 for(int p=0;p<3;p++){
  int u=uu[p]*stride,y=vv[p]*stride;float d=ranges[(frame*height+y)*width+u];if(!(d>.03f&&d<=10.00001f))return;
  double x=(u+.5)*2/width-1.,a=1.-(y+.5)*2/height,z=-1.,len=sqrt(x*x+a*a+1.);x/=len;a/=len;z/=len;
  const double* r=rotations+frame*9;const double* o=origins+frame*3;
  double wx=o[0]+d*(r[0]*x+r[1]*a+r[2]*z),wy=o[1]+d*(r[3]*x+r[4]*a+r[5]*z),wz=o[2]+d*(r[6]*x+r[7]*a+r[8]*z);
  for(int j=offsets[frame];j<offsets[frame+1];j++)if((wx-tracks[j*3])*(wx-tracks[j*3])+(wz-tracks[j*3+2])*(wz-tracks[j*3+2])<.25&&wy>tracks[j*3+1]-.05&&wy<tracks[j*3+1]+2.)return;
  v[p*3]=wx;v[p*3+1]=wy;v[p*3+2]=wz;
 }
 float e[3],f[3],g[3];float ee=0,ff=0,gg=0;for(int j=0;j<3;j++){e[j]=v[3+j]-v[j];f[j]=v[6+j]-v[j];g[j]=v[6+j]-v[3+j];ee+=e[j]*e[j];ff+=f[j]*f[j];gg+=g[j]*g[j];}
 float edge_limit_squared=.09f*stride*stride;
 if(ee>=edge_limit_squared||ff>=edge_limit_squared||gg>=edge_limit_squared)return;
 float nx=e[1]*f[2]-e[2]*f[1],ny=e[2]*f[0]-e[0]*f[2],nz=e[0]*f[1]-e[1]*f[0],area=sqrtf(nx*nx+ny*ny+nz*nz);if(area<=1.e-8f)return;
 unsigned long long code=0;
 for(int j=0;j<3;j++){int c=(int)floorf(((v[j]+v[j+3]+v[j+6])/3.f)/.035f)-low[j];if(c<0||c>65535)return;code|=(unsigned long long)c<<(j*16);}
 code|=(unsigned long long)((int)llrintf(nx/area/.2f)+5)<<48;
 code|=(unsigned long long)((int)llrintf(ny/area/.2f)+5)<<52;
 code|=(unsigned long long)((int)llrintf(nz/area/.2f)+5)<<56;
 keys[i]=(long long)code;for(int j=0;j<9;j++)triangles[i*9+j]=v[j];
}
__device__ unsigned int spread(unsigned int x){x&=1023;x=(x|(x<<16))&0x030000FF;x=(x|(x<<8))&0x0300F00F;x=(x|(x<<4))&0x030C30C3;x=(x|(x<<2))&0x09249249;return x;}
extern "C" __global__ void morton(const float* tri,int n,const float* low,const float* extent,long long* keys){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;unsigned int q[3];
 for(int j=0;j<3;j++){float p=(tri[i*9+j]+tri[i*9+j+3]+tri[i*9+j+6])/3.f;q[j]=min(1023,max(0,(int)((p-low[j])/extent[j]*1023)));}
 keys[i]=spread(q[0])|(spread(q[1])<<1)|(spread(q[2])<<2);
}
extern "C" __global__ void leaves(const float* vertices,int n,int group,int leaves,float* bounds,int* left,int* right,int* first,int* count,float* packed,int* indices){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=leaves)return;int node=leaves-1+i,begin=i*group,end=min(begin+group,n);first[node]=begin;count[node]=begin>=n?-1:end-begin;left[node]=right[node]=-1;
 for(int j=0;j<3;j++){float lo=1.e30f,hi=-1.e30f;for(int k=begin;k<end;k++)for(int v=0;v<3;v++){float x=vertices[k*9+v*3+j];lo=fminf(lo,x);hi=fmaxf(hi,x);}bounds[node*6+j]=lo;bounds[node*6+j+3]=hi;}
 for(int k=begin;k<end;k++){indices[k]=k;for(int j=0;j<3;j++){packed[k*9+j]=vertices[k*9+j];packed[k*9+3+j]=vertices[k*9+3+j]-vertices[k*9+j];packed[k*9+6+j]=vertices[k*9+6+j]-vertices[k*9+j];}}
}
extern "C" __global__ void parents(int begin,int n,float* bounds,int* left,int* right,int* first,int* count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int node=begin+i,a=node*2+1,b=a+1;left[node]=a;right[node]=b;first[node]=0;count[node]=0;
 for(int j=0;j<3;j++){bounds[node*6+j]=fminf(bounds[a*6+j],bounds[b*6+j]);bounds[node*6+j+3]=fmaxf(bounds[a*6+j+3],bounds[b*6+j+3]);}
}
extern "C" __global__ void observed_obstacles(const float* tri,int n,float low_y,float high_y,float r,int lx,int lz,int nx,int nz,int* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const float* t=tri+9*i;float x[3]={t[0],t[0]+t[3],t[0]+t[6]},y[3]={t[1],t[1]+t[4],t[1]+t[7]},z[3]={t[2],t[2]+t[5],t[2]+t[8]};
 if(fmaxf(y[0],fmaxf(y[1],y[2]))<=low_y||fminf(y[0],fminf(y[1],y[2]))>=high_y)return;
 int ax=max(0,(int)floorf(fminf(x[0],fminf(x[1],x[2]))/r)-lx),bx=min(nx-1,(int)floorf(fmaxf(x[0],fmaxf(x[1],x[2]))/r)-lx);
 int az=max(0,(int)floorf(fminf(z[0],fminf(z[1],z[2]))/r)-lz),bz=min(nz-1,(int)floorf(fmaxf(z[0],fmaxf(z[1],z[2]))/r)-lz);
 for(int a=ax;a<=bx;a++)for(int b=az;b<=bz;b++)atomicExch(out+a*nz+b,1);
}
'''


class GpuObservedRangeMesh(ObservedRangeMesh):
    def __init__(self,ranges,origins,rotations,timestamps,now,resolution,device='cuda:0',stride=1,tracks=None,plane_extension=0.,free_keys=None,legacy_dedup=False):
        if plane_extension:raise ValueError('GPU observed-only constructor excludes unvalidated plane completion')
        tick=time.perf_counter()
        validate(ranges,origins,rotations,timestamps,now)
        n,h,w=ranges.shape
        if len(origins)!=n or len(rotations)!=n or len(timestamps)!=n:raise ValueError('range history alignment')
        selected=[];last=None
        for i in range(n):
            if last is not None and i!=n-1 and np.linalg.norm(origins[i]-origins[last])<.07 and np.max(abs(rotations[i]-rotations[last]))<.07:continue
            selected.append(i);last=i
        self.keyframes=selected;self.planes=[];self.latest_sensor_time=float(timestamps[-1]);device=torch.device(device)
        mod=CudaModule(SOURCE,device)
        def tensor(x,dtype=None):return torch.from_numpy(np.ascontiguousarray(x,dtype=dtype)).to(device)
        active=[];offsets=[0]
        for i in selected:
            for track in tracks[i]if tracks is not None else[]:
                if not track.get('valid',True):continue
                pos=np.asarray(track['position'],np.float64)
                if track.get('velocity_valid')and track.get('estimated_velocity')is not None:pos=pos+np.asarray(track['estimated_velocity'])*min(float(track.get('observation_age',0)),.5)
                active.append(pos)
            offsets.append(len(active))
        per=((h-1)//stride)*((w-1)//stride)*2;total=len(selected)*per
        triangles=torch.empty((total,9),device=device);keys=torch.empty(total,device=device,dtype=torch.int64)
        lower=np.floor((np.asarray(origins)[selected].min(0)-10.01)/.035).astype(np.int32)
        mod.launch('measured_triangles',total,[tensor(ranges[selected],np.float32),tensor(origins[selected],np.float64),tensor(rotations[selected],np.float64),tensor(np.array(active).reshape(-1,3),np.float64),tensor(offsets,np.int32),ct.c_int(h),ct.c_int(w),ct.c_int(stride),ct.c_int(per),ct.c_int(total),triangles,keys,tensor(lower)])
        if legacy_dedup:
            values,order=torch.sort(keys,stable=True)
            keep=torch.ones_like(values,dtype=torch.bool);keep[1:]=values[1:]!=values[:-1];keep&=values!=torch.iinfo(torch.int64).max
            triangles=triangles[order[keep]]
        else:triangles=triangles[keys!=torch.iinfo(torch.int64).max]
        self.triangles=int(len(triangles))
        if not self.triangles:raise ValueError('no observed range surface')
        vertices=triangles.view(-1,3,3);low=vertices.amin((0,1));extent=(vertices.amax((0,1))-low).clamp_min(.001)
        code=torch.empty(self.triangles,device=device,dtype=torch.int64)
        mod.launch('morton',self.triangles,[triangles,ct.c_int(self.triangles),low,extent,code])
        triangles=triangles[torch.argsort(code)];group=12;leaves=1<<((self.triangles+group-1)//group-1).bit_length();nodes=leaves*2-1
        arrays=dict(keys=np.empty((0,2),np.int32),targets=np.empty((0,63,3),np.float64),triangles=torch.empty_like(triangles),bounds=torch.empty((nodes,6),device=device),indices=torch.empty(self.triangles,device=device,dtype=torch.int32))
        for name in ['left','right','first','count']:arrays[name]=torch.empty(nodes,device=device,dtype=torch.int32)
        args=[arrays[k]for k in ['bounds','left','right','first','count']]
        mod.launch('leaves',leaves,[triangles,ct.c_int(self.triangles),ct.c_int(group),ct.c_int(leaves),*args,arrays['triangles'],arrays['indices']])
        width=leaves//2
        while width:
            mod.launch('parents',width,[ct.c_int(width-1),ct.c_int(width),*args]);width//=2
        self.kernel=GeometryKernel(arrays,resolution,device)
        self.builder=mod;self.obstacle_cache={}
        torch.cuda.synchronize(device);self.prepare_seconds=time.perf_counter()-tick

    def obstacle_distance(self,keys,base_height):
        """Distance to measured body-height surfaces; no native support query."""
        from scipy.ndimage import distance_transform_edt
        r=self.kernel.resolution;lo=np.asarray(keys).min(0)-8;shape=np.asarray(keys).max(0)-lo+9
        cache_key=tuple(lo)+tuple(shape)+(float(base_height),)
        if cache_key not in self.obstacle_cache:
            nx,nz=map(int,shape);field=torch.zeros((nx,nz),dtype=torch.int32,device=self.kernel.device)
            self.builder.launch('observed_obstacles',self.triangles,[self.kernel.static['triangles'],ct.c_int(self.triangles),ct.c_float(base_height+.2),ct.c_float(base_height+1.4),ct.c_float(r),ct.c_int(int(lo[0])),ct.c_int(int(lo[1])),ct.c_int(nx),ct.c_int(nz),field])
            a=field.cpu().numpy();distance=distance_transform_edt(a==0)*r if a.any()else np.full(a.shape,100.)
            self.obstacle_cache[cache_key]=distance
        ij=np.asarray(keys)-lo
        return self.obstacle_cache[cache_key][ij[:,0],ij[:,1]]
