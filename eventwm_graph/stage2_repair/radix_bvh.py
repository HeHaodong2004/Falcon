"""GPU binary radix hierarchy for observed triangles; no geometry is removed."""
import ctypes as ct
import time
import torch
from .cuda_module import CudaModule

SOURCE=r'''
__device__ unsigned int spread(unsigned int x){x&=1023;x=(x|(x<<16))&0x030000FF;x=(x|(x<<8))&0x0300F00F;x=(x|(x<<4))&0x030C30C3;x=(x|(x<<2))&0x09249249;return x;}
extern "C" __global__ void triangle_codes(const float* t,int n,const float* lo,const float* extent,long long* codes){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;unsigned int q[3];for(int j=0;j<3;j++){float p=t[9*i+j]+(t[9*i+3+j]+t[9*i+6+j])/3.f;q[j]=max(0,min(1023,(int)((p-lo[j])/extent[j]*1023)));}codes[i]=spread(q[0])|(spread(q[1])<<1)|(spread(q[2])<<2);
}
extern "C" __global__ void make_leaves(const float* t,const long long* codes,int triangles,int group,int leaves,float* bounds,int* first,int* count,long long* keys){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=leaves)return;int node=leaves-1+i,begin=i*group,end=min(begin+group,triangles);first[node]=begin;count[node]=end-begin;keys[i]=(codes[(begin+end-1)/2]<<32)|(unsigned int)i;
 for(int j=0;j<3;j++){float lo=1.e30f,hi=-1.e30f;for(int k=begin;k<end;k++){float a=t[k*9+j],b=a+t[k*9+3+j],c=a+t[k*9+6+j];lo=fminf(lo,fminf(a,fminf(b,c)));hi=fmaxf(hi,fmaxf(a,fmaxf(b,c)));}bounds[node*6+j]=lo;bounds[node*6+j+3]=hi;}
}
__device__ int prefix(const long long* keys,int n,int i,int j){if(j<0||j>=n)return -1;return __clzll((unsigned long long)(keys[i]^keys[j]));}
extern "C" __global__ void radix_tree(const long long* keys,int n,int* left,int* right,int* parent){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n-1)return;
 int d=prefix(keys,n,i,i+1)>prefix(keys,n,i,i-1)?1:-1,minimum=prefix(keys,n,i,i-d),maximum=2;
 while(prefix(keys,n,i,i+maximum*d)>minimum)maximum*=2;
 int length=0;for(int step=maximum/2;step>=1;step/=2)if(prefix(keys,n,i,i+(length+step)*d)>minimum)length+=step;
 int j=i+length*d,common=prefix(keys,n,i,j),split=0;
 for(int step=(length+1)/2;step>=1;step=(step+1)/2){if(split+step<length&&prefix(keys,n,i,i+(split+step)*d)>common)split+=step;if(step==1)break;}
 int gamma=i+split*d+(d<0?-1:0),a=min(i,j)==gamma?n-1+gamma:gamma,b=max(i,j)==gamma+1?n-1+gamma+1:gamma+1;left[i]=a;right[i]=b;parent[a]=i;parent[b]=i;
}
extern "C" __global__ void reduce_bounds(int leaves,const int* left,const int* right,const int* parent,int* visited,float* bounds){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=leaves)return;int node=parent[leaves-1+i];
 while(node>=0){if(atomicAdd(visited+node,1)==0)return;int a=left[node],b=right[node];for(int j=0;j<3;j++){bounds[node*6+j]=fminf(bounds[a*6+j],bounds[b*6+j]);bounds[node*6+j+3]=fmaxf(bounds[a*6+j+3],bounds[b*6+j+3]);}__threadfence();node=parent[node];}
}
'''


def build_arrays(t):
    """Build directly from packed triangles; avoid constructing a discarded tree."""
    device=t.device;mod=CudaModule(SOURCE,device);n=len(t)
    if not n:raise ValueError('radix hierarchy requires a measured surface')
    centers=t[:,:3]+(t[:,3:6]+t[:,6:9])/3;lo=centers.amin(0);extent=(centers.amax(0)-lo).clamp_min(.001);codes=torch.empty(n,device=device,dtype=torch.int64);mod.launch('triangle_codes',n,[t,ct.c_int(n),lo,extent,codes]);codes,order=torch.sort(codes);t=t[order];group=12;leaves=(n+group-1)//group;nodes=leaves*2-1;bounds=torch.empty((nodes,6),device=device);first=torch.zeros(nodes,device=device,dtype=torch.int32);count=torch.zeros_like(first);left=torch.full_like(first,-1);right=torch.full_like(first,-1);parent=torch.full_like(first,-1);keys=torch.empty(leaves,device=device,dtype=torch.int64);visited=torch.zeros(max(1,leaves-1),device=device,dtype=torch.int32)
    mod.launch('make_leaves',leaves,[t,codes,ct.c_int(n),ct.c_int(group),ct.c_int(leaves),bounds,first,count,keys])
    if leaves>1:
        mod.launch('radix_tree',leaves-1,[keys,ct.c_int(leaves),left,right,parent]);mod.launch('reduce_bounds',leaves,[ct.c_int(leaves),left,right,parent,visited,bounds])
    return dict(triangles=t,bounds=bounds,left=left,right=right,first=first,count=count,indices=torch.arange(n,device=device,dtype=torch.int32))


def rebuild(mesh):
    tick=time.perf_counter();k=mesh.kernel;arrays=build_arrays(k.static['triangles']);k.static.update(arrays);k.arrays.update(arrays);k.mesh=[k.static[name]for name in ['triangles','bounds','left','right','first','count','indices']];mesh.obstacle_cache={};torch.cuda.synchronize(k.device);mesh.radix_build_seconds=time.perf_counter()-tick;mesh.prepare_seconds+=mesh.radix_build_seconds
    return mesh
