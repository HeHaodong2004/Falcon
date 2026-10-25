"""GPU range evidence without per-query 63-probe host tensors.

Same legal measured mesh and original output cells. Coordinates are generated
on device, map support is shared, camera views are deduplicated inside a batch.
Numerical equivalence is checked against range_evidence.features.
"""
import ctypes as ct
import numpy as np
import torch
from .cuda_module import CudaModule
from .geometry_cuda import SOURCE as RAYS
from .legal_camera import current_extrinsic,base_rotations
from .legal_camera_features import parameters

SOURCE=RAYS+r'''
extern "C" __global__ void generated_views(const int* keys,const float* floors,int nc,const double* cameras,int nv,
 const unsigned char* unknown,int lx,int lz,int nx,int nz,double resolution,int* output,MESH_ARGS){
 int tid=blockIdx.x*blockDim.x+threadIdx.x;if(tid>=nc*nv)return;int ci=tid%nc,cam=tid/nc;
 int a=keys[ci*2]-lx,b=keys[ci*2+1]-lz,cell=a*nz+b;if(!unknown[cell])return;const double* c=cameras+cam*13;
 double y=floors[ci],x=(keys[ci*2]+.5)*resolution,z=(keys[ci*2+1]+.5)*resolution;
 if(fabs(y-c[12])>.18||!frustum(x,c[12]+.55,z,c))return;
 double radius=.25+resolution/sqrt(2.);
 for(int j=0;j<63;j++){
  int at=j%21;double dx=0.,dz=0.;
  if(at>=5){double angle=(at-5)*6.2831853071795864769/16.;dx=cos(angle)*radius;dz=sin(angle)*radius;}
  else if(at){dx=(at<=2?-1.:1.)*.49*resolution;dz=(at==1||at==3?-1.:1.)*.49*resolution;}
  if(!frustum(x+dx,y+(j/21==0?.025:j/21==1?.7:1.4),z+dz,c))return;
 }
 MESH_VALUE;
 for(int j=0;j<63;j++){
  int at=j%21;double ax=0.,az=0.;
  if(at>=5){double angle=(at-5)*6.2831853071795864769/16.;ax=cos(angle)*radius;az=sin(angle)*radius;}
  else if(at){ax=(at<=2?-1.:1.)*.49*resolution;az=(at==1||at==3?-1.:1.)*.49*resolution;}
  double dx=x+ax-c[0],dy=y+(j/21==0?.025:j/21==1?.7:1.4)-c[1],dz=z+az-c[2],length=sqrt(dx*dx+dy*dy+dz*dz);if(length<1.e-7)continue;float limit=(float)(length-.015);
  float hit=ray(m,(float)c[0],(float)c[1],(float)c[2],(float)(dx/length),(float)(dy/length),(float)(dz/length),limit,true);
  if(hit<0.f){output[cam*nx*nz+cell]=-2147483647;return;}if(hit<limit)return;
 }
 output[cam*nx*nz+cell]=ci+1;
}
extern "C" __global__ void square_support(const int* keys,const float* floors,int n,double resolution,float* supported,float* difference,MESH_ARGS){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;MESH_VALUE;float first_hit=0.f;bool valid=true;
 for(int j=0;j<9;j++){
  double dx=0,dz=0;
  if(j>0&&j<5){dx=(j<=2?-1.:1.)*.49;dz=(j==1||j==3?-1.:1.)*.49;}
  else if(j==5)dx=-.49;else if(j==6)dx=.49;else if(j==7)dz=-.49;else if(j==8)dz=.49;
  float x=(keys[i*2]+.5+dx)*resolution,z=(keys[i*2+1]+.5+dz)*resolution,y=(double)floors[i]+.18;
  float hit=ray(m,x,y,z,0,-1,0,.36f,false);if(j==0)first_hit=hit;if(hit>=.3599f||fabsf(hit-first_hit)>.08f)valid=false;
 }
 supported[i]=valid?1.f:0.f;difference[i]=fmaxf(-.5f,fminf(.5f,.18f-first_hit));
}
__device__ double view_margin(const double* p,double x,double y,double z,double radius){
 double dx=x-p[0],dy=y-p[1],dz=z-p[2],margin=100.;
 for(int k=0;k<5;k++){int a=3+k*4;margin=fmin(margin,dx*p[a]+dy*p[a+1]+dz*p[a+2]+p[a+3]);}
 double horizontal=sqrt(dx*dx+dz*dz)+radius,vertical=fmax(fabs(dy+.025),fabs(dy+1.4));return fmin(margin,10.-sqrt(horizontal*horizontal+vertical*vertical));
}
extern "C" __global__ void pack_evidence(const int* frames,int area,const int* keys,const int* union_ids,const int* cells,const int* queries,int n,
 const int* steps,const int* offsets,const int* initial,const double* params,const double* times,const double* when,const double* durations,
 const float* floors,const float* support,const float* difference,const float* obstacle,double resolution,int lx,int lz,int nz,float* extra){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int q=queries[i],cell=cells[i],id=union_ids[cell],code=0;
 for(int j=offsets[q+1]-1;j>=offsets[q];j--){int v=frames[steps[j]*area+cell];if(v){code=v;break;}}
 float* out=extra+i*16;out[0]=code>0;out[1]=code<0;out[2]=code==0;out[3]=support[id];out[4]=difference[id];out[5]=fminf(obstacle[id],2.f)/2.f;out[6]=obstacle[id]>(.25+resolution/sqrt(2.)+.008);out[7]=0.f;
 double x=(keys[i*2]+.5)*resolution,z=(keys[i*2+1]+.5)*resolution,y=floors[id],radius=.25+resolution/sqrt(2.);
 double first=1.,last=0.,best=-10.,count=0.,weighted=0.,positive=0.;
 double initial_margin=view_margin(params+initial[q]*23,x,y,z,radius);
 for(int j=offsets[q];j<offsets[q+1];j++){
  double margin=view_margin(params+steps[j]*23,x,y,z,radius);best=fmax(best,margin);
  if(margin>=0.){count+=1.;weighted+=times[j];positive+=margin;first=fmin(first,when[j]);last=fmax(last,when[j]);}
 }
 double den=max(1,offsets[q+1]-offsets[q]);out[8]=fmax(-2.,fmin(2.,initial_margin/2.));out[9]=fmax(-2.,fmin(2.,best/2.));out[10]=count/den;out[11]=weighted/durations[q];out[12]=positive/(2*den);out[13]=first;out[14]=last;out[15]=fmax(-2.,fmin(2.,(best-initial_margin)/2.));
}
'''


class RangeFeatureKernel:
    camera_parameters=staticmethod(parameters)
    def __init__(self,device):self.device=torch.device(device);self.module=CudaModule(SOURCE,device)

    def tensor(self,value,dtype=None):return torch.from_numpy(np.ascontiguousarray(value,dtype=dtype)).to(self.device,non_blocking=True)

    def support(self,mesh,union,floors,uk,ff):
        support=torch.empty(len(union),device=self.device);difference=torch.empty_like(support)
        self.module.launch('square_support',len(union),[uk,ff,ct.c_int(len(union)),ct.c_double(mesh.kernel.resolution),support,difference,*mesh.kernel.mesh])
        return support,difference

    def compute(self,mesh,details,gkeys,origin,rotation):
        if not any(len(d[0])for d in details):return [torch.empty((0,16),device=self.device)for d in details]
        keys=np.concatenate([d[0]for d in details]);low=np.array([keys[:,0].min(),keys[:,1].min()])-4;high=np.array([keys[:,0].max(),keys[:,1].max()]);shape=high-low+5;nx,nz=map(int,shape);area=nx*nz;r=mesh.kernel.resolution
        cells=(keys[:,0]-low[0])*nz+keys[:,1]-low[1];active=np.zeros(area,bool);active[cells]=True;flat=np.flatnonzero(active);union=np.column_stack([flat//nz+low[0],flat%nz+low[1]]).astype(np.int32);ids=np.full(area,-1,np.int32);ids[flat]=np.arange(len(flat),dtype=np.int32)
        floors=np.empty(area,np.float32)
        for (kk,baseline,p)in details:
            # Mirror x88[:,24] (F32 baseline minus F32 start height), followed
            # by the F64 start height used by the independent CPU extractor.
            y=(baseline-np.float32(p[0,2])).astype(np.float64)+p[0,2];at=(kk[:,0]-low[0])*nz+kk[:,1]-low[1];floors[at]=y
        floors=floors[flat];u=np.ones((nx,nz),np.uint8);ij=gkeys-low;valid=np.all((ij>=0)&(ij<shape),1);ij=ij[valid];u[ij[:,0],ij[:,1]]=0
        views=[];param=[];dedup={};steps=[];initial=[];interval=[];when=[];durations=[]
        for kk,baseline,p in details:
            translation,extrinsic=current_extrinsic(p[:1],origin,rotation);base=base_rotations(p[:,4]);origins=p[:,1:4]+base@translation;rots=base@extrinsic;cams=np.concatenate([origins,rots.reshape(-1,9),p[:,2:3]],1);params,_=self.camera_parameters(p,translation,extrinsic,r);seq=[]
            for c,pr in zip(cams,params):
                key=c.tobytes()
                if key not in dedup:dedup[key]=len(views);views.append(c);param.append(pr)
                seq.append(dedup[key])
            initial.append(seq[0]);steps.append(np.array(seq[1:],np.int32));duration=max(p[-1,0]-p[0,0],1.e-4);durations.append(duration);interval.extend(np.diff(p[:,0]));when.extend((p[1:,0]-p[0,0])/duration)
        cameras=np.array(views,np.float64);nv=len(cameras)
        if nv*area*4>512*1024*1024:raise ValueError('range camera grid exceeds bounded 512MiB')
        frame=torch.zeros((nv,area),dtype=torch.int32,device=self.device);uk=self.tensor(union);ff=self.tensor(floors);cam=self.tensor(cameras);unknown=self.tensor(u);mesh_args=mesh.kernel.mesh
        self.module.launch('generated_views',len(union)*nv,[uk,ff,ct.c_int(len(union)),cam,ct.c_int(nv),unknown,ct.c_int(int(low[0])),ct.c_int(int(low[1])),ct.c_int(nx),ct.c_int(nz),ct.c_double(r),frame,*mesh_args])
        self.module.launch('wall_views',nv*105,[cam,ct.c_int(nv),unknown,ct.c_int(int(low[0])),ct.c_int(int(low[1])),ct.c_int(nx),ct.c_int(nz),ct.c_double(r),frame,*mesh_args])
        support,difference=self.support(mesh,union,floors,uk,ff)
        obstacle=self.tensor(mesh.obstacle_distance(union,details[0][2][0,2]).astype(np.float32))
        offsets=np.r_[0,np.cumsum([len(s)for s in steps])].astype(np.int32);queries=np.repeat(np.arange(len(details),dtype=np.int32),[len(d[0])for d in details]);out=torch.empty((len(keys),16),device=self.device)
        self.module.launch('pack_evidence',len(keys),[frame,ct.c_int(area),self.tensor(keys),self.tensor(ids),self.tensor(cells.astype(np.int32)),self.tensor(queries),ct.c_int(len(keys)),self.tensor(np.concatenate(steps)),self.tensor(offsets),self.tensor(initial,np.int32),self.tensor(np.array(param,np.float64)),self.tensor(interval,np.float64),self.tensor(when,np.float64),self.tensor(durations,np.float64),ff,support,difference,obstacle,ct.c_double(r),ct.c_int(int(low[0])),ct.c_int(int(low[1])),ct.c_int(nz),out])
        return list(out.split([len(d[0])for d in details]))
