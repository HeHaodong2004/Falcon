"""CUDA kernels for the legal tri-state voxel map and certificate simulation.

Every kernel consumes only legal inputs: measured past/current range images,
their recorded camera poses, the current graph and the supplied trajectory.
Voxel states: 0 UNKNOWN (never measured), 1 FREE (a measured ray passed
through it), 2 OCCUPIED (a measured surface hit lies inside it).

Coordinate conventions follow the environment sensor exactly: world y is up,
camera looks along local -z, pixel (row i, col j) has local direction
[(j+.5)*2/S-1, 1-(i+.5)*2/S, -1]. Grid keys are floor(x/res), floor(z/res).
The volume index is ((ix*nz)+iz)*ny+iy (vertical innermost).
"""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import threading

import torch

from eventwm_graph.stage2_segment.cuda_kernel import checked

SOURCE = r'''
#define MAXR 10.0f
// Unknown voxels within v.near m of a camera are its own robot/sensor volume,
// which the environment sensor ignores; they never occlude or hide surfaces.
struct Vol {float x0,z0,y0,res,dy;int nx,nz,ny;float cres;int sub;float near;};

__device__ __forceinline__ int vindex(const Vol& v,int ix,int iz,int iy){return (ix*v.nz+iz)*v.ny+iy;}

// Scatter measured surface hits. Plain stores of a constant are race-free.
extern "C" __global__ void scatter_hits(const float* ranges,const float* cams,int frames,int size,
    Vol v,unsigned char* occ,const unsigned char* keep){
  long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  long long total=(long long)frames*size*size;if(t>=total)return;
  int f=t/(size*size);int pix=t%(size*size);int i=pix/size,j=pix%size;
  if(keep&&!keep[t])return;
  float r=ranges[t];if(!(r==r))return;
  const float* c=cams+f*12;
  float lx=(j+.5f)*2.f/size-1.f,ly=1.f-(i+.5f)*2.f/size,lz=-1.f;
  float n=rsqrtf(lx*lx+ly*ly+1.f);lx*=n;ly*=n;lz*=n;
  // world = R @ local (R row-major in c[3..11])
  float dx=c[3]*lx+c[4]*ly+c[5]*lz,dy=c[6]*lx+c[7]*ly+c[8]*lz,dz=c[9]*lx+c[10]*ly+c[11]*lz;
  float px=c[0]+dx*r,py=c[1]+dy*r,pz=c[2]+dz*r;
  int ix=(int)floorf((px-v.x0)/v.res),iz=(int)floorf((pz-v.z0)/v.res),iy=(int)floorf((py-v.y0)/v.dy);
  if(ix<0||iz<0||iy<0||ix>=v.nx||iz>=v.nz||iy>=v.ny)return;
  occ[vindex(v,ix,iz,iy)]=1;
}

__device__ __forceinline__ bool pixel_point(const float* ranges,const float* c,int f,int size,int i,int j,float* p){
  float r=ranges[((long long)f*size+i)*size+j];if(!(r==r))return false;
  float lx=(j+.5f)*2.f/size-1.f,ly=1.f-(i+.5f)*2.f/size,lz=-1.f;
  float n=rsqrtf(lx*lx+ly*ly+1.f);lx*=n;ly*=n;lz*=n;
  p[0]=c[0]+(c[3]*lx+c[4]*ly+c[5]*lz)*r;p[1]=c[1]+(c[6]*lx+c[7]*ly+c[8]*lz)*r;p[2]=c[2]+(c[9]*lx+c[10]*ly+c[11]*lz)*r;
  p[3]=r;return true;
}

// Measured-surface splatting: each locally planar 2x2 pixel quad is
// resampled densely (bilinear) so grazing distant surfaces stay closed.
// Depth discontinuities (non-parallelogram quads) are never bridged.
__device__ __forceinline__ bool near_human(const float* p,const float* humans,const int* hstart,int f,float radius){
  if(!humans)return false;
  for(int h=hstart[f];h<hstart[f+1];h++){
    const float* q=humans+3*h;float dx=p[0]-q[0],dz=p[2]-q[2];
    if(dx*dx+dz*dz<=radius*radius&&p[1]>q[1]-.3f&&p[1]<q[1]+2.2f)return true;
  }
  return false;
}

extern "C" __global__ void splat_hits(const float* ranges,const float* cams,int frames,int size,
    Vol v,unsigned char* occ,float tol0,float tol1,float max_edge,const float* humans,const int* hstart,float hradius){
  long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  long long total=(long long)frames*size*size;if(t>=total)return;
  int f=t/(size*size);int pix=t%(size*size);int i=pix/size,j=pix%size;
  const float* c=cams+f*12;float a[4],b[4],d[4],e[4];
  if(!pixel_point(ranges,c,f,size,i,j,a))return;
  // Legally tracked humans (visible at this frame) are dynamic, not static.
  if(near_human(a,humans,hstart,f,hradius))return;
  {int ix=(int)floorf((a[0]-v.x0)/v.res),iz=(int)floorf((a[2]-v.z0)/v.res),iy=(int)floorf((a[1]-v.y0)/v.dy);
   if(ix>=0&&iz>=0&&iy>=0&&ix<v.nx&&iz<v.nz&&iy<v.ny)occ[vindex(v,ix,iz,iy)]=1;}
  if(i+1>=size||j+1>=size)return;
  if(!pixel_point(ranges,c,f,size,i,j+1,b)||!pixel_point(ranges,c,f,size,i+1,j,d)||!pixel_point(ranges,c,f,size,i+1,j+1,e))return;
  float rr=.25f*(a[3]+b[3]+d[3]+e[3]);
  float res0=a[0]+e[0]-b[0]-d[0],res1=a[1]+e[1]-b[1]-d[1],res2=a[2]+e[2]-b[2]-d[2];
  if(sqrtf(res0*res0+res1*res1+res2*res2)>tol0+tol1*rr)return;
  float edge=0.f;
  #define EL(P,Q) sqrtf((P[0]-Q[0])*(P[0]-Q[0])+(P[1]-Q[1])*(P[1]-Q[1])+(P[2]-Q[2])*(P[2]-Q[2]))
  edge=fmaxf(fmaxf(EL(a,b),EL(a,d)),fmaxf(EL(e,b),EL(e,d)));
  if(edge>max_edge)return;
  int k=(int)ceilf(edge/(.5f*fminf(v.res,v.dy)));if(k<2)return;if(k>12)k=12;
  for(int u=0;u<=k;u++)for(int w=0;w<=k;w++){
    float su=(float)u/k,sw=(float)w/k;
    float q0=(1-su)*((1-sw)*a[0]+sw*b[0])+su*((1-sw)*d[0]+sw*e[0]);
    float q1=(1-su)*((1-sw)*a[1]+sw*b[1])+su*((1-sw)*d[1]+sw*e[1]);
    float q2=(1-su)*((1-sw)*a[2]+sw*b[2])+su*((1-sw)*d[2]+sw*e[2]);
    int ix=(int)floorf((q0-v.x0)/v.res),iz=(int)floorf((q2-v.z0)/v.res),iy=(int)floorf((q1-v.y0)/v.dy);
    if(ix>=0&&iz>=0&&iy>=0&&ix<v.nx&&iz<v.nz&&iy<v.ny)occ[vindex(v,ix,iz,iy)]=1;
  }
}

// Projective carving: a voxel center is FREE when some measured ray passes
// beyond it by at least margin. Missing returns mean no hit within 10 m.
extern "C" __global__ void carve(const float* ranges,const float* cams,int frames,int size,
    Vol v,const unsigned char* occ,unsigned char* state,float margin){
  int t=blockIdx.x*blockDim.x+threadIdx.x;int total=v.nx*v.nz*v.ny;if(t>=total)return;
  if(occ[t]){state[t]=2;return;}
  int iy=t%v.ny;int iz=(t/v.ny)%v.nz;int ix=t/(v.ny*v.nz);
  float x=v.x0+(ix+.5f)*v.res,y=v.y0+(iy+.5f)*v.dy,z=v.z0+(iz+.5f)*v.res;
  unsigned char s=0;
  for(int f=0;f<frames&&!s;f++){
    const float* c=cams+f*12;float dx=x-c[0],dy=y-c[1],dz=z-c[2];
    float d=sqrtf(dx*dx+dy*dy+dz*dz);if(d>MAXR)continue;
    // local = R^T d
    float lx=c[3]*dx+c[6]*dy+c[9]*dz,ly=c[4]*dx+c[7]*dy+c[10]*dz,lz=c[5]*dx+c[8]*dy+c[11]*dz;
    float fw=-lz;if(fw<=1e-6f)continue;float u=lx/fw,w=ly/fw;if(fabsf(u)>=1.f||fabsf(w)>=1.f)continue;
    int j=(int)floorf((u+1.f)*.5f*size),i=(int)floorf((1.f-w)*.5f*size);
    j=min(max(j,0),size-1);i=min(max(i,0),size-1);
    float r=ranges[((long long)f*size+i)*size+j];if(!(r==r))r=MAXR+margin;
    if(d<r-margin)s=1;
  }
  state[t]=s;
}

// Exact frame culling for carving: per block of CB x CB voxel columns (full height), a bitmask of
// the frames that can carve some voxel center in it. A frame is dropped only if one of the
// per-voxel rejection tests of `carve` holds for EVERY voxel center of the block with a 1 cm
// margin (beyond 10 m; behind the camera; outside one frustum side plane), so `carve_culled`
// returns exactly what `carve` returns.
#define CB 8
extern "C" __global__ void cull_frames(const float* cams,int frames,Vol v,int nbx,int nbz,int words,unsigned int* masks){
  long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(t>=(long long)nbx*nbz*frames)return;
  int f=t%frames;int b=t/frames;int bx=b/nbz,bz=b%nbz;
  int ix0=bx*CB,iz0=bz*CB,ix1=min(ix0+CB,v.nx)-1,iz1=min(iz0+CB,v.nz)-1;
  float xa=v.x0+(ix0+.5f)*v.res,xb=v.x0+(ix1+.5f)*v.res,za=v.z0+(iz0+.5f)*v.res,zb=v.z0+(iz1+.5f)*v.res;
  float ya=v.y0+.5f*v.dy,yb=v.y0+(v.ny-.5f)*v.dy;
  const float* c=cams+f*12;
  float qx=fminf(fmaxf(c[0],xa),xb),qy=fminf(fmaxf(c[1],ya),yb),qz=fminf(fmaxf(c[2],za),zb);
  float ex=qx-c[0],ey=qy-c[1],ez=qz-c[2];
  if(ex*ex+ey*ey+ez*ez>(MAXR+.01f)*(MAXR+.01f))return;
  // Linear functions of the voxel center: fw=-lz, fw-lx, fw+lx, fw-ly, fw+ly; drop if the maximum
  // over the 8 box corners is below -1 cm (fw) / -1 cm (sides: |u|>=1 or |w|>=1 everywhere).
  float mfw=-1e30f,m1=-1e30f,m2=-1e30f,m3=-1e30f,m4=-1e30f;
  for(int k=0;k<8;k++){
    float dx=((k&1)?xb:xa)-c[0],dy=((k&2)?yb:ya)-c[1],dz=((k&4)?zb:za)-c[2];
    float lx=c[3]*dx+c[6]*dy+c[9]*dz,ly=c[4]*dx+c[7]*dy+c[10]*dz,fw=-(c[5]*dx+c[8]*dy+c[11]*dz);
    mfw=fmaxf(mfw,fw);m1=fmaxf(m1,fw-lx);m2=fmaxf(m2,fw+lx);m3=fmaxf(m3,fw-ly);m4=fmaxf(m4,fw+ly);
  }
  if(mfw<-.01f||m1<-.01f||m2<-.01f||m3<-.01f||m4<-.01f)return;
  atomicOr(masks+(long long)b*words+f/32,1u<<(f%32));
}

extern "C" __global__ void carve_culled(const float* ranges,const float* cams,int frames,int size,
    Vol v,const unsigned char* occ,unsigned char* state,float margin,int nbz,int words,const unsigned int* masks){
  int t=blockIdx.x*blockDim.x+threadIdx.x;int total=v.nx*v.nz*v.ny;if(t>=total)return;
  if(occ[t]){state[t]=2;return;}
  int iy=t%v.ny;int iz=(t/v.ny)%v.nz;int ix=t/(v.ny*v.nz);
  float x=v.x0+(ix+.5f)*v.res,y=v.y0+(iy+.5f)*v.dy,z=v.z0+(iz+.5f)*v.res;
  const unsigned int* mk=masks+(long long)((ix/CB)*nbz+iz/CB)*words;
  unsigned char s=0;
  for(int wd=0;wd<words&&!s;wd++){
    unsigned int bits=mk[wd];
    while(bits&&!s){
      int f=wd*32+__ffs(bits)-1;bits&=bits-1;
      const float* c=cams+f*12;float dx=x-c[0],dy=y-c[1],dz=z-c[2];
      float d=sqrtf(dx*dx+dy*dy+dz*dz);if(d>MAXR)continue;
      float lx=c[3]*dx+c[6]*dy+c[9]*dz,ly=c[4]*dx+c[7]*dy+c[10]*dz,lz=c[5]*dx+c[8]*dy+c[11]*dz;
      float fw=-lz;if(fw<=1e-6f)continue;float u=lx/fw,w=ly/fw;if(fabsf(u)>=1.f||fabsf(w)>=1.f)continue;
      int j=(int)floorf((u+1.f)*.5f*size),i=(int)floorf((1.f-w)*.5f*size);
      j=min(max(j,0),size-1);i=min(max(i,0),size-1);
      float r=ranges[((long long)f*size+i)*size+j];if(!(r==r))r=MAXR+margin;
      if(d<r-margin)s=1;
    }
  }
  state[t]=s;
}

// Amanatides-Woo traversal. Returns distance to first OCCUPIED voxel (or
// big), first UNKNOWN voxel (or big) and the total UNKNOWN length crossed
// before termination; stops at tmax or the first occupied voxel.
__device__ void march(const Vol& v,const unsigned char* state,float ox,float oy,float oz,
    float dx,float dy,float dz,float tmax,float* t_occ,float* t_unk,float* u_len){
  *t_occ=1e9f;*t_unk=1e9f;*u_len=0.f;
  float gx=(ox-v.x0)/v.res,gz=(oz-v.z0)/v.res,gy=(oy-v.y0)/v.dy;
  int ix=(int)floorf(gx),iz=(int)floorf(gz),iy=(int)floorf(gy);
  int sx=dx>0?1:-1,sz=dz>0?1:-1,sy=dy>0?1:-1;
  float inv_x=fabsf(dx)>1e-9f?1.f/fabsf(dx):1e30f,inv_z=fabsf(dz)>1e-9f?1.f/fabsf(dz):1e30f,inv_y=fabsf(dy)>1e-9f?1.f/fabsf(dy):1e30f;
  float nx_=dx>0?(ix+1-gx):(gx-ix),nz_=dz>0?(iz+1-gz):(gz-iz),ny_=dy>0?(iy+1-gy):(gy-iy);
  float tx=nx_*v.res*inv_x,tz=nz_*v.res*inv_z,ty=ny_*v.dy*inv_y;
  float ddx=v.res*inv_x,ddz=v.res*inv_z,ddy=v.dy*inv_y;
  float t=0.f;
  for(int step=0;step<4096;step++){
    if(ix<0||iz<0||iy<0||ix>=v.nx||iz>=v.nz||iy>=v.ny)return;
    unsigned char s=state[vindex(v,ix,iz,iy)];
    if(s>=2){*t_occ=t;return;}
    float tn;int axis;
    if(tx<tz&&tx<ty){tn=tx;axis=0;}else if(tz<ty){tn=tz;axis=1;}else{tn=ty;axis=2;}
    if(s==0&&tn>v.near){float t0=fmaxf(t,v.near);if(*t_unk>1e8f)*t_unk=t0;*u_len+=fminf(tn,tmax)-t0;}
    t=tn;
    if(axis==0){tx+=ddx;ix+=sx;}else if(axis==1){tz+=ddz;iz+=sz;}else{ty+=ddy;iy+=sy;}
    if(t>tmax)return;
  }
}

// Wall-ray certificate simulation: exactly the 35x3 directions of the
// environment observer, from supplied proxy camera poses.
extern "C" __global__ void wall_rays(const float* cams,int poses,Vol v,const unsigned char* state,float* out){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=poses*105)return;
  int p=t/105,k=t%105;int a=k/3,b=k%3;
  float lx=-.95f+1.9f*a/34.f,ly=(b==0?-.5f:(b==1?0.f:.5f)),lz=-1.f;
  const float* c=cams+p*12;
  float dx=c[3]*lx+c[4]*ly+c[5]*lz,dy=c[6]*lx+c[7]*ly+c[8]*lz,dz=c[9]*lx+c[10]*ly+c[11]*lz;
  float n=rsqrtf(dx*dx+dy*dy+dz*dz);dx*=n;dy*=n;dz*=n;
  float to,tu,ul;march(v,state,c[0],c[1],c[2],dx,dy,dz,MAXR,&to,&tu,&ul);
  float* o=out+t*6;o[0]=to;o[1]=tu;o[2]=dx;o[3]=dy;o[4]=dz;o[5]=ul;
}

// Render per-pose visibility images: (t_occ, t_unk) per pixel.
extern "C" __global__ void render(const float* cams,int poses,int size,Vol v,const unsigned char* state,float2* out){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=poses*size*size)return;
  int p=t/(size*size);int pix=t%(size*size);int i=pix/size,j=pix%size;
  float lx=(j+.5f)*2.f/size-1.f,ly=1.f-(i+.5f)*2.f/size,lz=-1.f;
  const float* c=cams+p*12;
  float dx=c[3]*lx+c[4]*ly+c[5]*lz,dy=c[6]*lx+c[7]*ly+c[8]*lz,dz=c[9]*lx+c[10]*ly+c[11]*lz;
  float n=rsqrtf(dx*dx+dy*dy+dz*dz);dx*=n;dy*=n;dz*=n;
  float to,tu,ul;march(v,state,c[0],c[1],c[2],dx,dy,dz,MAXR+.5f,&to,&tu,&ul);
  out[t]=make_float2(to,tu);
}

// Per candidate cell: certificate opportunity across a query's poses.
// Probes replicate the observer: 5 square points (center + corners at
// .49 res), 16 halo points at .25+res/sqrt2, at heights +.025/.7/1.4.
// out[i*8+..]: 0 known-all-visible (best pose), 1 possible (no known
// blocker), 2 max visible fraction, 3 min uncertain fraction over frustum
// poses, 4 frustum-ok poses fraction, 5 min known-blocked fraction,
// 6 min distance of a frustum-ok pose, 7 any pose with center in frustum.
extern "C" __global__ void probes(const int* keys,const float* floor_h,int n,const float* cams,
    const int* pose_index,int npose,int size,const float2* images,float res,
    float tol_occ,float tol_unk,float* out){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  float cx=(keys[2*i]+.5f)*res,cz=(keys[2*i+1]+.5f)*res,h=floor_h[i];
  float halo=.25f+res*0.70710678f;
  float known=0.f,possible=0.f,best_vis=0.f,best_unc=1.f,frustum_poses=0.f,best_blk=1.f,mind=99.f,center_any=0.f;
  for(int q=0;q<npose;q++){
    int p=pose_index[q];const float* c=cams+p*12;
    // quick center test (generous)
    float ex=cx-c[0],ez=cz-c[2];float hd=sqrtf(ex*ex+ez*ez);if(hd>10.5f)continue;
    {float ey=h+.7f-c[1];float lx=c[3]*ex+c[6]*ey+c[9]*ez,lz=c[5]*ex+c[8]*ey+c[11]*ez;float fw=-lz;
     if(fw<=0.f||fabsf(lx)>fw+.5f)continue;center_any=1.f;}
    int vis=0,unc=0,blk=0;bool inside=true;
    for(int k=0;k<63&&inside;k++){
      int m=k%21;int level=k/21;float px,pz;
      if(m==0){px=cx;pz=cz;}
      else if(m<5){float sx=(m==1||m==2)?-.49f:.49f,sz=(m==1||m==3)?-.49f:.49f;px=cx+sx*res;pz=cz+sz*res;}
      else{float ang=6.283185307f*(m-5)/16.f;px=cx+cosf(ang)*halo;pz=cz+sinf(ang)*halo;}
      float py=h+(level==0?.025f:(level==1?.7f:1.4f));
      float dx=px-c[0],dy=py-c[1],dz=pz-c[2];float d=sqrtf(dx*dx+dy*dy+dz*dz);
      float lx=c[3]*dx+c[6]*dy+c[9]*dz,ly=c[4]*dx+c[7]*dy+c[10]*dz,lz=c[5]*dx+c[8]*dy+c[11]*dz;
      float fw=-lz;
      if(!(fw>1e-6f&&fabsf(lx)<=fw&&fabsf(ly)<=fw&&d<=MAXR)){inside=false;break;}
      int j=(int)floorf((lx/fw+1.f)*.5f*size),ii=(int)floorf((1.f-ly/fw)*.5f*size);
      j=min(max(j,0),size-1);ii=min(max(ii,0),size-1);
      float2 r=images[((long long)p*size+ii)*size+j];
      if(r.x<d-tol_occ)blk++;else if(r.y<d-tol_unk)unc++;else vis++;
    }
    if(!inside)continue;
    frustum_poses+=1.f;mind=fminf(mind,hd);
    best_blk=fminf(best_blk,blk/63.f);
    if(blk==0){possible=1.f;best_unc=fminf(best_unc,unc/63.f);if(unc==0)known=1.f;}
    best_vis=fmaxf(best_vis,vis/63.f);
  }
  float* o=out+i*8;o[0]=known;o[1]=possible;o[2]=best_vis;o[3]=best_unc;
  o[4]=frustum_poses/(float)max(npose,1);o[5]=best_blk;o[6]=mind;o[7]=center_any;
}

// Map channels per CELL: graph FREE/BLOCKED then, per height band (in 0.1 m
// bins), fractions of OCCUPIED/FREE/UNKNOWN voxels over the cell's sub x sub
// voxel columns. bands: nb (start,end) pairs in 0.1 m bins.
extern "C" __global__ void map_channels(Vol v,const unsigned char* state,const unsigned char* graph_state,
    const int* bands,int nb,float* out,int with_humans){
  int cnx=v.nx/v.sub,cnz=v.nz/v.sub;long long plane=(long long)cnx*cnz;
  long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(t>=plane)return;
  int cx=t/cnz,cz=t%cnz;unsigned char g=graph_state[t];
  out[t]=g==1?1.f:0.f;out[plane+t]=g==2?1.f:0.f;
  for(int k=0;k<nb;k++){
    int a=bands[2*k]*v.sub,b=bands[2*k+1]*v.sub;int no=0,nf=0,nu=0;
    for(int sx=0;sx<v.sub;sx++)for(int sz=0;sz<v.sub;sz++){
      const unsigned char* col=state+((long long)(cx*v.sub+sx)*v.nz+(cz*v.sub+sz))*v.ny;
      for(int y=a;y<b;y++){unsigned char s=col[y];no+=s==2;nf+=s==1;nu+=s==0;}
    }
    float inv=1.f/(float)((b-a)*v.sub*v.sub);
    out[(2+3*k)*plane+t]=no*inv;out[(3+3*k)*plane+t]=nf*inv;out[(4+3*k)*plane+t]=nu*inv;
  }
  if(with_humans){
    int nh=0;
    for(int sx=0;sx<v.sub;sx++)for(int sz=0;sz<v.sub;sz++){
      const unsigned char* col=state+((long long)(cx*v.sub+sx)*v.nz+(cz*v.sub+sz))*v.ny;
      for(int y=0;y<v.ny;y++)nh+=col[y]==3;
    }
    out[(2+3*nb)*plane+t]=nh>0?1.f:0.f;
  }
}

// Current legally tracked humans as dynamic occluders (state 3): cylinders
// that block lines of sight; static OCCUPIED voxels are kept.
extern "C" __global__ void stamp_humans(Vol v,unsigned char* state,const float* humans,int nh,float radius,int bx,int bz,int by){
  long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;long long per=(long long)bx*bz*by;
  if(t>=per*nh)return;
  int h=t/per;long long r=t%per;int ax=r/(bz*by),az=(r/by)%bz,ay=r%by;
  const float* p=humans+3*h;
  int ix=(int)floorf((p[0]-radius-v.x0)/v.res)+ax,iz=(int)floorf((p[2]-radius-v.z0)/v.res)+az,iy=(int)floorf((p[1]-.2f-v.y0)/v.dy)+ay;
  if(ix<0||iz<0||iy<0||ix>=v.nx||iz>=v.nz||iy>=v.ny)return;
  float cx=v.x0+(ix+.5f)*v.res-p[0],cz=v.z0+(iz+.5f)*v.res-p[2],cy=v.y0+(iy+.5f)*v.dy-p[1];
  if(cx*cx+cz*cz>radius*radius||cy<-.2f||cy>1.9f)return;
  int vi=vindex(v,ix,iz,iy);if(state[vi]!=2)state[vi]=3;
}

// ---------------------------------------------------------------------------
// Frame kernels: each query q has its own network frame frames[q]=(gx0,gz0,h,w)
// in group-voxel coordinates; outputs are [q][channel][Hp][Wp] float32.

// EXACT original candidate domain (stage2_quality.features.candidate_mask):
// x=f32((k+.5)*f32(res)); keep if sqrt(f64 dx^2+f64 dz^2)<=11 for any unique
// f32 supplied position. Explicit _rn intrinsics: no contraction/fast math.
// Constant-velocity sweep of the CURRENT legally tracked humans over each
// query's duration (x/z of position and estimated velocity; motion rows
// [px,py,pz,vx,vy,vz]): 1-s/(T+1) for the earliest time s in [0,T] at which a
// human is within radius of the cell center, else 0.
extern "C" __global__ void human_sweep(const int* frames,int Q,int Hp,int Wp,Vol v,const float* motion,int nh,
    const float* durations,float radius,int C,int ch,float* x){
  long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(t>=(long long)Q*Hp*Wp)return;
  int q=t/((long long)Hp*Wp);int a=(t/Wp)%Hp,b=t%Wp;
  const int* fr=frames+4*q;if(a>=fr[2]||b>=fr[3])return;
  float cx=v.x0+(fr[0]+a+.5f)*v.cres,cz=v.z0+(fr[1]+b+.5f)*v.cres;
  float T=fmaxf(durations[q],0.f),best=0.f,r2=radius*radius;
  for(int h=0;h<nh;h++){
    const float* m=motion+6*h;
    float px=m[0]-cx,pz=m[2]-cz,vx=m[3],vz=m[5];
    float vv=vx*vx+vz*vz,p2=px*px+pz*pz;
    float s=0.f;
    if(vv>1e-8f){s=-(px*vx+pz*vz)/vv;s=fminf(fmaxf(s,0.f),T);}
    float dx=px+vx*s,dz=pz+vz*s;
    if(dx*dx+dz*dz>r2)continue;
    float s0=0.f;
    if(p2>r2&&vv>1e-8f){float bq=(px*vx+pz*vz)/vv,cq=(p2-r2)/vv;s0=fmaxf(-bq-sqrtf(fmaxf(bq*bq-cq,0.f)),0.f);}
    best=fmaxf(best,1.f-s0/(T+1.f));
  }
  x[(((long long)q*C+ch)*Hp+a)*Wp+b]=best;
}

extern "C" __global__ void domain_mask(const int* frames,int Q,int Hp,int Wp,Vol v,const unsigned char* graph_known,
    const float* positions,const int* pos_start,const int* pos_count,float res32,unsigned char* mask){
  long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(t>=(long long)Q*Hp*Wp)return;
  int q=t/((long long)Hp*Wp);int a=(t/Wp)%Hp,b=t%Wp;
  const int* fr=frames+4*q;unsigned char keep=0;
  if(a<fr[2]&&b<fr[3]){
    int gx=fr[0]+a,gz=fr[1]+b;
    int cnx=v.nx/v.sub,cnz=v.nz/v.sub;
    if(gx>=0&&gz>=0&&gx<cnx&&gz<cnz&&!graph_known[(long long)gx*cnz+gz]){
      int lox=(int)lrintf(v.x0/v.cres),loz=(int)lrintf(v.z0/v.cres);
      float x=__fmul_rn((float)(lox+gx)+.5f,res32),z=__fmul_rn((float)(loz+gz)+.5f,res32);
      for(int j=0;j<pos_count[q]&&!keep;j++){
        const float* p=positions+2*(pos_start[q]+j);
        // F32 prefilter with a 1e-5 relative margin on the squared distance (F32 error < 3e-7):
        // only borderline cells take the exact F64 path, so the result is unchanged.
        float fx=x-p[0],fz=z-p[1],s2=fx*fx+fz*fz;
        if(s2<120.9987f){keep=1;break;}
        if(s2>121.0013f)continue;
        double dx=__dsub_rn((double)x,(double)p[0]),dz=__dsub_rn((double)z,(double)p[1]);
        if(__dsqrt_rn(__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dz,dz)))<=11.)keep=1;
      }
    }
  }
  mask[t]=keep;
}

// Probe statistics written into the network input at channel offset c0.
extern "C" __global__ void probe_frame(const float* cams,const int* query_start,const int* query_count,
    const int* pose_list,int Q,int size,const float2* images,Vol v,const unsigned char* graph_known,
    const int* frames,const float* floor_h,float tol_occ,float tol_unk,int Hp,int Wp,int C,int c0,float* x){
  long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(t>=(long long)Q*Hp*Wp)return;
  int q=t/((long long)Hp*Wp);int a=(t/Wp)%Hp,b=t%Wp;
  const int* fr=frames+4*q;if(a>=fr[2]||b>=fr[3])return;
  int cnx=v.nx/v.sub,cnz=v.nz/v.sub;
  int gx=fr[0]+a,gz=fr[1]+b;if(gx<0||gz<0||gx>=cnx||gz>=cnz)return;
  if(graph_known[(long long)gx*cnz+gz])return;
  long long plane=(long long)Hp*Wp;float* o=x+((long long)q*C+c0)*plane+(long long)a*Wp+b;
  int lox=(int)lrintf(v.x0/v.cres),loz=(int)lrintf(v.z0/v.cres);
  float res=v.cres,cx=(lox+gx+.5f)*res,cz=(loz+gz+.5f)*res,h=floor_h[q];
  float halo=.25f+res*0.70710678f;
  float known=0.f,possible=0.f,best_vis=0.f,best_unc=1.f,frustum_poses=0.f,best_blk=1.f,mind=11.f,center_poses=0.f;
  float best_udepth=10.f,best_umax=10.f,possible_poses=0.f,known_poses=0.f;
  int npose=query_count[q];
  for(int qq=0;qq<npose;qq++){
    int p=pose_list[query_start[q]+qq];const float* c=cams+p*12;
    float ex=cx-c[0],ez=cz-c[2];float hd=sqrtf(ex*ex+ez*ez);if(hd>10.5f)continue;
    {float ey=h+.7f-c[1];float lx=c[3]*ex+c[6]*ey+c[9]*ez,ly=c[4]*ex+c[7]*ey+c[10]*ez,lz=c[5]*ex+c[8]*ey+c[11]*ez;float fw=-lz;
     if(fw<=0.f||fabsf(lx)>fw+.5f)continue;
     if(fabsf(lx)<=fw&&fabsf(ly)<=fw&&hd<=10.f)center_poses+=1.f;}
    int vis=0,unc=0,blk=0;bool inside=true;float usum=0.f,umax=0.f;
    for(int k=0;k<63;k++){
      int m=k%21;int level=k/21;float px,pz;
      if(m==0){px=cx;pz=cz;}
      else if(m<5){float sx=(m==1||m==2)?-.49f:.49f,sz=(m==1||m==3)?-.49f:.49f;px=cx+sx*res;pz=cz+sz*res;}
      else{float ang=6.283185307f*(m-5)/16.f;px=cx+cosf(ang)*halo;pz=cz+sinf(ang)*halo;}
      float py=h+(level==0?.025f:(level==1?.7f:1.4f));
      float dx=px-c[0],dy=py-c[1],dz=pz-c[2];float d=sqrtf(dx*dx+dy*dy+dz*dz);
      float lx=c[3]*dx+c[6]*dy+c[9]*dz,ly=c[4]*dx+c[7]*dy+c[10]*dz,lz=c[5]*dx+c[8]*dy+c[11]*dz;
      float fw=-lz;
      if(!(fw>1e-6f&&fabsf(lx)<=fw&&fabsf(ly)<=fw&&d<=MAXR)){inside=false;break;}
      int j=(int)floorf((lx/fw+1.f)*.5f*size),ii=(int)floorf((1.f-ly/fw)*.5f*size);
      j=min(max(j,0),size-1);ii=min(max(ii,0),size-1);
      float2 r=images[((long long)p*size+ii)*size+j];
      if(r.x<d-tol_occ)blk++;else if(r.y<d-tol_unk)unc++;else vis++;
      float ud=fmaxf(d-r.y,0.f);usum+=ud;umax=fmaxf(umax,ud);
    }
    if(!inside)continue;
    frustum_poses+=1.f;mind=fminf(mind,hd);
    best_blk=fminf(best_blk,blk/63.f);
    if(blk==0){possible=1.f;possible_poses+=1.f;best_unc=fminf(best_unc,unc/63.f);if(unc==0){known=1.f;known_poses+=1.f;}
               best_udepth=fminf(best_udepth,usum/63.f);best_umax=fminf(best_umax,umax);}
    best_vis=fmaxf(best_vis,vis/63.f);
  }
  float inv=1.f/(float)max(npose,1);
  o[15*plane]=fminf(best_udepth,10.f)/10.f;o[16*plane]=fminf(best_umax,10.f)/10.f;
  o[18*plane]=possible_poses*inv;o[19*plane]=known_poses*inv;
  o[0]=known;o[plane]=possible;o[2*plane]=best_vis;o[3*plane]=frustum_poses>0.f?best_unc:1.f;
  o[4*plane]=frustum_poses*inv;o[5*plane]=best_blk;o[6*plane]=mind/10.f;o[7*plane]=center_poses*inv;
}

// Wall rays rasterized into the network input at channel offset c0
// (0 known hit, 1 hit after unknown, 2 unknown traversal, 3 free traversal).
extern "C" __global__ void wall_frame(const float* cams,const int* pose_query,const float* pose_y,int poses,
    Vol v,const unsigned char* state,const int* frames,int Hp,int Wp,int C,int c0,float* x,float gap,float offset,
    unsigned int* traversal,int Q){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=poses*105)return;
  int p=t/105,k=t%105;int a=k/3,b=k%3;int q=pose_query[p];
  const int* fr=frames+4*q;
  float lx=-.95f+1.9f*a/34.f,ly=(b==0?-.5f:(b==1?0.f:.5f)),lz=-1.f;
  const float* c=cams+p*12;
  float dx=c[3]*lx+c[4]*ly+c[5]*lz,dy=c[6]*lx+c[7]*ly+c[8]*lz,dz=c[9]*lx+c[10]*ly+c[11]*lz;
  float n=rsqrtf(dx*dx+dy*dy+dz*dz);dx*=n;dy*=n;dz*=n;
  long long plane=(long long)Hp*Wp;float* base=x+((long long)q*C+c0)*plane;
  float ylo=pose_y[p]+.2f,yhi=pose_y[p]+1.4f;
  float ox=c[0],oy=c[1],oz=c[2];
  float gx=(ox-v.x0)/v.res,gz=(oz-v.z0)/v.res,gy=(oy-v.y0)/v.dy;
  int ix=(int)floorf(gx),iz=(int)floorf(gz),iy=(int)floorf(gy);
  int sx=dx>0?1:-1,sz=dz>0?1:-1,sy=dy>0?1:-1;
  float inv_x=fabsf(dx)>1e-9f?1.f/fabsf(dx):1e30f,inv_z=fabsf(dz)>1e-9f?1.f/fabsf(dz):1e30f,inv_y=fabsf(dy)>1e-9f?1.f/fabsf(dy):1e30f;
  float nx_=dx>0?(ix+1-gx):(gx-ix),nz_=dz>0?(iz+1-gz):(gz-iz),ny_=dy>0?(iy+1-gy):(gy-iy);
  float tx=nx_*v.res*inv_x,tz=nz_*v.res*inv_z,ty=ny_*v.dy*inv_y;
  float ddx=v.res*inv_x,ddz=v.res*inv_z,ddy=v.dy*inv_y;
  float tt=0.f,ulen=0.f,t_occ=-1.f;bool entered=false;
  for(int step=0;step<4096;step++){
    if(ix<0||iz<0||iy<0||ix>=v.nx||iz>=v.nz||iy>=v.ny)break;
    unsigned char s=state[vindex(v,ix,iz,iy)];
    if(s==3)return;               // current human: ray consumed, no static certificate
    if(s==2){t_occ=tt;break;}
    float tn;int axis;
    if(tx<tz&&tx<ty){tn=tx;axis=0;}else if(tz<ty){tn=tz;axis=1;}else{tn=ty;axis=2;}
    if(s==0&&tn<=v.near)s=1;        // own robot/sensor volume
    float seg=fminf(tn,MAXR)-tt;
    float ymid=oy+dy*(tt+.5f*seg);
    if(seg>0.f&&ymid>ylo&&ymid<yhi){
      int fa=ix/v.sub-fr[0],fb=iz/v.sub-fr[1];
      if(fa>=0&&fb>=0&&fa<fr[2]&&fb<fr[3]){
        // Deterministic fixed-point (1/4096 m) accumulation.
        long long cell=(long long)fa*Wp+fb;unsigned int units=(unsigned int)lrintf(seg*4096.f);
        atomicAdd(traversal+((long long)q*2+(s==0?0:1))*plane+cell,units);
        // First entry of this ray into UNKNOWN space (wall band).
        if(s==0&&!entered){entered=true;atomicAdd(traversal+(2LL*Q+q)*plane+cell,1u);}
        // Expected first-hit density in UNKNOWN space under an exponential free
        // path (1 m) in unknown length: P(survive so far)*P(hit in segment).
        if(s==0){float w=__expf(-ulen)*(1.f-__expf(-seg));
          atomicAdd(traversal+(4LL*Q+q)*plane+cell,(unsigned int)lrintf(w*4096.f));}
      }
    }
    if(s==0)ulen+=seg;
    tt=tn;
    if(axis==0){tx+=ddx;ix+=sx;}else if(axis==1){tz+=ddz;iz+=sz;}else{ty+=ddy;iy+=sy;}
    if(tt>MAXR)break;
  }
  if(t_occ<0.f||t_occ>=MAXR)return;
  float th=t_occ+offset;float px=ox+dx*th,py=oy+dy*th,pz=oz+dz*th;
  if(!(py>ylo&&py<yhi))return;
  int channel=ulen<gap?0:1;
  int kx=(int)floorf(px/v.cres),kz=(int)floorf(pz/v.cres);
  int lox=(int)lrintf(v.x0/v.cres),loz=(int)lrintf(v.z0/v.cres);
  int span=(int)ceilf(.45f/v.cres)+1;
  for(int ax=-span;ax<=span;ax++)for(int az=-span;az<=span;az++){
    float cx=(kx+ax+.5f)*v.cres-px,cz=(kz+az+.5f)*v.cres-pz;float d2=cx*cx+cz*cz;
    if(d2>.2025f)continue;
    int fa=kx+ax-lox-fr[0],fb=kz+az-loz-fr[1];if(fa<0||fb<0||fa>=fr[2]||fb>=fr[3])continue;
    long long cell=(long long)fa*Wp+fb;
    // Distance (1e-4 m) from the cell center to the nearest known simulated hit.
    if(channel==0)atomicMin(traversal+(3LL*Q+q)*plane+cell,(unsigned int)lrintf(sqrtf(d2)*10000.f));
    if(d2<=.0625f)base[channel*plane+cell]=1.f;
  }
}
// Sub-voxel surface statistics: per-voxel sum of measured hit points + count
// (fixed point, deterministic). Used to refine simulated wall-ray hits.
extern "C" __global__ void hit_sums(const float* ranges,const float* cams,int frames,int size,
    Vol v,long long* sums,const float* humans,const int* hstart,float hradius){
  long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  long long total=(long long)frames*size*size;if(t>=total)return;
  int f=t/(size*size);int pix=t%(size*size);int i=pix/size,j=pix%size;
  const float* c=cams+f*12;float a[4];
  if(!pixel_point(ranges,c,f,size,i,j,a))return;
  if(near_human(a,humans,hstart,f,hradius))return;
  int ix=(int)floorf((a[0]-v.x0)/v.res),iz=(int)floorf((a[2]-v.z0)/v.res),iy=(int)floorf((a[1]-v.y0)/v.dy);
  if(ix<0||iz<0||iy<0||ix>=v.nx||iz>=v.nz||iy>=v.ny)return;
  long long* o=sums+4LL*vindex(v,ix,iz,iy);
  // offsets within the voxel in 1/65536 m
  atomicAdd((unsigned long long*)o,(unsigned long long)(long long)llrintf((a[0]-v.x0-ix*v.res)*65536.f));
  atomicAdd((unsigned long long*)(o+1),(unsigned long long)(long long)llrintf((a[1]-v.y0-iy*v.dy)*65536.f));
  atomicAdd((unsigned long long*)(o+2),(unsigned long long)(long long)llrintf((a[2]-v.z0-iz*v.res)*65536.f));
  atomicAdd((unsigned long long*)(o+3),1ULL);
}

// Wall rays with sub-voxel refinement: hit distance = projection of the
// voxel's mean measured hit point onto the ray, clamped to the voxel span.
extern "C" __global__ void wall_rays_refined(const float* cams,int poses,Vol v,const unsigned char* state,const long long* sums,float* out){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=poses*105)return;
  int p=t/105,k=t%105;int a=k/3,b=k%3;
  float lx=-.95f+1.9f*a/34.f,ly=(b==0?-.5f:(b==1?0.f:.5f)),lz=-1.f;
  const float* c=cams+p*12;
  float dx=c[3]*lx+c[4]*ly+c[5]*lz,dy=c[6]*lx+c[7]*ly+c[8]*lz,dz=c[9]*lx+c[10]*ly+c[11]*lz;
  float n=rsqrtf(dx*dx+dy*dy+dz*dz);dx*=n;dy*=n;dz*=n;
  float ox=c[0],oy=c[1],oz=c[2];
  float gx=(ox-v.x0)/v.res,gz=(oz-v.z0)/v.res,gy=(oy-v.y0)/v.dy;
  int ix=(int)floorf(gx),iz=(int)floorf(gz),iy=(int)floorf(gy);
  int sx=dx>0?1:-1,sz=dz>0?1:-1,sy=dy>0?1:-1;
  float inv_x=fabsf(dx)>1e-9f?1.f/fabsf(dx):1e30f,inv_z=fabsf(dz)>1e-9f?1.f/fabsf(dz):1e30f,inv_y=fabsf(dy)>1e-9f?1.f/fabsf(dy):1e30f;
  float nx_=dx>0?(ix+1-gx):(gx-ix),nz_=dz>0?(iz+1-gz):(gz-iz),ny_=dy>0?(iy+1-gy):(gy-iy);
  float tx=nx_*v.res*inv_x,tz=nz_*v.res*inv_z,ty=ny_*v.dy*inv_y;
  float ddx=v.res*inv_x,ddz=v.res*inv_z,ddy=v.dy*inv_y;
  float tt=0.f,ulen=0.f,thit=1e9f,tunk=1e9f;
  for(int step=0;step<4096;step++){
    if(ix<0||iz<0||iy<0||ix>=v.nx||iz>=v.nz||iy>=v.ny)break;
    int vi=vindex(v,ix,iz,iy);unsigned char s=state[vi];
    float tn=fminf(tx,fminf(tz,ty));
    if(s==2){
      const long long* q=sums+4LL*vi;long long cnt=q[3];
      if(cnt>0){
        float mx=v.x0+ix*v.res+(float)q[0]/(65536.f*cnt),my=v.y0+iy*v.dy+(float)q[1]/(65536.f*cnt),mz=v.z0+iz*v.res+(float)q[2]/(65536.f*cnt);
        float tp=(mx-ox)*dx+(my-oy)*dy+(mz-oz)*dz;thit=fminf(fmaxf(tp,tt),tn);
      }else thit=tt+.03f;
      break;
    }
    if(s==0){if(tunk>1e8f)tunk=tt;ulen+=fminf(tn,MAXR)-tt;}
    if(tx<tz&&tx<ty){tt=tx;tx+=ddx;ix+=sx;}else if(tz<ty){tt=tz;tz+=ddz;iz+=sz;}else{tt=ty;ty+=ddy;iy+=sy;}
    if(tt>MAXR)break;
  }
  float* o=out+t*6;o[0]=thit<MAXR?thit-.03f:1e9f;o[1]=tunk;o[2]=dx;o[3]=dy;o[4]=dz;o[5]=ulen;
}
'''


class _Module:
    _cache = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, device):
        device = torch.device(device)
        index = torch.cuda.current_device() if device.index is None else device.index
        with cls._lock:
            if index not in cls._cache:
                cls._cache[index] = cls(torch.device('cuda', index))
            return cls._cache[index]

    def __init__(self, device):
        self.device = device
        self.functions = {}
        self.driver = ct.CDLL('libcuda.so.1')
        with torch.cuda.device(device):
            torch.empty(1, device=device)
            cap = torch.cuda.get_device_capability(device)
            options = [('--gpu-architecture=compute_%d%d' % cap).encode(), b'--use_fast_math', b'--std=c++11']
            identity = hashlib.sha256(SOURCE.encode()+b'\0'.join(options)).hexdigest()
            cache = Path(os.environ.get('NUMBA_CACHE_DIR', tempfile.gettempdir()))/'claude_cuda'
            cache.mkdir(parents=True, exist_ok=True)
            path = cache/(identity+'.ptx')
            if not path.exists():
                lib = ct.CDLL(str(Path(sys.prefix)/'lib/libnvrtc.so'))
                program = ct.c_void_p()
                checked(lib, 'nvrtcCreateProgram', [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_void_p],
                        ct.byref(program), SOURCE.encode(), b'claude.cu', 0, None, None)
                try:
                    args = (ct.c_char_p*len(options))(*options)
                    try:
                        checked(lib, 'nvrtcCompileProgram', [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)], program, len(options), args)
                    except RuntimeError:
                        size = ct.c_size_t()
                        checked(lib, 'nvrtcGetProgramLogSize', [ct.c_void_p, ct.POINTER(ct.c_size_t)], program, ct.byref(size))
                        buf = ct.create_string_buffer(size.value)
                        checked(lib, 'nvrtcGetProgramLog', [ct.c_void_p, ct.c_void_p], program, buf)
                        raise RuntimeError(buf.value.decode())
                    size = ct.c_size_t()
                    checked(lib, 'nvrtcGetPTXSize', [ct.c_void_p, ct.POINTER(ct.c_size_t)], program, ct.byref(size))
                    buf = ct.create_string_buffer(size.value)
                    checked(lib, 'nvrtcGetPTX', [ct.c_void_p, ct.c_void_p], program, buf)
                    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
                        stream.write(buf.raw)
                        temporary = stream.name
                    os.replace(temporary, path)
                finally:
                    checked(lib, 'nvrtcDestroyProgram', [ct.POINTER(ct.c_void_p)], ct.byref(program))
            self.module = ct.c_void_p()
            checked(self.driver, 'cuModuleLoadData', [ct.POINTER(ct.c_void_p), ct.c_void_p], ct.byref(self.module),
                    ct.create_string_buffer(path.read_bytes()))

    def launch(self, name, total, values, threads=128):
        if not total:
            return
        if torch.cuda.current_device() != self.device.index:
            with torch.cuda.device(self.device):
                return self.launch(name, total, values, threads)
        if name not in self.functions:
            function = ct.c_void_p()
            checked(self.driver, 'cuModuleGetFunction', [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p],
                    ct.byref(function), self.module, name.encode())
            self.functions[name] = function
        args = []
        for v in values:
            if isinstance(v, torch.Tensor):
                args.append(ct.c_void_p(v.data_ptr()))
            elif v is None:
                args.append(ct.c_void_p(0))
            else:
                args.append(v)
        pointers = (ct.c_void_p*len(args))(*[ct.addressof(v) for v in args])
        stream = ct.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
        checked(self.driver, 'cuLaunchKernel', [ct.c_void_p, *([ct.c_uint]*7), ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p],
                self.functions[name], (total+threads-1)//threads, 1, 1, threads, 1, 1, 0, stream, pointers, None)


class Vol(ct.Structure):
    _fields_ = [('x0', ct.c_float), ('z0', ct.c_float), ('y0', ct.c_float), ('res', ct.c_float), ('dy', ct.c_float),
                ('nx', ct.c_int), ('nz', ct.c_int), ('ny', ct.c_int), ('cres', ct.c_float), ('sub', ct.c_int), ('near', ct.c_float)]


def module(device):
    return _Module.get(device)
