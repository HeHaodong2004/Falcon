"""Diagnostic (TRAIN validation): does a pocket-aware unknown length along the
future probe rays carry information beyond the frozen model's prediction?

The production render stores, per pixel, the first OCCUPIED and first UNKNOWN
distance, so a probe behind the first unknown voxel counts as uncertain with
unknown length d - t_unk. Here an experimental render also records where the
ray re-enters known FREE space (t_re) and the next UNKNOWN entry (t_unk2); the
probe's unknown length is then (min(d, t_re) - t_unk) + max(d - t_unk2, 0).
Within the stratum of cells whose probes are possible but cross unknown space,
it reports how well each statistic separates revealed from unrevealed cells
(AUC) and whether a logistic model on [model logit, new statistics] beats one
on the model logit alone (held-out Roots).
"""
import argparse
import ctypes as ct

import numpy as np
import torch

from . import kernels
from .data import split_roots, load_root
from .features import channel_index, build_inputs, crop_box, MAP_CHANNELS, QUERY_CHANNELS
from .geometry import proxy_cameras, pack_cams, upload
from .model import GraphRunner, build_model
from .strata import stratum, STRATA
from .train import build_context, frame_labels

EXTRA = r'''
__device__ void march4(const Vol& v,const unsigned char* state,float ox,float oy,float oz,
    float dx,float dy,float dz,float tmax,float4* r){
  float t_occ=1e9f,t_unk=1e9f,t_re=1e9f,t_unk2=1e9f;
  float gx=(ox-v.x0)/v.res,gz=(oz-v.z0)/v.res,gy=(oy-v.y0)/v.dy;
  int ix=(int)floorf(gx),iz=(int)floorf(gz),iy=(int)floorf(gy);
  int sx=dx>0?1:-1,sz=dz>0?1:-1,sy=dy>0?1:-1;
  float inv_x=fabsf(dx)>1e-9f?1.f/fabsf(dx):1e30f,inv_z=fabsf(dz)>1e-9f?1.f/fabsf(dz):1e30f,inv_y=fabsf(dy)>1e-9f?1.f/fabsf(dy):1e30f;
  float nx_=dx>0?(ix+1-gx):(gx-ix),nz_=dz>0?(iz+1-gz):(gz-iz),ny_=dy>0?(iy+1-gy):(gy-iy);
  float tx=nx_*v.res*inv_x,tz=nz_*v.res*inv_z,ty=ny_*v.dy*inv_y;
  float ddx=v.res*inv_x,ddz=v.res*inv_z,ddy=v.dy*inv_y;
  float t=0.f;
  for(int step=0;step<4096;step++){
    if(ix<0||iz<0||iy<0||ix>=v.nx||iz>=v.nz||iy>=v.ny)break;
    unsigned char s=state[vindex(v,ix,iz,iy)];
    if(s>=2){t_occ=t;break;}
    float tn;int axis;
    if(tx<tz&&tx<ty){tn=tx;axis=0;}else if(tz<ty){tn=tz;axis=1;}else{tn=ty;axis=2;}
    bool unk=(s==0&&tn>v.near);
    if(unk){float t0=fmaxf(t,v.near);if(t_unk>1e8f)t_unk=t0;else if(t_re<1e8f&&t_unk2>1e8f)t_unk2=t0;}
    else if(t_unk<1e8f&&t_re>1e8f)t_re=t;
    t=tn;
    if(axis==0){tx+=ddx;ix+=sx;}else if(axis==1){tz+=ddz;iz+=sz;}else{ty+=ddy;iy+=sy;}
    if(t>tmax)break;
  }
  *r=make_float4(t_occ,t_unk,t_re,t_unk2);
}

extern "C" __global__ void render4(const float* cams,int poses,int size,Vol v,const unsigned char* state,float4* out){
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=poses*size*size)return;
  int p=t/(size*size);int pix=t%(size*size);int i=pix/size,j=pix%size;
  float lx=(j+.5f)*2.f/size-1.f,ly=1.f-(i+.5f)*2.f/size,lz=-1.f;
  const float* c=cams+p*12;
  float dx=c[3]*lx+c[4]*ly+c[5]*lz,dy=c[6]*lx+c[7]*ly+c[8]*lz,dz=c[9]*lx+c[10]*ly+c[11]*lz;
  float n=rsqrtf(dx*dx+dy*dy+dz*dz);dx*=n;dy*=n;dz*=n;
  march4(v,state,c[0],c[1],c[2],dx,dy,dz,MAXR+.5f,out+t);
}

// Per cell: best (over frustum poses without known blocker) fraction of probes whose
// pocket-aware unknown length is <= lim, min mean / max pocket-aware unknown length.
extern "C" __global__ void probe_pocket(const float* cams,const int* query_start,const int* query_count,
    const int* pose_list,int Q,int size,const float4* images,Vol v,const unsigned char* graph_known,
    const int* frames,const float* floor_h,float tol_occ,float tol_unk,float lim,int Hp,int Wp,float* out){
  long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(t>=(long long)Q*Hp*Wp)return;
  int q=t/((long long)Hp*Wp);int a=(t/Wp)%Hp,b=t%Wp;
  const int* fr=frames+4*q;if(a>=fr[2]||b>=fr[3])return;
  int cnx=v.nx/v.sub,cnz=v.nz/v.sub;
  int gx=fr[0]+a,gz=fr[1]+b;if(gx<0||gz<0||gx>=cnx||gz>=cnz)return;
  if(graph_known[(long long)gx*cnz+gz])return;
  long long plane=(long long)Hp*Wp;float* o=out+(long long)q*3*plane+(long long)a*Wp+b;
  int lox=(int)lrintf(v.x0/v.cres),loz=(int)lrintf(v.z0/v.cres);
  float res=v.cres,cx=(lox+gx+.5f)*res,cz=(loz+gz+.5f)*res,h=floor_h[q];
  float halo=.25f+res*0.70710678f;
  float best_pv=0.f,best_mean=10.f,best_max=10.f;
  int npose=query_count[q];
  for(int qq=0;qq<npose;qq++){
    int p=pose_list[query_start[q]+qq];const float* c=cams+p*12;
    float ex=cx-c[0],ez=cz-c[2];float hd=sqrtf(ex*ex+ez*ez);if(hd>10.5f)continue;
    {float ey=h+.7f-c[1];float lx=c[3]*ex+c[6]*ey+c[9]*ez,lz=c[5]*ex+c[8]*ey+c[11]*ez;float fw=-lz;
     if(fw<=0.f||fabsf(lx)>fw+.5f)continue;}
    int blk=0,pv=0;bool inside=true;float lsum=0.f,lmax=0.f;
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
      float4 r=images[((long long)p*size+ii)*size+j];
      if(r.x<d-tol_occ){blk++;continue;}
      float L=0.f;
      if(r.y<d-tol_unk){L=fminf(d,r.z)-r.y;if(d>r.w)L+=d-r.w;}
      L=fmaxf(L,0.f);lsum+=L;lmax=fmaxf(lmax,L);if(L<=lim)pv++;
    }
    if(!inside||blk)continue;
    best_pv=fmaxf(best_pv,pv/63.f);best_mean=fminf(best_mean,lsum/63.f);best_max=fminf(best_max,lmax);
  }
  o[0]=best_pv;o[plane]=fminf(best_mean,10.f)/10.f;o[2*plane]=fminf(best_max,10.f)/10.f;
}
'''


class _Diag(kernels._Module):
    _cache = {}


def diag_module(device):
    original = kernels.SOURCE
    kernels.SOURCE = original+EXTRA
    try:
        return _Diag.get(device)
    finally:
        kernels.SOURCE = original


def pocket_features(ctx, paths, fr, Hp, Wp, lim=.15):
    """3 x (Q, Hp, Wp) pocket statistics on the same frames as build_inputs."""
    dev = ctx.device; m = diag_module(dev); q = len(paths)
    unique, pose_rows, pose_list, query_start, query_count = {}, [], [], [], []
    for p in paths:
        p = np.asarray(p, np.float64)
        query_start.append(len(pose_list)); query_count.append(len(p)-1)
        for row in p[1:]:
            key = row[1:].tobytes()
            if key not in unique:
                unique[key] = len(pose_rows); pose_rows.append(row)
            pose_list.append(unique[key])
    out = torch.zeros((q, 3, Hp, Wp), dtype=torch.float32, device=dev)
    if not pose_rows:
        return out
    rows = np.asarray(pose_rows, np.float64)
    oo, R = proxy_cameras(rows, ctx.extrinsic)
    cams = upload(pack_cams(oo, R).reshape(-1).astype(np.float32), dev)
    ints = upload(np.concatenate([fr.reshape(-1), query_start, query_count, pose_list]).astype(np.int32), dev)
    t_frames, t_qs, t_qc, t_pl = ints[:4*q], ints[4*q:5*q], ints[5*q:6*q], ints[6*q:]
    floor = upload(np.asarray([np.asarray(p)[0, 2] for p in paths], np.float32), dev)
    size = ctx.render; P = len(pose_rows)
    images = torch.empty((P, size, size, 4), dtype=torch.float32, device=dev)
    m.launch('render4', P*size*size, [cams, ct.c_int(P), ct.c_int(size), ctx.vm.vol, ctx.vm.state, images], 128)
    m.launch('probe_pocket', q*Hp*Wp, [cams, t_qs, t_qc, t_pl, ct.c_int(q), ct.c_int(size), images, ctx.vm.vol, ctx.graph_known,
                                         t_frames, floor, ct.c_float(.15 if ctx.sub == 1 else .09), ct.c_float(.1 if ctx.sub == 1 else .06),
                                         ct.c_float(lim), ct.c_int(Hp), ct.c_int(Wp), out], 128)
    return out


def auc(score, label):
    order = np.argsort(score, kind='stable'); s = score[order]; l = label[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j+1 < len(s) and s[j+1] == s[i]:
            j += 1
        ranks[i:j+1] = (i+j)/2+1; i = j+1
    pos = l.sum(); neg = len(l)-pos
    return (ranks[l == 1].sum()-pos*(pos+1)/2)/max(pos*neg, 1)


def logistic(Xtr, ytr, Xte, iters=200):
    """Plain logistic regression (standardized inputs, LBFGS) -> test probabilities."""
    mu, sd = Xtr.mean(0), Xtr.std(0)+1e-6
    A = torch.as_tensor((Xtr-mu)/sd, dtype=torch.float64); y = torch.as_tensor(ytr, dtype=torch.float64)
    w = torch.zeros(A.shape[1], dtype=torch.float64, requires_grad=True); b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(A@w+b, y); loss.backward(); return loss
    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        return torch.sigmoid(torch.as_tensor((Xte-mu)/sd, dtype=torch.float64)@w+b).numpy()


@torch.no_grad()
def run(checkpoint, device='cuda:0', roots=80, per_window=4000, seed=0):
    saved = torch.load(checkpoint, map_location='cpu'); config = saved['config']
    cidx = channel_index(config); cin = len(config['map_channels'])+len(config['query_channels'])
    model = build_model(cin, config); model.load_state_dict(saved['model'])
    runner = GraphRunner(model.to(device), device, batches=(1,), sizes=(240, 256, 272, 288, 304, 320, 352, 384), channels=cin)
    _, val = split_roots()
    val = sorted(val, key=lambda r: r['root_id'])[:roots]
    nm = len(MAP_CHANNELS); qi = {n: nm+i for i, n in enumerate(QUERY_CHANNELS)}
    rng = np.random.default_rng(seed)
    rows = []
    for ri, row in enumerate(val):
        for g in load_root(row, tracks=bool(config.get('human_filter'))):
            windows = [w for w in g['windows'] if len(w['path']) > 1]
            if not windows:
                continue
            ctx = build_context(g, device, config=config)
            boxes = [crop_box(w['path'], ctx.res) for w in windows]
            x, mask, fr = build_inputs(ctx, [w['path'] for w in windows], [(lo, hi-lo) for lo, hi in boxes])
            pk = pocket_features(ctx, [w['path'] for w in windows], fr, x.shape[2], x.shape[3])
            y = frame_labels(mask, fr, windows, ctx.lo)
            logits = runner(x[:, cidx] if cidx is not None else x, frames=fr[:, 2:])
            for i in range(len(windows)):
                st = stratum(x[i].float())
                sel = (y[i] >= 0) & (st == STRATA.index('probe_possible_uncertain'))
                idx = torch.nonzero(sel.flatten()).flatten()
                if not len(idx):
                    continue
                if len(idx) > per_window:
                    idx = idx[torch.as_tensor(rng.choice(len(idx), per_window, replace=False), device=idx.device)]
                f = lambda t: t.flatten()[idx].float().cpu().numpy()
                rows.append(np.stack([np.full(len(idx), ri), f(y[i] > 0), f(logits[i, 0]),
                                      f(x[i, qi['probe_best_uncertain']]), f(x[i, qi['probe_unknown_depth']]),
                                      f(x[i, qi['probe_unknown_depth_max']]), f(pk[i, 0]), f(pk[i, 1]), f(pk[i, 2])], 1))
    data = np.concatenate(rows)
    root, lab, logit = data[:, 0], data[:, 1].astype(int), data[:, 2]
    names = ['model_logit', 'probe_best_uncertain', 'probe_unknown_depth', 'probe_unknown_depth_max', 'pocket_visible_frac',
             'pocket_mean_len', 'pocket_max_len']
    sign = [1, -1, -1, -1, 1, -1, -1]
    print('cells', len(data), 'revealed rate %.3f' % lab.mean(), flush=True)
    for k, (n, s) in enumerate(zip(names, sign)):
        print('AUC %-26s %.4f' % (n, auc(s*data[:, 2+k], lab)), flush=True)
    # Held-out Roots: logistic model on the logit alone vs logit + pocket statistics.
    test = root % 4 == 0
    for cols, tag in (([2], 'logit'), ([2, 3, 4, 5], 'logit+existing probe stats'), ([2, 6, 7, 8], 'logit+pocket'),
                      ([2, 3, 4, 5, 6, 7, 8], 'logit+existing+pocket')):
        Xtr, Xte = data[~test][:, cols], data[test][:, cols]
        p = logistic(Xtr, lab[~test], Xte)
        ll = -np.mean(lab[test]*np.log(p+1e-9)+(1-lab[test])*np.log(1-p+1e-9))
        print('held-out %-30s AUC %.4f log-loss %.4f' % (tag, auc(p, lab[test]), ll), flush=True)


if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    pa.add_argument('--checkpoint', required=True)
    pa.add_argument('--device', default='cuda:0')
    pa.add_argument('--roots', type=int, default=80)
    run(**vars(pa.parse_args()))
