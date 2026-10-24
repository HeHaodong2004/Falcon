"""Native first-event likelihood supervision on shared latent world mixtures."""
import argparse,json,pathlib,time,os
import numpy as np
import torch
from .data import rows,root_data,read
from .model import GraphBeliefWorldModel
from .neural import LearnedDynamics
from .queries import event_features,NativeEventModel,FEATURE_NAMES

def prepare(snapshot,split,checkpoint,population,output):
    prior=json.loads(pathlib.Path(population).read_text());wm=GraphBeliefWorldModel(dynamics=LearnedDynamics.load(checkpoint),population_prior=prior)
    records=[];start=time.perf_counter()
    for ri,(meta,folder) in enumerate(rows(snapshot,split)):
        d=root_data(folder,True);labels=read(folder/'privileged_step_labels.json.gz');b=wm.initialize(d['legal']['history'],d['legal']['partial_graph'])
        times=d['times'][d['times']<=b.timestamp+3.+1e-8];f=wm.predict_shared(b,times[-1]-b.timestamp,16,seed=31,times=times-b.timestamp)
        primary=max(b.hypotheses,key=lambda h:h.weight);known=set()
        if primary.n and d['truth'].shape[1]:
            from scipy.optimize import linear_sum_assignment
            dist=np.linalg.norm(primary.mean[:,None,:2]-d['truth'][None,0],axis=-1);ii,jj=linear_sum_assignment(dist)
            known={int(j) for i,j in zip(ii,jj) if dist[i,j]<1.5}
        root_records=[];hidden=np.array([h.expected_hidden for h in f.worlds]);total=hidden+np.array([h.n for h in f.worlds])
        age=np.mean([max(0,b.timestamp-h.last_seen.mean()) if h.n else 0 for h in f.worlds])
        for q,o in zip(d['queries'],d['outcomes']):
            n=min(len(o['node_ids']),len(times)-1)
            if not n:continue
            poses=np.asarray(q['compiled']['planned_poses'][:n+1]);x=event_features(f.positions[:,:n+1],f.velocities[:,:n+1],poses,hidden,age,times[:n+1]-b.timestamp)[:,1:]
            event=np.zeros((n,3),bool)
            for j,nid in enumerate(o['node_ids'][:n]):
                lab=labels[nid];event[j,0]=lab['human_mesh_contact'];event[j,1]=lab['psc_violation']
                for track in d['nodes'][nid]['observation']['tracks']:
                    if not track['visible_now'] or not d['truth'].shape[1]:continue
                    center=np.asarray(track['position'])[[0,2]];dist=np.linalg.norm(d['truth'][j+1]-center,axis=1);identity=int(dist.argmin())
                    if dist[identity]<1e-5 and identity not in known:event[j,2]=True
            first=np.array([int(np.flatnonzero(event[:,k])[0]) if event[:,k].any() else -1 for k in range(3)])
            allowed=np.stack([total>0,total>0,hidden>0],-1)
            root_records.append({'root_id':meta['root_id'],'scene_id':meta['scene_id'],'query_id':q['query_id'],'x':x.astype(np.float16),
                'allowed':allowed,'weights':f.weights,'unresolved':f.unresolved_mass,'first':first,'steps':n,'root_weight':0.})
        for r in root_records:r['root_weight']=1/len(root_records)
        records.extend(root_records)
        if (ri+1)%200==0:print(split,ri+1,len(records),'seconds',round(time.perf_counter()-start),flush=True)
    np.save(output,np.asarray(records,dtype=object),allow_pickle=True);print('saved',len(records),output,flush=True)

def batch(records):
    B=len(records);W=16;T=max(r['steps'] for r in records);F=len(FEATURE_NAMES)
    x=np.zeros((B,W,T,F),np.float32);mask=np.zeros((B,T,3),np.float32);first=np.zeros_like(mask);weights=np.zeros((B,W),np.float32);allowed=np.zeros((B,W,3),np.float32);rw=np.zeros(B,np.float32);u=np.zeros(B,np.float32)
    for i,r in enumerate(records):
        n=r['steps'];x[i,:,:n]=r['x'];weights[i]=r['weights'];allowed[i]=r['allowed'];rw[i]=r['root_weight'];u[i]=r['unresolved']
        for k,at in enumerate(r['first']):
            mask[i,:(at+1 if at>=0 else n),k]=1
            if at>=0:first[i,at,k]=1
    return [torch.from_numpy(v) for v in (x,mask,first,weights,allowed,rw,u)]

def likelihood(coef,b):
    x,mask,y,weights,allowed,rw,u=b;logits=x@coef.T
    hazard=torch.sigmoid(logits)*allowed[:,:,None,:]
    # Impossible structural events remain zero, epsilon is numerical likelihood
    # floor, not a physical event probability.
    logl=(y[:,None]*hazard.clamp(min=1e-10).log()+(1-y[:,None])*torch.log1p(-hazard.clamp(max=1-1e-7)))*mask[:,None]
    logl=logl.sum((2,3));joint=torch.logsumexp(logl+weights.clamp(min=1e-30).log(),-1)
    residual=((y*np.log(.005)+(1-y)*np.log(.995))*mask).sum((1,2))+u.clamp(min=1e-30).log()
    lp=torch.logaddexp(joint,residual);return -(lp*rw).sum()/rw.sum()

def probabilities(coef,record):
    x=np.asarray(record['x'],float);hz=1/(1+np.exp(-np.clip(x@coef.T,-30,30)));hz*=record['allowed'][:,None,:]
    hc,hp,ha=hz[...,0],hz[...,1],hz[...,2];before=np.c_[np.ones(len(hc)),np.cumprod(1-hc,axis=1)[:,:-1]]
    psc=(before*np.c_[np.ones(len(hp)),np.cumprod(1-hp,axis=1)[:,:-1]]*hp).sum(1)
    values=np.array([record['weights']@(1-np.prod(1-hc,axis=1)),record['weights']@psc,record['weights']@(1-np.prod(1-ha,axis=1))])
    n=record['steps'];u=record['unresolved'];values+=u*np.array([1-.995**n,float((.995**(2*np.arange(n))*.005).sum()),1-.995**n])
    return values

def fit(train,dev,output,epochs=10):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","2")));torch.manual_seed(59);rng=np.random.default_rng(59)
    tr=np.load(train,allow_pickle=True).tolist();dv=np.load(dev,allow_pickle=True).tolist();coef=torch.nn.Parameter(torch.tensor(NativeEventModel().coefficients,dtype=torch.float32));opt=torch.optim.Adam([coef],lr=.025)
    output=pathlib.Path(output);history=[];best=1e30;bestcoef=None
    for epoch in range(epochs+1):
        vals=[]
        if epoch:
            for ids in np.array_split(rng.permutation(len(tr)),max(1,int(np.ceil(len(tr)/48)))):
                opt.zero_grad();loss=likelihood(coef,batch([tr[i] for i in ids]))+.0005*(coef[:,1:]**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_([coef],10);opt.step();vals.append(float(loss.detach()))
        with torch.no_grad():devloss=float(np.mean([float(likelihood(coef,batch(dv[j:j+48]))) for j in range(0,len(dv),48)]))
        row={'epoch':epoch,'train_loss':float(np.mean(vals)) if vals else None,'dev_loss':devloss};history.append(row);print(json.dumps(row),flush=True)
        if devloss<best:best=devloss;bestcoef=coef.detach().numpy().copy()
    # Three bounded scalar intercept calibrations; minimizes native dev Brier.
    # This modifies the shared hazard law itself, including survival updates.
    from scipy.optimize import minimize_scalar
    offsets=[]
    for k in range(3):
        def score(offset):
            c=bestcoef.copy();c[k,0]+=offset;num=den=0.
            for r in dv:
                pred=probabilities(c,r)[k];w=r['root_weight'];num+=w*(pred-float(r['first'][k]>=0))**2;den+=w
            return num/den
        found=minimize_scalar(score,bounds=(-2.,2.),method='bounded',options={'xatol':.025});bestcoef[k,0]+=found.x;offsets.append(float(found.x))
    value={'coefficients':bestcoef.tolist(),'features':FEATURE_NAMES,'intercept_offsets':offsets,'history':history,
           'training_queries':len(tr),'dev_queries':len(dv),'seed':59,'semantics':'native first-event shared-mixture joint likelihood; root-balanced; dev hazard-intercept Brier calibration',
           'appearance_target':'person not matched to persistent root tracks detected at least once; offline identity matching proxy'}
    output.write_text(json.dumps(value,indent=2));print('saved',output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('prepare');a.add_argument('snapshot');a.add_argument('split');a.add_argument('checkpoint');a.add_argument('population');a.add_argument('output')
    a=s.add_parser('fit');a.add_argument('train');a.add_argument('dev');a.add_argument('output');a.add_argument('--epochs',type=int,default=10)
    args=vars(p.parse_args());cmd=args.pop('cmd');(prepare if cmd=='prepare' else fit)(**args)
