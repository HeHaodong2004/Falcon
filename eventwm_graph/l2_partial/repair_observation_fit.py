"""One bounded train-only fit of the persistent approximate optical law.

One canonical executed query per independent root; exact center matching
creates OFFLINE detection labels. Truth never enters the runtime sensor model.
The fitted law uses only relative position/base heading and a persistent regime.
"""
import argparse,hashlib,json,pathlib,time
import numpy as np
from scipy.optimize import minimize,linear_sum_assignment
from scipy.special import logsumexp
from .data import rows,root_data
from .observation import EndpointObservationLaw
from .repair_events import _json,_publish,_sha

def extract(snapshot,split):
    if split not in ('train','dev'):raise ValueError('optical fitting is train/dev only')
    gate=[];visible=[];root_index=[];ids=[];scenes=[];query_ids=[];unmatched=0;started=time.perf_counter()
    law=EndpointObservationLaw();law.in_view=(1.,1.);law.out_view=(0.,0.)
    for i,(meta,folder) in enumerate(rows(snapshot,split)):
        data=root_data(folder,True)
        qi=max(range(len(data['queries'])),key=lambda j:(len(data['queries'][j]['compiled']['actions']),data['queries'][j]['query_id']))
        q=data['queries'][qi];outcome=data['outcomes'][qi];n=min(len(outcome['node_ids']),62,len(data['times'])-1)
        ids.append(meta['root_id']);scenes.append(meta['scene_id']);query_ids.append(q['query_id'])
        if n and data['truth'].shape[1]:
            y=np.zeros((n,data['truth'].shape[1]));poses=[]
            for j,nid in enumerate(outcome['node_ids'][:n]):
                obs=data['nodes'][nid]['observation'];poses.append(obs['robot_pose'])
                if abs(float(obs['timestamp'])-data['times'][j+1])>1e-6:raise ValueError('optical supervision clock')
                detected=[r['position'] for r in obs['tracks'] if r['visible_now'] and r.get('valid',True)]
                if detected:
                    d=np.linalg.norm(np.asarray(detected)[:,None,[0,2]]-data['truth'][None,j+1],axis=-1)
                    ii,jj=linear_sum_assignment(d);good=d[ii,jj]<1e-5
                    if not good.all() or len(ii)!=len(detected):unmatched+=len(detected)-int(good.sum())
                    y[j,jj[good]]=1
            g=law.detection(data['truth'][1:n+1],np.asarray(poses),0)
            gate.extend(g.ravel());visible.extend(y.ravel());root_index.extend([i]*y.size)
        if (i+1)%500==0:print(split,i+1,'optical extraction seconds',round(time.perf_counter()-started,2),flush=True)
    if unmatched:raise ValueError('unmatched exact optical labels: '+str(unmatched))
    return {'gate':np.asarray(gate),'visible':np.asarray(visible),'root':np.asarray(root_index,int),'ids':ids,'scenes':scenes,'query_ids':query_ids}

def decode(theta):
    lo,hi,farlo,farhi,pi=np.asarray(theta)
    return {'in_view':[float(lo),float(hi)],'out_view':[float(farlo),float(farhi)],'regime_prior':[float(pi),float(1-pi)]}

def scores(data,theta):
    pars=decode(theta);g=data['gate'];y=data['visible'];ri=data['root'];nr=len(data['ids']);ll=[]
    for lo,hi in zip(pars['out_view'],pars['in_view']):
        pd=np.clip(lo+(hi-lo)*g,1e-12,1-1e-12)
        ll.append(np.bincount(ri,weights=y*np.log(pd)+(1-y)*np.log1p(-pd),minlength=nr))
    joint=np.stack(ll)+np.log(pars['regime_prior'])[:,None]
    loge=logsumexp(joint,axis=0);post=np.exp(joint-loge[None])
    return loge,post

def fit(snapshot,output):
    output=pathlib.Path(output)
    if output.exists():raise FileExistsError(output)
    start=time.perf_counter();train=extract(snapshot,'train');dev=extract(snapshot,'dev')
    if set(train['scenes'])&set(dev['scenes']):raise ValueError('optical scene leakage')
    initial=np.array([.12,.85,.005,.025,.35])
    def objective(t):
        # A small fixed regularizer keeps both regimes supported. This is a
        # single predeclared optimization, not development hyperparameter search.
        if t[0]>t[1] or t[2]>t[0] or t[3]>t[1]:return 1e8+float(np.square(t-initial).sum())
        return -float(scores(train,t)[0].mean())+.01*float(np.square(t-initial).sum())
    result=minimize(objective,initial,method='L-BFGS-B',bounds=[(.02,.70),(.20,.98),(.0001,.04),(.0001,.15),(.05,.95)],options={'maxiter':60,'ftol':1e-8,'maxls':30})
    theta=result.x if objective(result.x)<=objective(initial) else initial
    value={'schema':'persistent_endpoint_optical_fit_1','parameters':decode(theta),'initial_parameters':decode(initial),
        'fit_success':bool(result.success),'fit_message':str(result.message),'iterations':int(result.nit),'seed':'deterministic L-BFGS-B',
        'train_roots':len(train['ids']),'train_scenes':len(set(train['scenes'])),'dev_roots':len(dev['ids']),'dev_scenes':len(set(dev['scenes'])),
        'train_person_endpoints':len(train['gate']),'dev_person_endpoints':len(dev['gate']),
        'train_initial_root_nll':float(-scores(train,initial)[0].mean()),'train_fitted_root_nll':float(-scores(train,theta)[0].mean()),
        'dev_initial_root_nll':float(-scores(dev,initial)[0].mean()),'dev_fitted_root_nll':float(-scores(dev,theta)[0].mean()),
        'snapshot_manifest_sha256':_sha(pathlib.Path(snapshot)/'manifest.json'),'source_sha256':{p.name:_sha(p) for p in [pathlib.Path(__file__),pathlib.Path(__file__).with_name('observation.py')]},
        'scope':'one legal compiled query per root, max62 arrived native endpoints, offline exact visibility labels; no model selection on old holdout',
        'limitations':'global persistent regime approximates camera/mesh occlusion; sensor-law miss integration and belief closure remain approximate',
        'elapsed_s':time.perf_counter()-start}
    _json(output,value)
    for name,data in [('train',train),('dev',dev)]:
        p=output.with_name(output.stem+'_'+name+'_labels.npz')
        _publish(p,lambda f,d=data:np.savez_compressed(f,gate=d['gate'],visible=d['visible'],root=d['root'],root_ids=np.asarray(d['ids']),scene_ids=np.asarray(d['scenes'])))
    print(json.dumps(value,indent=2),flush=True);return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('snapshot');p.add_argument('output');a=p.parse_args();fit(a.snapshot,a.output)
