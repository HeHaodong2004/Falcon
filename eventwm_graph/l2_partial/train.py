"""Root-balanced multi-horizon proper likelihood; dev-only checkpoint selection."""
import argparse,json,pathlib,time,random,os
import numpy as np
import torch
from .neural import GraphDynamics

def batch(rows):
    B=len(rows);N=max(1,max(len(r['mean']) for r in rows));H=4
    arrays={k:np.zeros(shape,np.float32) for k,shape in {'mean':(B,N,4),'cov':(B,N,4,4),'age':(B,N),'graph':(B,N,18),'mask':(B,N),'target':(B,N,H,2),'valid':(B,N,H)}.items()}
    for i,r in enumerate(rows):
        n=len(r['mean']);arrays['mean'][i,:n]=r['mean'];arrays['cov'][i,:n]=r['covariance'];arrays['age'][i,:n]=r['age'];arrays['graph'][i,:n]=r['graph'];arrays['mask'][i,:n]=1
        arrays['target'][i,:n]=r['targets'];arrays['valid'][i,:n]=r['valid']
    return {k:torch.from_numpy(v) for k,v in arrays.items()}

def distribution(net,b,scale=1.):
    target,q,logits=net(b['mean'],b['age'],b['graph'],b['mask'])
    t=torch.tensor([.5,1.,2.,3.]);k=.7;e=torch.exp(-k*t);a=(1-e)/k
    mu=a[None,None,:,None,None]*b['mean'][:,:,None,None,2:]+(t-a)[None,None,:,None,None]*target[:,:,None]
    qpp=(t-2*a+(1-torch.exp(-2*k*t))/(2*k))/(k*k)
    F=torch.zeros(4,2,4);F[:,:2,:2]=torch.eye(2)[None];F[:,:,2:]=a[:,None,None]*torch.eye(2)[None]
    initial=torch.einsum('hij,bnjk,hlk->bnhil',F,b['cov'],F)
    initial=initial+(.0025*t)[None,None,:,None,None]*torch.eye(2)[None,None,None]
    cov=initial[:,:,:,None]+qpp[None,None,:,None,None,None]*q[:,:,None,:,None,None]*scale*torch.eye(2)[None,None,None,None]+torch.eye(2)[None,None,None,None]*1e-5
    return mu,cov,torch.log_softmax(logits,-1)

def objective(net,b):
    mu,cov,logpi=distribution(net,b);d=b['target'][:,:,:,None]-mu
    inv=torch.linalg.inv(cov);ld=torch.logdet(cov)
    component=-np.log(2*np.pi)-.5*ld-.5*torch.einsum('bnhki,bnhkij,bnhkj->bnhk',d,inv,d)
    nll=-torch.logsumexp(component+logpi[:,:,None,:],-1)
    counts=b['valid'].sum((1,2));scores=(nll*b['valid']).sum((1,2))/counts.clamp(min=1)
    return scores[counts>0].mean()

def score(net,rows):
    vals=[]
    with torch.no_grad():
        for j in range(0,len(rows),64):
            subset=rows[j:j+64];vals.append((len(subset),float(objective(net,batch(subset)))))
    return sum(n*v for n,v in vals)/sum(n for n,_ in vals)

def energy_score(net,rows,worlds=32):
    torch.manual_seed(113);values=[]
    with torch.no_grad():
        for j in range(0,len(rows),32):
            b=batch(rows[j:j+32]);mu,cov,logpi=distribution(net,b);B,N,H,K,_=mu.shape
            prob=logpi.exp()[:,:,None].expand(B,N,H,K).reshape(-1,K)
            ids=torch.multinomial(prob,worlds,replacement=True)
            means=mu.reshape(-1,K,2).gather(1,ids[:,:,None].expand(-1,-1,2))
            covs=cov.reshape(-1,K,2,2).gather(1,ids[:,:,None,None].expand(-1,-1,2,2))
            draws=means+(torch.linalg.cholesky(covs)@torch.randn(*means.shape,1)).squeeze(-1)
            truth=b['target'].reshape(-1,2);first=(draws-truth[:,None]).norm(dim=-1).mean(-1)
            second=torch.cdist(draws,draws).sum((-1,-2))/(worlds*(worlds-1));es=(first-.5*second).reshape(B,N,H)
            scores=(es*b['valid']).sum((1,2))/b['valid'].sum((1,2)).clamp(min=1);values.extend(scores.tolist())
    return float(np.mean(values))

def run(train,dev,output,epochs=30,seed=11,width=64,root_limit=0):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","2")));torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    train_rows=[r for r in np.load(train,allow_pickle=True) if r['valid'].any()];dev_rows=[r for r in np.load(dev,allow_pickle=True) if r['valid'].any()]
    if root_limit:train_rows=[train_rows[i] for i in np.random.default_rng(seed).permutation(len(train_rows))[:root_limit]]
    net=GraphDynamics(width);optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    output=pathlib.Path(output);output.mkdir(exist_ok=True,parents=True);history=[];best=float('inf');start=time.perf_counter()
    rng=np.random.default_rng(seed)
    for epoch in range(epochs+1):
        loss=[]
        if epoch:
            net.train()
            for ids in np.array_split(rng.permutation(len(train_rows)),max(1,int(np.ceil(len(train_rows)/48)))):
                b=batch([train_rows[i] for i in ids]);optimizer.zero_grad();value=objective(net,b);value.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step();loss.append(float(value.detach()))
        net.eval();val=score(net,dev_rows);es=energy_score(net,dev_rows);training=score(net,train_rows) if epoch%5==0 else float(np.mean(loss))
        row={'epoch':epoch,'train_nll':training,'dev_nll':val,'dev_energy_score':es,'training_roots':len(train_rows),'elapsed_s':time.perf_counter()-start};history.append(row);print(json.dumps(row),flush=True)
        if es<best:
            best=es;torch.save({'state_dict':net.state_dict(),'width':width,'seed':seed,'epoch':epoch,'dev_nll':val,'dev_energy_score':es,'diffusion_scale':1.,'horizons':[.5,1,2,3],'parameters':sum(p.numel() for p in net.parameters()),'independent_training_roots':len(train_rows)},output/'checkpoint.pt')
        (output/'learning.json').write_text(json.dumps(history,indent=2))
    metadata={'seed':seed,'epochs':epochs,'selected_dev_energy_score':best,'train_supervised_roots':len(train_rows),'dev_supervised_roots':len(dev_rows),'parameters':sum(p.numel() for p in net.parameters()),'seconds':time.perf_counter()-start,'threads':torch.get_num_threads(),'objective':'root-balanced conditional position mixture NLL with propagated initial covariance'}
    (output/'training.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('train');p.add_argument('dev');p.add_argument('output');p.add_argument('--epochs',type=int,default=30);p.add_argument('--seed',type=int,default=11);p.add_argument('--width',type=int,default=64);p.add_argument('--root-limit',type=int,default=0);a=p.parse_args();run(**vars(a))
