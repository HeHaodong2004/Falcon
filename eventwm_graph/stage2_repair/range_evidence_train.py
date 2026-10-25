"""Bounded fit with fixed real training Roots and unchanged sampling weights."""
import argparse
from collections import Counter
import time
import numpy as np
import torch
from torch.nn import functional as F
from .diagnose import OUT,read,save,sha
from .range_evidence import EvidenceNet


def train(epochs=40,device='cuda:0',name='range_evidence64'):
    torch.set_num_threads(1);torch.manual_seed(4817);torch.backends.cuda.matmul.allow_tf32=False
    m=read(OUT/'range_evidence_train/manifest.json');counts=Counter(r['scene_id']for r in m['records']);factor={s:m['roots']/(len(counts)*n)for s,n in counts.items()};arrays={k:[]for k in ['x','extra','y','weight']}
    for row in m['records']:
        if sha(row['path'])!=row['sha256']:raise ValueError('training range feature cache changed')
        with np.load(row['path'])as z:
            for k in arrays:arrays[k].append(z[k]*(factor[row['scene_id']]if k=='weight'else 1))
    a={k:torch.from_numpy(np.concatenate(v)).to(device)for k,v in arrays.items()};a['x']=a['x'].float();a['extra']=a['extra'].float();a['weight']=a['weight'].float()*m['rows']/m['roots'];a['y']=a['y'].long()
    point=OUT/'training/point_residual_full/epoch06.pt';model=EvidenceNet();model.initialize(point);model.to(device);optim=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.001,weight_decay=.0001)
    dest=OUT/'training'/name;dest.mkdir(exist_ok=False);cfg=dict(schema='legal_range_evidence16_1',name=name,epochs=epochs,seed=4817,learning_rate=.001,pos_weight=8.,data_manifest_sha256=sha(OUT/'range_evidence_train/manifest.json'),roots=m['roots'],scenes=m['scenes'],windows=m['windows'],rows=m['rows'],point_checkpoint=str(point),point_checkpoint_sha256=sha(point),inputs='LEGAL current/past measured range images, legally observed current camera calibration, current graph, supplied base trajectory. Sensor extension; no hidden geometry/navmesh/future camera inputs.');save(dest/'config.json',cfg);logs=[];tick=time.perf_counter()
    for epoch in range(1,epochs+1):
        order=torch.randperm(m['rows'],device=device);loss_sum=0.;weight_sum=0.
        for ids in order.split(8192):
            y=a['y'][ids];r=(y>0).float();rl,bl,_=model.raw(a['x'][ids],a['extra'][ids]);ce=F.binary_cross_entropy_with_logits(rl,r,pos_weight=rl.new_tensor(8.),reduction='none')+8*r*F.binary_cross_entropy_with_logits(bl,(y==2).float(),reduction='none');w=a['weight'][ids];loss=(ce*w).mean();optim.zero_grad(set_to_none=True);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optim.step();loss_sum+=float((ce*w).sum());weight_sum+=float(w.sum())
        if epoch%5==0 or epoch==1:
            cms=np.zeros((3,3));model.eval()
            with torch.inference_mode():
                for begin in range(0,m['rows'],16384):
                    sl=slice(begin,begin+16384);p,_=model.probabilities(a['x'][sl],a['extra'][sl]);hard=torch.where(p[:,1]+p[:,2]>=.2,torch.where(p[:,1]>=p[:,2],1,2),0);cm=torch.bincount(a['y'][sl]*3+hard,weights=a['weight'][sl],minlength=9).reshape(3,3);cms+=cm.cpu().numpy()
            from scripts.stage2_diagnostic.common import extended_rates
            rec=dict(epoch=epoch,loss=loss_sum/weight_sum,gradient_norm_last=float(gradient),weighted_sampled_training=extended_rates(cms),seconds=time.perf_counter()-tick);logs.append(rec);save(dest/'learning.json',logs);print(rec,flush=True);model.train()
        if epoch%10==0 or epoch==epochs:torch.save(dict(schema=cfg['schema'],config=cfg,epoch=epoch,threshold=.2,model=model.state_dict()),dest/('epoch%02d.pt'%epoch))
    save(dest/'complete.json',dict(seconds=time.perf_counter()-tick,epochs=epochs))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=40);p.add_argument('--device',default='cuda:0');p.add_argument('--name',default='range_evidence64');train(**vars(p.parse_args()))
