"""Bounded full fit of the pilot-selected residual without recency inputs."""
import argparse
from collections import Counter
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .diagnose import OUT,HF,PREP,CKPT,read,save,sha
from eventwm_graph.stage2_hf.common import source
from eventwm_graph.stage2_hf.time_model import TimeNet


def fit(epochs=6,device='cuda:0',name='point_residual_full'):
    torch.set_num_threads(1);random.seed(2526);np.random.seed(2526);torch.manual_seed(2526);torch.backends.cuda.matmul.allow_tf32=False
    mpath=PREP/'full_mixed_ray/manifest.json';m=read(mpath);dest=OUT/'training'/name;dest.mkdir(parents=True,exist_ok=False)
    warm=HF/'training/time_zero_control/epoch04.pt';saved=torch.load(warm,map_location='cpu');assert saved['config']['ablate_time']
    model=TimeNet(**saved['model_config']);model.load_state_dict(saved['model']);model.to(device)
    roots={e['root_id']:e['scene_id']for s in m['shards']for e in s['segments']};scenes=Counter(roots.values())
    assert len(roots)==m['roots']==105838 and m['counts']['candidate_segments']==2482285
    factors={s:len(roots)/(len(scenes)*n)for s,n in scenes.items()};scale=m['sampled_cells']/m['roots']
    cfg=dict(schema='stage2_hf_time_1',epochs=epochs,split='full',seed=2526,device=device,train_occupancy=True,width=64,pos_weight=8.,learning_rate=.0003,batch_size=16384,
        weight_decay=.0001,source=source(),repair_source_sha256=sha(__file__),prepared_manifest_sha256=sha(mpath),prior_checkpoint_sha256=sha(CKPT),
        warm=str(warm.resolve()),warm_sha256=sha(warm),ablate_time=True,roots=len(roots),scenes=len(scenes),windows=m['counts']['candidate_segments'],sampled_rows=m['sampled_cells'],exclusions=m['failures'],
        rationale='Independent replay/features passed; original point model fits same-example supervision; matched pilot time-ablation .219916 exceeds recency .218899 and baseline .217413. No demonstrated benefit from extra recency input or larger scratch model.',
        loss='Unchanged exact inverse-inclusion equalScene/Root/window revelationBCEpos8 +8*R*conditionalBCE; posterior odds divided by8. Frozen Stage2 prior and height; no Stage1 input.',
        selection='At most six full epochs. Pilot .20 threshold fixed; choose epoch by same 16-scene pilot reconstruction, then evaluate all108 dev scenes without further model/threshold changes.',precision='float32; TF32 disabled')
    save(dest/'config.json',cfg);optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0003,weight_decay=.0001);rng=np.random.default_rng(2526);logs=[];tick=time.perf_counter();initial={k:v.detach().clone()for k,v in model.correction.state_dict().items()}
    for epoch in range(1,epochs+1):
        total=np.zeros(3);gradient=[];steps=0;model.train()
        for si in rng.permutation(len(m['shards'])):
            shard=m['shards'][int(si)]
            if epoch==1 and sha(shard['path'])!=shard['sha256']:raise ValueError('training shard checksum changed')
            with np.load(shard['path'],allow_pickle=False)as z:x=z['x'][:,16:].astype(np.float32);y=z['y'].astype(np.int64);w=z['weight'].astype(np.float32)
            for e in shard['segments']:w[e['start']:e['end']]*=factors[e['scene_id']]
            order=rng.permutation(len(y))
            for begin in range(0,len(order),16384):
                ids=order[begin:begin+16384];xt=torch.from_numpy(x[ids]).to(device);yt=torch.from_numpy(y[ids]).to(device);wt=torch.from_numpy(w[ids]).to(device)*scale;r=(yt>0).float()
                rlog,blog,_=model.raw(xt,torch.zeros((len(ids),32),device=device));ce=F.binary_cross_entropy_with_logits(rlog,r,reduction='none',pos_weight=rlog.new_tensor(8.))+8*r*F.binary_cross_entropy_with_logits(blog,(yt==2).float(),reduction='none');loss=(ce*wt).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();steps+=1
                if steps%1024==0:gradient.append(float(gn))
                total+=[float((ce*wt).sum()),float(wt.sum()),len(ids)]
            if int(si)%256==0:print(dict(epoch=epoch,rows=int(total[2]),seconds=time.perf_counter()-tick),flush=True)
        assert int(total[2])==m['sampled_cells']
        log=dict(epoch=epoch,rows=int(total[2]),steps=steps,loss=float(total[0]/total[1]),seconds=time.perf_counter()-tick,gradient_norm_samples=gradient,
            parameter_update_norm=float(sum((v-initial[k]).square().sum()for k,v in model.correction.state_dict().items()).sqrt()))
        logs.append(log);save(dest/'learning.json',logs);cp=dest/('epoch%02d.pt'%epoch)
        torch.save(dict(schema=cfg['schema'],config=cfg,model_config=saved['model_config'],model={k:v.detach().cpu()for k,v in model.state_dict().items()},epoch=epoch,threshold=.2),cp)
        print(log,flush=True)
        from .evaluate_pilot import evaluate
        evaluate(cp,device=device,name=name+'_epoch%02d'%epoch)
    save(dest/'complete.json',dict(seconds=time.perf_counter()-tick,epochs=epochs,peak_cuda_allocated=torch.cuda.max_memory_allocated(device),trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=6);p.add_argument('--device',default='cuda:0');p.add_argument('--name',default='point_residual_full');fit(**vars(p.parse_args()))
