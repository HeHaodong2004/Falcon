"""Matched signed-pose feature pilot, same valid samples and risk weights."""
import argparse
from collections import Counter
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .diagnose import OUT,PREP,read,save,sha
from .pose_model import PoseNet
from eventwm_graph.stage2_hf.common import source


def fit(name='signed_pose_pilot',split='pilot_fit',epochs=6,device='cuda:0',warm=None):
    torch.set_num_threads(1);random.seed(6031);np.random.seed(6031);torch.manual_seed(6031);torch.backends.cuda.matmul.allow_tf32=False
    path=PREP/(split+'_mixed_ray/manifest.json');m=read(path);posepath=OUT/'pose_prepared'/split/'manifest.json';pm=read(posepath);assert pm['identity']['source_manifest_sha256']==sha(path)and pm['rows']==m['sampled_cells']
    assert pm['identity']['source_feature_sha256']==sha(Path(__file__).with_name('pose_features.py'))
    roots={e['root_id']:e['scene_id']for s in m['shards']for e in s['segments']};counts=Counter(roots.values());factor={s:len(roots)/(len(counts)*n)for s,n in counts.items()};dest=OUT/'training'/name;dest.mkdir(parents=True,exist_ok=False);point=OUT/'training/point_residual_full/epoch06.pt';model=PoseNet();model.initialize(point)
    if warm:model.load_state_dict(torch.load(warm,map_location='cpu')['model'])
    model.to(device);optim=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0005,weight_decay=.0001);rng=np.random.default_rng(6031);scale=m['sampled_cells']/m['roots'];tick=time.perf_counter();logs=[]
    cfg=dict(schema='stage2_signed_pose_1',name=name,split=split,epochs=epochs,seed=6031,point_checkpoint=str(point),point_checkpoint_sha256=sha(point),source=source(),source_sha256=sha(__file__),prepared_manifest_sha256=sha(path),pose_manifest_sha256=sha(posepath),learning_rate=.0005,pos_weight=8.,roots=m['roots'],windows=m['counts']['candidate_segments'],warm=warm,
        inputs='Original88legalfeatures +23signedbearings ofSTART/firstfuture/mid/end suppliedposes, theirabsoluteheadingcos/sin andstart-to-futureviewmargins. No camera truth. Yaw was always present in suppliedtrajectory.',loss='Original exact inverseinclusion equalScene/Root/window risk; pos8 revelationBCE +8R conditionalBCE; divideodds8 at evaluation.')
    save(dest/'config.json',cfg)
    cfg['pose_sources']={p.name:sha(p)for p in Path(__file__).parent.glob('pose_*.py')}
    save(dest/'config.json',cfg)
    for epoch in range(1,epochs+1):
        total=np.zeros(3)
        for si in rng.permutation(len(m['shards'])):
            shard=m['shards'][int(si)];ps=pm['shards'][int(si)];assert ps['source_shard_sha256']==shard['sha256']
            with np.load(shard['path'],allow_pickle=False)as z:x=z['x'][:,16:].astype(np.float32);y=z['y'].astype(np.int64);w=z['weight'].astype(np.float32)
            with np.load(ps['path'],allow_pickle=False)as z:extra=z['extra']
            for e in shard['segments']:w[e['start']:e['end']]*=factor[e['scene_id']]
            order=rng.permutation(len(y))
            for begin in range(0,len(order),16384):
                ids=order[begin:begin+16384];xt=torch.from_numpy(x[ids]).to(device);et=torch.from_numpy(extra[ids]).to(device);yt=torch.from_numpy(y[ids]).to(device);wt=torch.from_numpy(w[ids]).to(device)*scale;r=(yt>0).float();rl,bl,_=model.raw(xt,et);ce=F.binary_cross_entropy_with_logits(rl,r,reduction='none',pos_weight=rl.new_tensor(8.))+8*r*F.binary_cross_entropy_with_logits(bl,(yt==2).float(),reduction='none');loss=(ce*wt).mean()
                optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optim.step();total+=[float((ce*wt).sum()),float(wt.sum()),len(ids)]
        assert int(total[2])==m['sampled_cells'];logs.append(dict(epoch=epoch,rows=int(total[2]),loss=float(total[0]/total[1]),seconds=time.perf_counter()-tick));save(dest/'learning.json',logs);cp=dest/('epoch%02d.pt'%epoch);torch.save(dict(schema=cfg['schema'],config=cfg,model=model.state_dict(),epoch=epoch,threshold=.2),cp);print(logs[-1],flush=True)
        if epoch in [2,4,epochs]:
            from .pose_eval import evaluate
            evaluate(cp,device=device,name=name+'_epoch%02d'%epoch)
    save(dest/'complete.json',dict(epochs=epochs,seconds=time.perf_counter()-tick))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='signed_pose_pilot');p.add_argument('--split',default='pilot_fit');p.add_argument('--epochs',type=int,default=6);p.add_argument('--device',default='cuda:0');p.add_argument('--warm');fit(**vars(p.parse_args()))
