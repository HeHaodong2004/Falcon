"""Matched original-risk training using legal observed camera calibration."""
import argparse
from collections import Counter
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .diagnose import OUT,PREP,read,save,sha
from .current_camera_model import CameraNet
from eventwm_graph.stage2_hf.common import source


def fit(name='current_camera_pilot',split='pilot_fit',epochs=8,device='cuda:0',warm=None,ablate_camera=False):
    torch.set_num_threads(1);random.seed(7183);np.random.seed(7183);torch.manual_seed(7183);torch.backends.cuda.matmul.allow_tf32=False
    path=PREP/(split+'_mixed_ray/manifest.json');m=read(path);side=OUT/'current_camera_prepared'/split/'manifest.json';cm=read(side);assert cm['identity']['sample_manifest_sha256']==sha(path)and cm['rows']==m['sampled_cells']
    for name_,digest in cm['identity']['sources'].items():assert sha(Path(__file__).with_name(name_))==digest
    roots={e['root_id']:e['scene_id']for s in m['shards']for e in s['segments']};counts=Counter(roots.values());factor={s:len(roots)/(len(counts)*n)for s,n in counts.items()};dest=OUT/'training'/name;dest.mkdir(parents=True,exist_ok=False);point=OUT/'training/point_residual_full/epoch06.pt';model=CameraNet();model.initialize(point)
    if warm:model.load_state_dict(torch.load(warm,map_location='cpu')['model'])
    model.to(device);optim=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0007,weight_decay=.0001);rng=np.random.default_rng(7183);scale=m['sampled_cells']/m['roots'];logs=[];tick=time.perf_counter()
    cfg=dict(schema='stage2_legal_current_camera_1',name=name,split=split,epochs=epochs,seed=7183,point_checkpoint=str(point),point_checkpoint_sha256=sha(point),source=source(),sources={p.name:sha(p)for p in Path(__file__).parent.glob('current_camera_*.py')},prepared_manifest_sha256=sha(path),camera_manifest_sha256=sha(side),learning_rate=.0007,pos_weight=8.,roots=m['roots'],windows=m['counts']['candidate_segments'],warm=warm,
        inputs='Current legal graph and supplied trajectory, plus current OBSERVED camera extrinsic (translation and rotation relative to base). Eight continuous 63-probe frustum cylinder-margin summaries use only that calibration, planned base poses, map resolution and nearest-known height. No privileged geometry, future cameras, Stage1 future predictions, or truth-human inputs.',risk='Unchanged exact inverse inclusion equalScene/Root/window BCEpos8 revelation +8R conditionalBCE; odds divided8 in evaluation.',operating_point='Select epoch and revelation threshold only on original 31-Root/16-scene/666-window fixed pilot; full development is already inspected and not called untouched.')
    save(dest/'config.json',cfg)
    cfg['ablate_camera']=ablate_camera;save(dest/'config.json',cfg)
    for epoch in range(1,epochs+1):
        total=np.zeros(3)
        for si in rng.permutation(len(m['shards'])):
            s=m['shards'][int(si)];c=cm['shards'][int(si)];assert c['source_shard_sha256']==s['sha256']
            with np.load(s['path'],allow_pickle=False)as z:x=z['x'][:,16:].astype(np.float32);y=z['y'].astype(np.int64);w=z['weight'].astype(np.float32)
            with np.load(c['path'],allow_pickle=False)as z:geometry=z['geometry'];calibration=z['calibration'];owner=z['owner']
            for e in s['segments']:w[e['start']:e['end']]*=factor[e['scene_id']]
            order=rng.permutation(len(y))
            for begin in range(0,len(order),16384):
                ids=order[begin:begin+16384];xt=torch.from_numpy(x[ids]).to(device);extra=torch.from_numpy(np.concatenate([geometry[ids],calibration[owner[ids]]],1)).to(device)
                if ablate_camera:extra.zero_()
                yt=torch.from_numpy(y[ids]).to(device);wt=torch.from_numpy(w[ids]).to(device)*scale;r=(yt>0).float();rl,bl,_=model.raw(xt,extra);ce=F.binary_cross_entropy_with_logits(rl,r,reduction='none',pos_weight=rl.new_tensor(8.))+8*r*F.binary_cross_entropy_with_logits(bl,(yt==2).float(),reduction='none');loss=(ce*wt).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite current-camera objective')
                optim.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optim.step();total+=[float((ce*wt).sum()),float(wt.sum()),len(ids)]
        assert int(total[2])==m['sampled_cells'];logs.append(dict(epoch=epoch,rows=int(total[2]),loss=float(total[0]/total[1]),gradient_norm_last=float(grad),seconds=time.perf_counter()-tick));save(dest/'learning.json',logs);cp=dest/('epoch%02d.pt'%epoch);torch.save(dict(schema=cfg['schema'],config=cfg,model=model.state_dict(),epoch=epoch,threshold=.2),cp);print(logs[-1],flush=True)
        if epoch%2==0 or epoch==epochs:
            from .current_camera_eval import evaluate
            evaluate(cp,device=device,name=name+'_epoch%02d'%epoch)
    save(dest/'complete.json',dict(epochs=epochs,seconds=time.perf_counter()-tick))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='current_camera_pilot');p.add_argument('--split',default='pilot_fit');p.add_argument('--epochs',type=int,default=8);p.add_argument('--device',default='cuda:0');p.add_argument('--warm');p.add_argument('--ablate-camera',action='store_true');fit(**vars(p.parse_args()))
