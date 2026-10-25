"""Shared-map reconstruction training on original exact window supervision."""
import argparse
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .diagnose import OUT,HF,PREP,read,save,sha
from .shared_map import SharedMapNet
from .shared_data import SharedGraphs,collate
from eventwm_graph.stage2_hf.common import source


def loss_fn(rlog,blog,b,scale,dice_weight=.2):
    y=b['y'].long();target=(y>0).float();ce=F.binary_cross_entropy_with_logits(rlog,target,reduction='none',pos_weight=rlog.new_tensor(8.))+8*target*F.binary_cross_entropy_with_logits(blog,(y==2).float(),reduction='none')
    proper=(ce*b['weight']).sum();r=(rlog-np.log(8.)).sigmoid();blocked=blog.sigmoid();correct=r*torch.where(y==1,1-blocked,blocked)*target
    nw=len(b['window_weight']);ids=b['window_id'];w=b['cell_weight'];intersection=torch.zeros(nw,device=rlog.device).scatter_add_(0,ids,correct*w);pm=torch.zeros_like(intersection).scatter_add_(0,ids,r*w);tm=torch.zeros_like(intersection).scatter_add_(0,ids,target*w)
    dice=1-(2*intersection+b['eps'])/(pm+tm+b['eps']);di=(dice*b['window_weight']).sum()
    n=b['maps'].shape[0];return (proper+dice_weight*di)*scale/n,dict(proper=float(proper.detach()),dice=float(di.detach()))


def fit(name='shared_map_pilot',split='pilot_fit',epochs=2,device='cuda:0',workers=6,batch_size=12,warm=None,dice=.2):
    torch.set_num_threads(1);random.seed(3011);np.random.seed(3011);torch.manual_seed(3011);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    mpath=PREP/(split+'_mixed_ray/manifest.json');m=read(mpath);dataset=SharedGraphs(m,split,0);dest=OUT/'training'/name;dest.mkdir(parents=True,exist_ok=False)
    model=SharedMapNet();point=OUT/'training/point_residual_full/epoch06.pt';spatial=HF/'training/spatial_bce/epoch08.pt';model.initialize(point,spatial)
    if warm:model.load_state_dict(torch.load(warm,map_location='cpu')['model'])
    model.to(device);scale=dataset.n_groups/m['roots'];cfg=dict(schema='stage2_shared_map_1',source=source(),sources={p.name:sha(p)for p in Path(__file__).parent.glob('shared_*.py')},
        name=name,split=split,epochs=epochs,seed=3011,point_checkpoint=str(point),point_checkpoint_sha256=sha(point),spatial_initialization_sha256=sha(spatial),warm=warm,manifest_sha256=sha(mpath),
        roots=m['roots'],windows=m['counts']['candidate_segments'],starting_graphs=dataset.n_groups,learning_rate=.0005,batch_size=batch_size,workers=workers,dice_weight=dice,pos_weight=8.,
        input='Current legal map states/heights/EDTs and certificate-publication age; supplied trajectory in unchanged88point features. No future truth, camera or human input.',
        weighting='Original inverse inclusion cell weights/equalScene/Root/window; grouped by starting graph only for compute reuse. Full original evaluation universe.',precision='AMP map encoder, F32 point heads/loss; F32 scientific evaluation',shared_encoder='Four map scales, masked padding after every convolution; map canvas depends on current graph only; outside-canvas candidates retain zero local plus global context, never dropped.')
    save(dest/'config.json',cfg);optim=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0005,weight_decay=.0001);scaler=torch.cuda.amp.GradScaler();logs=[];tick=time.perf_counter()
    for epoch in range(1,epochs+1):
        data=SharedGraphs(m,split,epoch);loader=DataLoader(data,batch_size=batch_size,num_workers=workers,collate_fn=collate,pin_memory=True,prefetch_factor=2);steps=0;ng=0;nw=0;total=np.zeros(2);model.train()
        for batch in loader:
            b={k:v.to(device,non_blocking=True)for k,v in batch.items()};optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():encoded=model.encode(b['maps'],b['sizes'])
            r,bl,_=model.raw(b['x'],b['indices'],b['map_ids'],encoded);loss,parts=loss_fn(r,bl,b,scale,dice)
            if not torch.isfinite(loss):raise ValueError('nonfinite shared-map objective')
            scaler.scale(loss).backward();scaler.unscale_(optim);grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optim);scaler.update();steps+=1;ng+=len(b['maps']);nw+=len(b['window_weight']);total+=[parts['proper'],parts['dice']]
            if steps%100==0:print(dict(epoch=epoch,steps=steps,graphs=ng,windows=nw,seconds=time.perf_counter()-tick,loss=float(loss),gradient_norm=float(grad),map_gradient=float(model.enc0.c1.weight.grad.norm())),flush=True)
        assert ng==dataset.n_groups and nw==m['counts']['candidate_segments'];logs.append(dict(epoch=epoch,steps=steps,graphs=ng,windows=nw,seconds=time.perf_counter()-tick,loss_sums=total.tolist()));save(dest/'learning.json',logs)
        cp=dest/('epoch%02d.pt'%epoch);torch.save(dict(schema=cfg['schema'],config=cfg,model=model.state_dict(),epoch=epoch,threshold=.2),cp)
        from .shared_eval import evaluate
        evaluate(cp,device=device,name=name+'_epoch%02d'%epoch)
    save(dest/'complete.json',dict(seconds=time.perf_counter()-tick,epochs=epochs,parameters=sum(p.numel()for p in model.parameters()),trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad),peak_cuda_allocated=torch.cuda.max_memory_allocated(device)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='shared_map_pilot');p.add_argument('--split',default='pilot_fit');p.add_argument('--epochs',type=int,default=2);p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=6);p.add_argument('--batch-size',type=int,default=12);p.add_argument('--warm');p.add_argument('--dice',type=float,default=.2);fit(**vars(p.parse_args()))
