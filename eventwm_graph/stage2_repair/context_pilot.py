"""Single targeted context candidate after the matched same-example control."""
import argparse
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .diagnose import OUT,HF,read,save,sha
from eventwm_graph.stage2_hf.common import source
from eventwm_graph.stage2_hf.spatial import SpatialNet
from eventwm_graph.stage2_hf.spatial_train import Windows,collate,loss_fn
from eventwm_graph.stage2_hf.spatial_cuda import QueryKernel


def fit(epochs=2,device='cuda:1',workers=4):
    torch.set_num_threads(1);random.seed(2526);np.random.seed(2526);torch.manual_seed(2526);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    mpath=HF/'packed/pilot_fit/manifest.json';m=read(mpath);warm=HF/'training/spatial_bce/epoch08.pt';saved=torch.load(warm,map_location='cpu')
    model=SpatialNet(12,context_dilations=[2,4,8]).to(device);model.load_state_dict(saved['model'],strict=False);kernel=QueryKernel(device);dest=HF/'training/context_repair_pilot';dest.mkdir(exist_ok=False)
    cfg=dict(schema='stage2_hf_spatial_1',source=source(),repair_source_sha256=sha(__file__),width=12,context_dilations=[2,4,8],pos_weight=8.,dice_weight=0.,warm=str(warm),warm_sha256=sha(warm),epochs=epochs,manifest_sha256=sha(mpath),batch_size=16,workers=workers,seed=2526,learning_rate=.001,
        hypothesis='One context residual, motivated by matched small-set fit .895 vs .774. No architecture sweep; full original candidates/targets and window weighting.',precision='FP16 AMP with F32 losses; F32 scientific inference')
    save(dest/'config.json',cfg);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scaler=torch.cuda.amp.GradScaler();tick=time.perf_counter();logs=[]
    for epoch in range(1,epochs+1):
        loader=DataLoader(Windows(m,epoch),batch_size=16,num_workers=workers,collate_fn=collate,pin_memory=True,prefetch_factor=2)
        steps=0;seen=0
        for batch in loader:
            b={k:v.to(device,non_blocking=True)for k,v in batch.items()};q=kernel(b['maps'],b['path'],b['pre'],b['globals'],b['spec']);optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():o=model(b['maps'],q,b['fine']);loss,parts=loss_fn(o,b,8.,0.)
            if not torch.isfinite(loss):raise ValueError('nonfinite context objective')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optimizer);scaler.update();steps+=1;seen+=len(b['weight'])
            if steps%250==0:print(dict(epoch=epoch,steps=steps,windows=seen,seconds=time.perf_counter()-tick),flush=True)
        assert seen==m['segments'];logs.append(dict(epoch=epoch,windows=seen,steps=steps,seconds=time.perf_counter()-tick));save(dest/'learning.json',logs)
        cp=dest/('epoch%02d.pt'%epoch);torch.save(dict(schema=cfg['schema'],config=cfg,model_config=dict(width=12,context_dilations=[2,4,8]),model={k:v.detach().cpu()for k,v in model.state_dict().items()},epoch=epoch),cp)
        from eventwm_graph.stage2_hf.spatial_eval import evaluate
        evaluate('context_repair_pilot',[epoch],device=device)
    save(dest/'complete.json',dict(seconds=time.perf_counter()-tick,epochs=epochs))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=2);p.add_argument('--device',default='cuda:1');p.add_argument('--workers',type=int,default=4);fit(**vars(p.parse_args()))
