"""One fixed offline supervised recipe on all eligible Roots/Queries."""
import argparse
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
from .common import BASE,OUT,SCHEMA,dump,sha,source_identity
from .model import GraphDeltaNet


def run(name='final',epochs=6,batch_size=8192,device='cuda:0'):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    seed=2410;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    manifest_path=OUT/'prepared/train/manifest.json';manifest=json.loads(manifest_path.read_text())
    if manifest['identity']['split']!='train' or manifest['failures']:raise ValueError('invalid supervised fitting cache')
    dest=OUT/'training'/name;dest.mkdir(parents=True,exist_ok=False)
    config=dict(schema=SCHEMA,epochs=epochs,batch_size=batch_size,seed=seed,learning_rate=.001,weight_decay=.0001,
        gradient_clip=5.,height_huber_weight=.2,model_config=dict(width=48),dataset_manifest=str(manifest_path),
        dataset_sha256=sha(manifest_path),sources=source_identity(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=BASE,text=True).strip(),
        selection='fixed final epoch; no hyperparameter search or development selection',
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,device=device),
        roots=manifest['roots'],scenes=manifest['scenes'],counts=manifest['counts'])
    dump(dest/'config.json',config)
    model=GraphDeltaNet(**config['model_config']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed);steps=0;ledger=[];start=time.perf_counter()
    total_rows=manifest['counts']['sampled_cells'];weight_scale=total_rows/manifest['roots']
    for epoch in range(1,epochs+1):
        model.train();sums=np.zeros(4,np.float64);rows_seen=0
        for shard_index in rng.permutation(len(manifest['shards'])):
            shard=manifest['shards'][int(shard_index)];path=Path(shard['path'])
            if sha(path)!=shard['sha256']:raise ValueError('training shard changed')
            with np.load(path,allow_pickle=False) as archive:
                x=archive['x'].astype(np.float32);y=archive['y'].astype(np.int64);height=archive['height'].astype(np.float32);weight=archive['weight']
            order=rng.permutation(len(x))
            for begin in range(0,len(order),batch_size):
                at=order[begin:begin+batch_size]
                xb=torch.from_numpy(x[at]).to(device);yb=torch.from_numpy(y[at]).to(device)
                hb=torch.from_numpy(height[at]).to(device);wb=torch.from_numpy(weight[at]).to(device)
                logits,h=model(xb);ce=F.cross_entropy(logits,yb,reduction='none')
                hl=F.smooth_l1_loss(h,hb,reduction='none')*(yb>0)
                # A shard-final short batch retains each row's global weight.
                loss=((ce+.2*hl)*wb).sum()*weight_scale/batch_size
                if not torch.isfinite(loss):raise RuntimeError('nonfinite Stage2 objective')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();steps+=1
                sums+=np.array([float((ce*wb).sum()),float((hl*wb).sum()),float(wb.sum()),len(at)])
                rows_seen+=len(at)
            if steps%100<10:
                print(json.dumps(dict(epoch=epoch,steps=steps,rows=rows_seen,seconds=time.perf_counter()-start)),flush=True)
        if rows_seen!=total_rows:raise ValueError('epoch omitted supervised rows')
        row=dict(epoch=epoch,steps=steps,rows=rows_seen,root_weight=float(sums[2]),logloss=float(sums[0]/sums[2]),height_huber=float(sums[1]/sums[2]),seconds=time.perf_counter()-start)
        ledger.append(row);dump(dest/'learning.json',ledger)
        torch.save(dict(schema=SCHEMA,model_config=config['model_config'],model={k:v.cpu() for k,v in model.state_dict().items()},config=config,
            optimizer=optimizer.state_dict(),epoch=epoch,steps=steps,torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state),dest/'last.pt')
        print(json.dumps(row),flush=True)
    dump(dest/'complete.json',dict(status='COMPLETE_RESEARCH_CHECKPOINT',checkpoint_sha256=sha(dest/'last.pt'),epochs=epochs,steps=steps,
        parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='final');p.add_argument('--epochs',type=int,default=6);p.add_argument('--batch-size',type=int,default=8192);p.add_argument('--device',default='cuda:0')
    run(**vars(p.parse_args()))
