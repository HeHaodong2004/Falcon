"""One controlled maximum-cohort run with the existing proper-score objective."""
import json
import math
import os
import random
import resource
import shutil
import time
import numpy as np
import torch
from .common import BASE,OUT,PRIOR,PREVIOUS,sha,dump
from eventwm_graph.l2_intermittent.neural import IntermittentDynamics
from eventwm_graph.l2_intermittent.fit import make_batch,objective,evaluate
from eventwm_graph.l2_intermittent import fit as reference_fit
from .factor_kernel import sample_factors as compiled_sample_factors


def run():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.manual_seed(221);np.random.seed(221);random.seed(221)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.2)
    preparation=json.loads((OUT/'preparation.json').read_text());rows=[]
    contract=json.loads((OUT/'preparation_contract.json').read_text())
    assert sha(OUT/'snapshot/manifest.json')==contract['snapshot_sha256']
    assert sha(OUT/'final_snapshot_manifest.json')==contract['final_snapshot_sha256']
    assert all(sha(BASE/name)==value for name,value in contract['estimator_sources'].items())
    for shard in preparation['shards']:
        path=OUT/shard['path'];assert sha(path)==shard['sha256'];rows.extend(np.load(path,allow_pickle=True))
    assert len(rows)==preparation['motion_roots'] and len({r['root_id'] for r in rows})==len(rows)
    dev=list(np.load(PRIOR/'dev.npy',allow_pickle=True));assert not ({r['root_id'] for r in rows}&{r['root_id'] for r in dev})
    law=json.loads((PRIOR/'motion.json').read_text());dump(OUT/'motion.json',law)
    dest=OUT/'training';dest.mkdir(exist_ok=False);net=IntermittentDynamics(64).to(device)
    initial=torch.load(PREVIOUS/'training/epoch0.pt',map_location='cpu')
    assert all(torch.equal(v.cpu(),initial['state_dict'][k]) for k,v in net.state_dict().items())
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=80,eta_min=.00005)
    cfg=dict(seed=221,width=64,parameters=sum(p.numel() for p in net.parameters()),law=law,device=device,epochs=80,
             minimum_epochs=30,patience=12,min_improvement=.001,batch_size=48,learning_rate=.001,weight_decay=.001,gradient_norm=5.,
             objective='unchanged root-balanced path Energy Score; 3 modes x initial Stop/Move; four paths/alternative train, eight dev; half shifted future timestamps',
             train_roots=len(rows),train_scenes=preparation['motion_scenes'],dev_roots=len(dev),
             snapshot_sha256=sha(OUT/'final_snapshot_manifest.json'),main_snapshot_sha256=sha(OUT/'snapshot/manifest.json'),preparation_sha256=sha(OUT/'preparation.json'),dev_sha256=sha(PRIOR/'dev.npy'),
             protocol_sha256=sha(BASE/'docs/l2/final_800k_20261023/PROTOCOL.md'),
             sources={str(p.relative_to(BASE)):sha(p) for folder in ('l2_final','l2_intermittent','l2_120k','l2_partial') for p in (BASE/'eventwm_graph'/folder).glob('*.py')},
             preparation_contract_sha256=sha(OUT/'preparation_contract.json'),environment_sha256=sha(OUT/'environment.json'),
             gpu=torch.cuda.get_device_name() if device=='cuda' else None,
             memory_policy='compact immutable root shards; complete cohort fits in RAM; complete deterministic root permutation each epoch')
    cfg['training_sampler']='compiled existing CTMC factor arithmetic; same Generator draw order; reference sampler retained for exact incumbent validation comparison'
    for name,expected in cfg['sources'].items():
        target=dest/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/name,target)
        assert sha(target)==expected
    dump(dest/'configuration.json',cfg);rng=np.random.default_rng(221)
    best=significant=math.inf;last_significant=0;history=[];start=time.perf_counter();cpu=time.process_time();steps=0
    for epoch in range(81):
        losses=[];net.train();order=rng.permutation(len(rows)) if epoch else np.empty(0,int)
        selection_hash=__import__('hashlib').sha256();visited=0
        for offset in range(0,len(order),48):
            part=[rows[j] for j in order[offset:offset+48]];selections=[]
            original_sampler=reference_fit.sample_factors
            try:
                reference_fit.sample_factors=compiled_sample_factors
                batch=make_batch(part,rng,law,4,device,selection_log=selections)
            finally:reference_fit.sample_factors=original_sampler
            selection_hash.update(json.dumps(selections,sort_keys=True).encode());visited+=len(part)
            optimizer.zero_grad(set_to_none=True);loss=objective(net,batch)
            if not torch.isfinite(loss):raise ValueError('nonfinite objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step();steps+=1;losses.append(float(loss))
        assert visited==(len(rows) if epoch else 0)
        net.eval();validation=evaluate(net,dev,law,device)
        if epoch==0:assert validation==initial['dev_path_es'],(validation,initial['dev_path_es'])
        rec=dict(epoch=epoch,train_es=float(np.mean(losses)) if losses else None,dev_es=validation,
                 elapsed_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu,learning_rate=optimizer.param_groups[0]['lr'],
                 optimizer_steps=steps,visited_training_roots=visited,selection_sha256=selection_hash.hexdigest(),
                 rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0)
        history.append(rec);print(json.dumps(rec),flush=True)
        if epoch:scheduler.step()
        ck=dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},width=64,epoch=epoch,dev_path_es=validation,configuration=cfg,
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),numpy_rng_state=rng.bit_generator.state,
                python_rng_state=random.getstate(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state() if device=='cuda' else None)
        def save(name):
            target=dest/name;temporary=target.with_suffix('.pt.tmp');torch.save(ck,temporary);temporary.replace(target)
        save('last.pt')
        if epoch==0:save('epoch0.pt')
        if epoch and validation<best:best=validation;save('checkpoint.pt')
        if epoch and validation<=significant-.001:significant=validation;last_significant=epoch
        dump(dest/'learning.json',history)
        if epoch>=30 and epoch-last_significant>=12:break
    dump(dest/'training.json',dict(**cfg,completed_epochs=epoch,selected_dev_path_es=best,seconds=time.perf_counter()-start,
        cpu_seconds=time.process_time()-cpu,optimizer_steps=steps,stop_reason='significant_best_patience' if epoch<80 else 'maximum_epochs',
        initial_tensors_bitwise_equal=True,initial_dev_score_equal=True,checkpoint_sha256=sha(dest/'checkpoint.pt')))


if __name__=='__main__':run()
