"""Resumable Root-cycled fitting of the fully history-conditioned Stage1 law."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from .association import AssociationGraph,LINEAGE_KERNEL
from .common import OUT,dump,sha
from .conditioned_training import make_batch,loss
from .data import load_records
from .future_population_targets import attach_targets
from .history_proposal import augment_torch
from .last_event_metadata import attach_metadata
from .neural import HumanGraphPredictor
from .positive_history import attach_histories
from .train import root_validation


def run(name,split='train_pilot_native_v2',sidecar='train_pilot_conditioned',epochs=8,
        seed=413,batch_size=48,full_fit=False,history_fraction=.5,learning_rate=.0002,
        initial=None,resume=None,association_guidance=False):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed)
    dest=OUT/'experiments'/name;dest.mkdir(parents=True,exist_ok=False)
    initial=Path(initial or OUT/'experiments/pilot_k_bounded_unique/best.pt')
    saved=torch.load(initial,map_location='cpu');model_config=saved['model_config']
    model=HumanGraphPredictor(**model_config).to('cuda:0');model.load_state_dict(saved['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=.001)
    records=load_records(split)
    audits=dict(history=attach_histories(records,sidecar),events=attach_metadata(records,sidecar),
        future=attach_targets(records,sidecar))
    dump(dest/'data_join_audit.json',audits)
    print(json.dumps(dict(status='building_association_state',anchors=len(records))),flush=True)
    # Build immutable per-anchor states in causal Root order. No model inference
    # cache participates in these offline training data transformations.
    graphs={};previous={}
    for r in records:
        root=r['root_id'];history=r['positive_history']
        g=previous[root].append(history) if root in previous else AssociationGraph.from_history(history)
        graphs[root,r['anchor_index']]=g;previous[root]=g
    grouped={}
    for r in records:grouped.setdefault(r['root_id'],[]).append(r)
    fit={root:rows for root,rows in grouped.items() if full_fit or not root_validation(root)}
    validation=[r for r in records if root_validation(r['root_id'])]
    cycle=max(len(rows) for rows in fit.values())
    if epochs<cycle:raise ValueError('schedule must expose every eligible causal anchor')
    from .guided_association import GUIDANCE
    kernel=LINEAGE_KERNEL+('; '+GUIDANCE+'; source_first=True' if association_guidance else '')
    config=dict(schema='history_conditioned_stage1_training_1',name=name,split=split,sidecar=sidecar,
        epochs=epochs,seed=seed,batch_size=batch_size,full_fit=full_fit,model_config=model_config,
        conditioned_config=dict(history_fraction=history_fraction,lineage_kernel=kernel,association_guidance=association_guidance),
        learning_rate=learning_rate,weight_decay=.001,gradient_clip=5.,iid_worlds=4,
        objective='physical-lineage joint tracked-path Energy Score + .2 complete-future-set kernel score + .05 normalized current finite-set NLL',
        objective_rationale='Three proper scores on declared complete probability law; independent accepted latents correct normalizer gradients. Current NLL uses fixed-success unbiased log-normalizer value. No risk heads.',
        normalization='exact rejection sampling; accepted-latent negative phase; independent leave-pair-out score correction',
        initial_checkpoint=str(initial),initial_sha256=sha(initial),
        fit_roots=len(fit),fit_anchors=sum(map(len,fit.values())),fit_scenes=len({r['scene_id'] for rows in fit.values() for r in rows}),
        validation_roots=len({r['root_id'] for r in validation}),validation_anchors=len(validation),
        validation_role='in_sample_diagnostic_only' if full_fit else 'held_roots_in_fitting_scenes',
        minimum_selection_epoch=cycle,selection_rule='fixed_last_epoch' if full_fit else 'held_root_proper_score_after_full_anchor_cycle',
        data_manifest_sha256=sha(OUT/'prepared'/split/'manifest.json'),data_audits=audits,
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0),torch_threads=torch.get_num_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
        source_hashes={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')})
    from .common import BASE
    config['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip()
    config['git_status']=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=BASE,text=True).strip()
    log=[];steps=0;start_epoch=0;best=math.inf
    if resume:
        resume=Path(resume);old=torch.load(resume,map_location='cpu')
        if old.get('resume_exact',True) is False:raise ValueError('mid-epoch diagnostic is not an exact resumable checkpoint; use prior last.pt or epoch0.pt')
        for key in ('split','sidecar','seed','batch_size','full_fit','model_config','conditioned_config','learning_rate','data_manifest_sha256','data_audits','source_hashes'):
            if old['config'][key]!=config[key]:raise ValueError('resume identity mismatch:'+key)
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
        start_epoch=old['epoch'];steps=old['steps'];rng.bit_generator.state=old['sampling_rng']
        if epochs<=start_epoch:raise ValueError('resume must request additional complete epochs')
        torch.set_rng_state(old['torch_rng']);torch.cuda.set_rng_state(old['cuda_rng'])
        np.random.set_state(old['numpy_rng']);random.setstate(old['python_rng'])
        logpath=resume.parent/'learning.json'
        if not logpath.exists() and start_epoch!=0:raise ValueError('completed resume epoch has no immutable learning ledger')
        log=[row for row in json.loads(logpath.read_text()) if row['epoch']<=start_epoch] if logpath.exists() else []
        best=min((row['validation_score'] for row in log if row['epoch']>=cycle),default=math.inf)
        if not full_fit and math.isfinite(best):
            if torch.load(resume.parent/'best.pt',map_location='cpu')['epoch']>start_epoch:raise ValueError('historical resume requires its historical best checkpoint')
            shutil.copyfile(resume.parent/'best.pt',dest/'best.pt')
        config.update(resume=str(resume),resume_sha256=sha(resume))
    dump(dest/'config.json',config)
    (dest/'source').mkdir()
    for source in Path(__file__).parent.glob('*.py'):shutil.copyfile(source,dest/'source'/source.name)
    def save(filename,epoch,resume_exact=True):
        torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},model_config=model_config,
            conditioned_config=config['conditioned_config'],optimizer=optimizer.state_dict(),epoch=epoch,steps=steps,config=config,resume_exact=resume_exact,
            sampling_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate()),dest/filename)
    save('epoch0.pt',start_epoch)
    offsets={root:int(hashlib.sha256(('anchor-cycle:'+root).encode()).hexdigest()[:8],16) for root in fit}
    start=time.perf_counter()
    def forward(rows,sampling_rng):
        batch=make_batch(rows);states=[graphs[r['root_id'],r['anchor_index']] for r in rows]
        p=model(batch['x'],batch['graph'],batch['mask'],batch['global_features'],batch['global_graph'])
        p=augment_torch(p,rows,states,history_fraction)
        return loss(p,batch,rows,states,sampling_rng,guidance=association_guidance)
    for epoch in range(start_epoch+1,epochs+1):
        epoch_rows=[dict(rows[(epoch-1+offsets[root])%len(rows)],weight=1.) for root,rows in fit.items()]
        order=np.random.permutation(len(epoch_rows));model.train();values=[];term_sums={};seen=0
        try:
            for begin in range(0,len(order),batch_size):
                rows=[epoch_rows[int(i)] for i in order[begin:begin+batch_size]]
                objective,terms=forward(rows,rng)
                if not torch.isfinite(objective):raise RuntimeError('nonfinite proper training score')
                optimizer.zero_grad(set_to_none=True);objective.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(norm):raise RuntimeError('nonfinite training gradient')
                optimizer.step();steps+=1;seen+=len(rows);values.append(float(objective.detach())*len(rows))
                for key,value in terms.items():
                    term_sums[key]=term_sums.get(key,0.)+float(value.sum().detach().cpu() if torch.is_tensor(value) else value.sum())
                if steps%100==0:print(json.dumps(dict(epoch=epoch,steps=steps,roots_seen=seen,seconds=time.perf_counter()-start)),flush=True)
            model.eval();validation_sum=0.;validation_terms={};denominator=sum(r['weight'] for r in validation)
            torch_state=torch.cuda.get_rng_state();torch.cuda.manual_seed(seed+10000)
            validation_rng=np.random.default_rng(seed+10000)
            with torch.no_grad():
                for begin in range(0,len(validation),batch_size):
                    rows=validation[begin:begin+batch_size];objective,terms=forward(rows,validation_rng)
                    if not torch.isfinite(objective):raise RuntimeError('nonfinite held-Root proper score')
                    validation_sum+=float(objective)*len(rows)
                    weights=np.array([r['weight'] for r in rows])
                    for key,value in terms.items():
                        validation_values=value.detach().cpu().numpy() if torch.is_tensor(value) else value
                        validation_terms[key]=validation_terms.get(key,0.)+float((validation_values*weights).sum())
            torch.cuda.set_rng_state(torch_state);score=validation_sum/denominator
            row=dict(epoch=epoch,steps=steps,training_score=sum(values)/seen,validation_score=score,
                terms={k:v/seen for k,v in term_sums.items()},seconds=time.perf_counter()-start)
            row['validation_terms']={k:v/denominator for k,v in validation_terms.items()}
            log.append(row);dump(dest/'learning.json',log);save('last.pt',epoch)
            if not full_fit and epoch>=cycle and score<best:best=score;save('best.pt',epoch)
            print(json.dumps(row),flush=True)
        except Exception as error:
            save('failed_mid_epoch.pt',epoch,resume_exact=False);dump(dest/'failure.json',dict(epoch=epoch,steps=steps,error=repr(error),
                exact_resume_checkpoint='last.pt' if epoch>1 else 'epoch0.pt',
                roots=[dict(root_id=r['root_id'],anchor=r['anchor_index']) for r in rows],
                interpretation='Explicit failure. No Root/target removed; checkpoint is resumable research, not accepted.'))
            raise
    selected='last.pt' if full_fit else 'best.pt'
    dump(dest/'complete.json',dict(status='TRAINING_COMPLETE_NOT_ACCEPTANCE',selected_checkpoint=selected,
        checkpoint_sha256=sha(dest/selected),epochs=epochs,steps=steps,seconds=time.perf_counter()-start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),unique_fit_roots=len(fit),fit_anchors=config['fit_anchors']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name');parser.add_argument('--split',default='train_pilot_native_v2')
    parser.add_argument('--sidecar',default='train_pilot_conditioned');parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--seed',type=int,default=413);parser.add_argument('--batch-size',type=int,default=48)
    parser.add_argument('--full-fit',action='store_true');parser.add_argument('--history-fraction',type=float,default=.5)
    parser.add_argument('--learning-rate',type=float,default=.0002);parser.add_argument('--initial');parser.add_argument('--resume')
    parser.add_argument('--association-guidance',action='store_true')
    run(**vars(parser.parse_args()))
