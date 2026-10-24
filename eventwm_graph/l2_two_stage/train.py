"""Root-weighted probabilistic training, resumable immutable experiment folders."""
import argparse
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from .common import OUT,dump,sha
from .data import load_records,collate
from .neural import HumanGraphPredictor,proper_loss


def root_validation(root):return int(hashlib.sha256(('two-stage-internal:'+root).encode()).hexdigest()[:8],16)%5==0


def run(name,split='train_pilot',epochs=12,tail='gaussian',graph=True,seed=231,batch_size=128,full_fit=False,population_objective='residual',root_cycle=False,endpoint_weight=0.,validation_endpoint_weight=0.05,persistent_offset_velocity=False,resume=None,compulsory_unique_episode=False,max_speed=None,compulsory_observed_clique=False):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    dest=OUT/'experiments'/name;dest.mkdir(parents=True,exist_ok=False)
    records=load_records(split);fit=[r for r in records if full_fit or not root_validation(r['root_id'])]
    validation=[r for r in records if root_validation(r['root_id'])]
    if population_objective=='set':
        from .population import attach_targets
        target_audit=attach_targets(records,split);dump(dest/'set_target_audit.json',target_audit)
    channels=records[0]['graph'].shape[-1]
    fixed_motion=None
    if channels>10 and population_objective=='set':
        numerator=denominator=0.
        for r in fit:
            valid=r['unseen_velocity_valid']>0
            if valid.any():
                velocity=r['unseen_velocity'][valid]
                if max_speed is not None:
                    from .speed import inverse_numpy
                    velocity=inverse_numpy(velocity,max_speed)
                numerator+=r['weight']*float((velocity**2).mean());denominator+=r['weight']
        fixed_motion=max(.05,float(np.sqrt(numerator/max(denominator,1e-12))))
    model_config=dict(width=48,graph_enabled=graph,tail=tail,independent_initial=population_objective=='set',graph_channels=channels,unseen_motion_std=fixed_motion,minimum_count_conditioned=channels>10 and population_objective=='set',persistent_offset_velocity=persistent_offset_velocity,compulsory_unique_episode=compulsory_unique_episode,max_speed=max_speed,compulsory_observed_clique=compulsory_observed_clique)
    model=HumanGraphPredictor(**model_config).to('cuda:0');optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    anchor_counts={}
    for r in fit:anchor_counts[r['root_id']]=anchor_counts.get(r['root_id'],0)+1
    minimum_selection_epoch=max(anchor_counts.values()) if root_cycle else 1
    if epochs<minimum_selection_epoch:raise ValueError('schedule cannot cover every causal anchor')
    config=dict(schema='two_stage_stage1_training_2',name=name,split=split,epochs=epochs,seed=seed,
        batch_size=batch_size,full_fit=full_fit,root_cycle=root_cycle,model_config=model_config,population_objective=population_objective,
        endpoint_weight=endpoint_weight,validation_endpoint_weight=validation_endpoint_weight,
        objective=('joint_path_ES + .05 exact_current_MBPPP_set_NLL + endpoint_weight proper_native_position_score' if population_objective=='set' else 'joint_path_ES + .05 Bernoulli_BCE + .05 marked_spatial_Poisson_NLL'),
        objective_rationale='Joint path ES scores coherent futures; current finite-set NLL accounts for all population mass; marginal future NLL penalizes overconfident point marginals of the same trajectory law. All objectives are proper for their declared targets; no risk heads.',
        marginal_score='native_future_position_energy' if max_speed is not None else 'native_future_position_mixture_NLL',
        fit_roots=len({r['root_id'] for r in fit}),fit_anchors=len(fit),validation_roots=len({r['root_id'] for r in validation}),
        learning_rate=.001,weight_decay=.001,gradient_clip=5,worlds_per_mode=4,
        minimum_selection_epoch=minimum_selection_epoch,selection_rule='fixed_last_epoch' if full_fit else 'held_root_minimum_proper_loss_after_one_full_anchor_cycle',
        validation_role='in_sample_diagnostic_only' if full_fit else 'held_root_shared_fitting_scenes',
        data_manifest_sha256=sha(OUT/'prepared'/split/'manifest.json'),
        source_hashes={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')})
    log=[];best=math.inf;steps=0;start_epoch=0
    if resume:
        resume=Path(resume).resolve();saved=torch.load(resume,map_location='cpu');old=saved['config']
        for key in ('split','seed','batch_size','full_fit','root_cycle','model_config','population_objective',
                    'endpoint_weight','validation_endpoint_weight','data_manifest_sha256','source_hashes'):
            if old[key]!=config[key]:raise ValueError('resume identity mismatch: '+key)
        start_epoch=int(saved['epoch']);steps=int(saved['steps'])
        if start_epoch>=epochs:raise ValueError('resume checkpoint already reached requested total epochs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        log=[r for r in json.loads((resume.parent/'learning.json').read_text()) if r['epoch']<=start_epoch]
        if not full_fit:
            eligible=[r['validation_loss'] for r in log if r['epoch']>=minimum_selection_epoch]
            best=min(eligible,default=math.inf)
            if eligible:
                prior_best=torch.load(resume.parent/'best.pt',map_location='cpu')
                if prior_best['epoch']>start_epoch:raise ValueError('historical resume needs its own preserved best checkpoint')
                shutil.copyfile(resume.parent/'best.pt',dest/'best.pt')
        config.update(resumed_from=str(resume),resumed_sha256=sha(resume),resumed_epoch=start_epoch)
    dump(dest/'config.json',config);start=time.perf_counter();previous_seconds=log[-1]['seconds'] if log else 0.
    (dest/'source').mkdir()
    for source in Path(__file__).parent.glob('*.py'):shutil.copyfile(source,dest/'source'/source.name)
    def make_batch(rows):
        b=collate(rows,'cuda:0')
        if population_objective=='set':
            from .population import add_batch
            add_batch(rows,b,'cuda:0')
        return b
    def save(path,epoch):
        torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},model_config=model_config,
            optimizer=optimizer.state_dict(),epoch=epoch,steps=steps,config=config,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),path)
    save(dest/('resume_initial.pt' if resume else 'epoch0.pt'),start_epoch)
    grouped={}
    for r in fit:grouped.setdefault(r['root_id'],[]).append(r)
    offsets={root:int(hashlib.sha256(('anchor-cycle:'+root).encode()).hexdigest()[:8],16) for root in grouped}
    for epoch in range(start_epoch+1,epochs+1):
        epoch_fit=[dict(rows[(epoch-1+offsets[root])%len(rows)],weight=1.) for root,rows in grouped.items()] if root_cycle else fit
        model.train();order=np.random.permutation(len(epoch_fit));train_loss=[]
        for begin in range(0,len(order),batch_size):
            rows=[epoch_fit[i] for i in order[begin:begin+batch_size]];batch=make_batch(rows)
            p=model(batch['x'],batch['graph'],batch['mask'],batch['global_features'],batch['global_graph'])
            loss,_=proper_loss(p,batch,tail=tail,population_objective=population_objective,independent_initial=model_config['independent_initial'],condition_minimum=model_config['minimum_count_conditioned'],endpoint_weight=endpoint_weight,persistent_offset_velocity=persistent_offset_velocity,compulsory_unique_episode=compulsory_unique_episode,max_speed=max_speed,compulsory_observed_clique=compulsory_observed_clique)
            if not torch.isfinite(loss):
                dump(dest/'failure.json',dict(status='NONFINITE_TRAINING_SCORE',epoch=epoch,steps=steps,
                    roots=[dict(root_id=r['root_id'],anchor=r['anchor_index']) for r in rows],
                    interpretation='No target exclusion or fabricated finite score; experiment stops.'))
                save(dest/'failed_score.pt',epoch-1)
                raise RuntimeError('nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            train_loss.append(float(loss.detach()));steps+=1
        model.eval();valid_loss=[]
        # Fixed seed at each epoch makes checkpoint selection comparable.
        rng=torch.cuda.get_rng_state();torch.cuda.manual_seed(seed+10000)
        with torch.no_grad():
            for begin in range(0,len(validation),batch_size):
                rows=validation[begin:begin+batch_size];batch=make_batch(rows)
                p=model(batch['x'],batch['graph'],batch['mask'],batch['global_features'],batch['global_graph'])
                loss,_=proper_loss(p,batch,tail=tail,population_objective=population_objective,independent_initial=model_config['independent_initial'],condition_minimum=model_config['minimum_count_conditioned'],endpoint_weight=validation_endpoint_weight,persistent_offset_velocity=persistent_offset_velocity,compulsory_unique_episode=compulsory_unique_episode,max_speed=max_speed,compulsory_observed_clique=compulsory_observed_clique);valid_loss.append((float(loss),len(rows)))
        torch.cuda.set_rng_state(rng);score=float(sum(v*n for v,n in valid_loss)/sum(r['weight'] for r in validation))
        if not math.isfinite(score):
            dump(dest/'failure.json',dict(status='NONFINITE_VALIDATION_SCORE',epoch=epoch,steps=steps,
                nonfinite_batches=[i for i,(value,count) in enumerate(valid_loss) if not math.isfinite(value)],
                interpretation='Valid supported targets cannot be censored to repair a model support failure.'))
            save(dest/'failed_validation.pt',epoch)
            raise RuntimeError('nonfinite validation loss; inspect failed_validation.pt and preserved targets')
        row=dict(epoch=epoch,train_loss=float(np.mean(train_loss)),validation_loss=score,steps=steps,seconds=previous_seconds+time.perf_counter()-start)
        log.append(row);dump(dest/'learning.json',log);save(dest/'last.pt',epoch)
        if not full_fit and epoch>=minimum_selection_epoch and score<best:best=score;save(dest/'best.pt',epoch)
        print(json.dumps(row),flush=True)
    dump(dest/'complete.json',dict(config=config,epochs=epochs,steps=steps,seconds=previous_seconds+time.perf_counter()-start,
        checkpoint_sha256=sha(dest/'last.pt' if full_fit else dest/'best.pt'),final_sha256=sha(dest/'last.pt'),
        selected_checkpoint='last.pt' if full_fit else 'best.pt',best_validation=None if full_fit else best,
        peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--split',default='train_pilot');p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--tail',choices=['gaussian','student'],default='gaussian');p.add_argument('--no-graph',action='store_true');p.add_argument('--full-fit',action='store_true')
    p.add_argument('--seed',type=int,default=231);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--population-objective',choices=['residual','set'],default='residual');p.add_argument('--root-cycle',action='store_true')
    p.add_argument('--endpoint-weight',type=float,default=0.);p.add_argument('--validation-endpoint-weight',type=float,default=0.05);p.add_argument('--persistent-offset-velocity',action='store_true');p.add_argument('--resume');p.add_argument('--compulsory-unique-episode',action='store_true');p.add_argument('--max-speed',type=float,choices=[3.]);p.add_argument('--compulsory-observed-clique',action='store_true');a=p.parse_args()
    run(a.name,a.split,a.epochs,a.tail,not a.no_graph,a.seed,a.batch_size,a.full_fit,a.population_objective,a.root_cycle,a.endpoint_weight,a.validation_endpoint_weight,a.persistent_offset_velocity,a.resume,a.compulsory_unique_episode,a.max_speed,a.compulsory_observed_clique)
