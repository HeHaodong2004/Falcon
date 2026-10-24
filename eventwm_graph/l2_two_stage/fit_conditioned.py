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


def tilt_initial_bias(initialization,cutoff):
    """Parse the audited initializer's actual schema; never mutate its artifact."""
    if type(cutoff) is not int or cutoff!=7 or initialization.get('schema')!='fitting_only_GLOBAL_total_count_tilt_initializer_1':
        raise ValueError('unrecognized total-count initializer schema/cutoff')
    raw=np.asarray(initialization['log_tilt_weights8'])
    if raw.dtype.kind not in 'fiu' or raw.shape!=(cutoff+1,) or not np.isfinite(raw).all():
        raise ValueError('invalid fitting-only total-count tilt initialization')
    biases=np.asarray(raw,np.float64)
    if np.any(abs(biases)>np.finfo(np.float32).max):raise ValueError('count-head bias exceeds the neural numeric representation')
    return biases.copy()


def run(name,split='train_pilot_native_v2',sidecar='train_pilot_conditioned',epochs=8,
        seed=413,batch_size=48,full_fit=False,history_fraction=.5,learning_rate=.0002,
        initial=None,resume=None,association_guidance=False,marginal_weight=0.,validation_marginal_weight=None,visible_initial_reparameterization=False,
        worlds=4,validation_worlds=None,residual_count_components=1,count_head_only=False,count_initialization=None,total_count_tilt_cutoff=None,current_likelihood_only=False,analytic_empty_normalizer=False):
    if validation_worlds is None:validation_worlds=worlds
    for count in (worlds,validation_worlds):
        if isinstance(count,bool) or not isinstance(count,(int,np.integer)) or count<3:
            raise ValueError('proper normalization gradients need at least3 integer iid worlds')
    worlds,validation_worlds=int(worlds),int(validation_worlds)
    if type(residual_count_components) is not int or residual_count_components not in (1,2):
        raise ValueError('supported residual count components are1 or2')
    tilted=total_count_tilt_cutoff is not None
    if tilted and (type(total_count_tilt_cutoff) is not int or total_count_tilt_cutoff!=7 or residual_count_components!=1):
        raise ValueError('total-count tilt requires cutoff7 and the original single-rate base')
    new_count_head=residual_count_components==2 or tilted
    if count_head_only and not new_count_head:raise ValueError('count-only fitting requires a new count head')
    if new_count_head and count_initialization is None:
        raise ValueError('new count fitting requires an audited fitting-only initialization')
    if current_likelihood_only and not (tilted and count_head_only):
        raise ValueError('conditional count-likelihood calibration requires total-count tilting and a frozen base')
    if analytic_empty_normalizer and not current_likelihood_only:
        raise ValueError('analytic-empty normalization is declared only for count-likelihood calibration')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed)
    if validation_marginal_weight is None:validation_marginal_weight=marginal_weight
    if not all(math.isfinite(w) and w>=0 for w in (marginal_weight,validation_marginal_weight)):
        raise ValueError('proper marginal score weights must be finite and nonnegative')
    if current_likelihood_only and (marginal_weight or validation_marginal_weight or visible_initial_reparameterization):
        raise ValueError('future-only score controls cannot enter count-likelihood calibration')
    dest=OUT/'experiments'/name;dest.mkdir(parents=True,exist_ok=False)
    initial=Path(initial or OUT/'experiments/pilot_k_bounded_unique/best.pt')
    saved=torch.load(initial,map_location='cpu');model_config=dict(saved['model_config'])
    if model_config.get('residual_count_components',1)!=1 or model_config.get('total_count_tilt_cutoff') is not None:
        raise ValueError('count-family campaign initialization must use the frozen single-rate teacher; use exact resume for a mixed checkpoint')
    initialization=None
    if new_count_head:
        if tilted:model_config['total_count_tilt_cutoff']=total_count_tilt_cutoff
        else:model_config['residual_count_components']=2
        count_initialization=Path(count_initialization)
        initialization=json.loads(count_initialization.read_text())
        if initialization.get('teacher_sha256')!=sha(initial) or initialization.get('no_validation_or_dev_parameter_fit') is not True:
            raise ValueError('count initialization is not bound to the fitting-only teacher')
    model=HumanGraphPredictor(**model_config).to('cuda:0')
    incompatible=model.load_state_dict(saved['model'],strict=not new_count_head)
    if new_count_head:
        head_name='count_tilt' if tilted else 'count_mixture'
        if incompatible.unexpected_keys or set(incompatible.missing_keys)!={head_name+'.weight',head_name+'.bias'}:
            raise ValueError('unexpected base-to-count-head initialization mismatch')
        if tilted:
            biases=tilt_initial_bias(initialization,total_count_tilt_cutoff)
        else:
            scales=np.asarray(initialization['relative_scales'],np.float64)
            weights=np.asarray(initialization['weights'],np.float64)
            if scales.shape!=(2,) or weights.shape!=(2,) or not np.isfinite(scales).all() or not np.isfinite(weights).all() or np.any(scales<=1e-5) or np.any(weights<=0) or abs(weights.sum()-1)>1e-10:
                raise ValueError('invalid fitting-only count mixture initialization')
            biases=np.r_[np.log(np.expm1(scales-1e-5)),np.log(weights[1]/weights[0])]
        count_head=getattr(model,head_name)
        with torch.no_grad():
            count_head.weight.zero_();count_head.bias.copy_(torch.as_tensor(biases,device='cuda:0',dtype=count_head.bias.dtype))
    if count_head_only:
        model.requires_grad_(False);count_head.requires_grad_(True)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=learning_rate,weight_decay=.001)
    records=load_records(split)
    audits=dict(history=attach_histories(records,sidecar),events=attach_metadata(records,sidecar),
        future=attach_targets(records,'train_campaign_20261023'))
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
    if initialization is not None:
        def root_digest(values):return hashlib.sha256(json.dumps(sorted(values),separators=(',',':')).encode()).hexdigest()
        expected_validation=[] if full_fit else {r['root_id'] for r in validation}
        if (initialization.get('fitting_Root_ids_sha256')!=root_digest(fit) or
                initialization.get('validation_Root_ids_sha256')!=root_digest(expected_validation) or
                initialization.get('data_manifests',{}).get('prepared')!=sha(OUT/'prepared'/split/'manifest.json')):
            raise ValueError('count initialization fitting/data boundary differs from this run')
        for label,audit_key in (('history','history'),('events','events'),('targets','future')):
            if initialization.get('data_manifests',{}).get(label)!=audits[audit_key]['manifest_sha256']:
                raise ValueError('count initialization '+label+' provenance differs from this run')
    cycle=max(len(rows) for rows in fit.values())
    if epochs<cycle:raise ValueError('schedule must expose every eligible causal anchor')
    from .guided_association import GUIDANCE
    kernel=LINEAGE_KERNEL+('; '+GUIDANCE+'; source_first=True' if association_guidance else '')
    config=dict(schema='history_conditioned_stage1_training_1',name=name,split=split,sidecar=sidecar,
        epochs=epochs,seed=seed,batch_size=batch_size,full_fit=full_fit,model_config=model_config,
        conditioned_config=dict(history_fraction=history_fraction,lineage_kernel=kernel,association_guidance=association_guidance),
        learning_rate=learning_rate,weight_decay=.001,gradient_clip=5.,iid_worlds=worlds,validation_iid_worlds=validation_worlds,
        objective=f'physical-lineage joint tracked-path Energy Score + {marginal_weight} mean native2D marginal Energy Score + .2 complete-future-set kernel score + .05 normalized current finite-set NLL',
        marginal_weight=marginal_weight,validation_marginal_weight=validation_marginal_weight,
        visible_initial_reparameterization=bool(visible_initial_reparameterization),
        residual_count_components=residual_count_components,count_head_only=bool(count_head_only),
        total_count_tilt_cutoff=total_count_tilt_cutoff,
        current_likelihood_only=bool(current_likelihood_only),analytic_empty_normalizer=bool(analytic_empty_normalizer),
        count_initialization_mapping='schema1 log_tilt_weights8 -> count_tilt.bias; cutoff=len-1=7; weights zero' if tilted else None,
        count_initialization=None if count_initialization is None else str(count_initialization),
        count_initialization_sha256=None if count_initialization is None else sha(count_initialization),
        trainable_parameters=[key for key,value in model.named_parameters() if value.requires_grad],
        count_context='certified full-history DAG minimum, actual current positive multiset size, anonymous proposal episode count; /10',
        visible_initial_rationale='Optional mixed-coordinate gradient estimator only: EXACTage0 known-source initial Gaussian noise is independent of current feasibility/association; future pathwise derivative replaces only its Gaussian latent score. Positive-age and ghost coefficients stay scored. Forward law/RNG unchanged.',
        marginal_rationale='Native2D proper score of the SAME accepted, lineage-gathered trajectories, supplementing the global path norm; no unconditioned modes or ghost paths. Weight0 is an unchanged-objective control. Common validation weight is explicit.',
        objective_rationale='Joint path, optional native2D marginal, complete-set and current-density proper scores on the SAME declared law; independent accepted latents correct normalizer gradients. No risk heads.',
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
    if current_likelihood_only:
        config.update(objective='Normalized current finite-set NLL only; scalar is set-density NLL, head gradient equals conditional COUNT NLL with frozen base.',
            objective_rationale='Cardinality calibration/convergence experiment; all base motion/graph/spatial/presence parameters frozen. Independent future/trajectory quality remains mandatory. No H0 kernel-conflict claim.',
            normalization='Exact accepted-current negative phase; known-zero phase/logZA for H0 only when explicitly enabled. Kappa integrates out; no lineage/future construction in this training objective.')
    from .common import BASE
    config['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip()
    config['git_status']=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=BASE,text=True).strip()
    log=[];steps=0;start_epoch=0;best=math.inf
    if resume:
        resume=Path(resume);old=torch.load(resume,map_location='cpu')
        if old.get('resume_exact',True) is False:raise ValueError('mid-epoch diagnostic is not an exact resumable checkpoint; use prior last.pt or epoch0.pt')
        for key in ('split','sidecar','seed','batch_size','full_fit','model_config','conditioned_config','learning_rate','iid_worlds','validation_iid_worlds','marginal_weight','validation_marginal_weight','visible_initial_reparameterization','residual_count_components','total_count_tilt_cutoff','current_likelihood_only','analytic_empty_normalizer','count_head_only','count_initialization_sha256','trainable_parameters','data_manifest_sha256','data_audits','source_hashes'):
            if key not in old['config'] or old['config'][key]!=config[key]:raise ValueError('resume identity mismatch:'+key)
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
    def forward(rows,sampling_rng,point_weight=marginal_weight,iid_worlds=worlds):
        batch=make_batch(rows);states=[graphs[r['root_id'],r['anchor_index']] for r in rows]
        arguments=(batch['x'],batch['graph'],batch['mask'],batch['global_features'],batch['global_graph'])
        if new_count_head:
            from .mixed_count import legal_context
            count_context=torch.as_tensor(np.asarray([legal_context(r,r['positive_history'],g) for r,g in zip(rows,states)]),device=batch['x'].device)
            p=model(*arguments,count_context=count_context)
        else:p=model(*arguments)
        p=augment_torch(p,rows,states,history_fraction)
        return loss(p,batch,rows,states,sampling_rng,worlds=iid_worlds,guidance=association_guidance,marginal_weight=point_weight,visible_initial_reparameterization=visible_initial_reparameterization,current_likelihood_only=current_likelihood_only,analytic_empty_normalizer=analytic_empty_normalizer)
    for epoch in range(start_epoch+1,epochs+1):
        epoch_rows=[dict(rows[(epoch-1+offsets[root])%len(rows)],weight=1.) for root,rows in fit.items()]
        order=np.random.permutation(len(epoch_rows));model.train();values=[];term_sums={};seen=0;gradient_norms=[]
        try:
            for begin in range(0,len(order),batch_size):
                rows=[epoch_rows[int(i)] for i in order[begin:begin+batch_size]]
                objective,terms=forward(rows,rng)
                if not torch.isfinite(objective):raise RuntimeError('nonfinite proper training score')
                optimizer.zero_grad(set_to_none=True);objective.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(norm):raise RuntimeError('nonfinite training gradient')
                gradient_norms.append(float(norm.detach()))
                optimizer.step();steps+=1;seen+=len(rows);values.append(float(objective.detach())*len(rows))
                for key,value in terms.items():
                    term_sums[key]=term_sums.get(key,0.)+float(value.sum().detach().cpu() if torch.is_tensor(value) else value.sum())
                if steps%100==0:print(json.dumps(dict(epoch=epoch,steps=steps,roots_seen=seen,seconds=time.perf_counter()-start)),flush=True)
            model.eval();validation_sum=0.;validation_terms={};denominator=sum(r['weight'] for r in validation)
            torch_state=torch.cuda.get_rng_state();torch.cuda.manual_seed(seed+10000)
            validation_rng=np.random.default_rng(seed+10000)
            with torch.no_grad():
                for begin in range(0,len(validation),batch_size):
                    rows=validation[begin:begin+batch_size];objective,terms=forward(rows,validation_rng,validation_marginal_weight,validation_worlds)
                    if not torch.isfinite(objective):raise RuntimeError('nonfinite held-Root proper score')
                    validation_sum+=float(objective)*len(rows)
                    weights=np.array([r['weight'] for r in rows])
                    for key,value in terms.items():
                        validation_values=value.detach().cpu().numpy() if torch.is_tensor(value) else value
                        validation_terms[key]=validation_terms.get(key,0.)+float((validation_values*weights).sum())
            torch.cuda.set_rng_state(torch_state);score=validation_sum/denominator
            row=dict(epoch=epoch,steps=steps,training_score=sum(values)/seen,validation_score=score,
                terms={k:v/seen for k,v in term_sums.items()},seconds=time.perf_counter()-start)
            row['preclip_gradient_norm']=dict(mean=float(np.mean(gradient_norms)),p95=float(np.quantile(gradient_norms,.95)),
                maximum=float(np.max(gradient_norms)),clipped_fraction=float(np.mean(np.asarray(gradient_norms)>5.)))
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
    parser.add_argument('--marginal-weight',type=float,default=0.)
    parser.add_argument('--validation-marginal-weight',type=float)
    parser.add_argument('--visible-initial-reparameterization',action='store_true')
    parser.add_argument('--worlds',type=int,default=4)
    parser.add_argument('--validation-worlds',type=int)
    parser.add_argument('--residual-count-components',type=int,default=1)
    parser.add_argument('--count-head-only',action='store_true')
    parser.add_argument('--count-initialization')
    parser.add_argument('--total-count-tilt-cutoff',type=int)
    parser.add_argument('--current-likelihood-only',action='store_true')
    parser.add_argument('--analytic-empty-normalizer',action='store_true')
    run(**vars(parser.parse_args()))
