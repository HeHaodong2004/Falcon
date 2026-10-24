"""Exact, Root-weighted H0 count calibration; no future-world supervision.

H0 means the ENTIRE actual positive history is empty. Then A is identically
one, there are no proposal sources, and all count normalizers are analytic.
Only the optional seven-output affine correction is fitted. All original
parameters and the positive-history inference branch remain unchanged.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import poisson
import scipy
import torch

from .association import AssociationGraph
from .common import BASE,OUT,dump,sha
from .data import collate,load_records
from .future_population_targets import attach_targets
from .last_event_metadata import attach_metadata
from .mixed_count import legal_context
from .neural import HumanGraphPredictor
from .positive_history import attach_histories
from .train import root_validation


def guarded_source():
    source=Path(__file__).resolve().parents[2]
    path=source/'manifest.json'
    if not path.is_file():raise ValueError('fit from an immutable executable source')
    manifest=json.loads(path.read_text())
    for relative,expected in manifest['files'].items():
        if sha(source/relative)!=expected:raise ValueError('changed immutable source: '+relative)
    return source,manifest


def exact_h0_log_probabilities(logits,rate,counts):
    """Individual-count log scores AND complete 0..6/unbounded-tail classes."""
    logits=np.asarray(logits,np.float64);rate=np.asarray(rate,np.float64);counts=np.asarray(counts)
    if (logits.shape!=(len(rate),8) or counts.shape!=rate.shape or counts.dtype.kind not in 'iu'
            or not np.isfinite(logits).all() or not np.isfinite(rate).all()
            or np.any(rate<=0) or np.any(counts<0)):
        raise ValueError('supported exact H0 rate/potentials/individual counts required')
    masses=np.c_[poisson.logpmf(np.arange(7)[None],rate[:,None]),poisson.logsf(6,rate)]
    if not np.isfinite(masses).all():raise RuntimeError('H0 count masses exceed numerical support')
    norm=logsumexp(logits+masses,axis=-1)
    individual=logits[np.arange(len(rate)),np.minimum(counts,7)]+poisson.logpmf(counts,rate)-norm
    return logits+masses-norm[:,None],individual


def affine_objective(flat,design,log_classes,log_individual,counts,weights,penalty):
    """Convex proper individual-count likelihood plus a declared L2 penalty."""
    matrix=np.asarray(flat,np.float64).reshape(7,design.shape[1])
    delta=np.c_[design@matrix.T,np.zeros(len(design))]
    norm=logsumexp(log_classes+delta,axis=1)
    category=np.minimum(counts,7)
    score=-float(weights@(log_individual+delta[np.arange(len(design)),category]-norm))
    probabilities=np.exp(log_classes+delta-norm[:,None])
    gradient=probabilities.copy();gradient[np.arange(len(design)),category]-=1.
    derivative=(gradient[:,:7]*weights[:,None]).T@design+penalty*matrix
    return score+.5*penalty*float(np.square(matrix).sum()),derivative.ravel()


def encode_h0(records,model):
    """Use the actual neural forward, taking only legal inputs from each row."""
    selected=[];captured=[];totals=collections.Counter(r['root_id'] for r in records)
    hook=model.population[1].register_forward_hook(lambda module,args,value:captured.append(value.detach().numpy().copy()))
    try:
        for record in records:
            history=record['positive_history']
            if int(history.positive_offsets[-1])!=0:continue
            graph=AssociationGraph.from_history(history)
            context=legal_context(record,history,graph)
            if graph.detections or graph.minimum_current_count or np.any(context):
                raise ValueError('H0 must certify the complete actual positive history')
            target=np.asarray(record['population_current_positions'])
            if target.ndim!=2 or target.shape[1]!=2 or not np.isfinite(target).all():
                raise ValueError('every H0 row needs its native supported current target')
            batch=collate([record],'cpu');captured.clear()
            with torch.no_grad():
                parameters=model(batch['x'],batch['graph'],batch['mask'],batch['global_features'],
                    batch['global_graph'],count_context=torch.as_tensor(context[None]))
            if len(captured)!=1:raise RuntimeError('ambiguous population hidden-feature capture')
            selected.append(dict(root_id=record['root_id'],scene_id=record['scene_id'],anchor=record['anchor_index'],
                hidden=captured[0][0],logits=parameters['total_count_tilt_logits'][0].numpy().copy(),
                rate=float(parameters['rate'][0]),count=len(target),weight=1./totals[record['root_id']]))
    finally:hook.remove()
    if not selected:raise ValueError('no supported H0 anchors in this declared fold')
    arrays={key:np.asarray([r[key] for r in selected]) for key in selected[0]}
    arrays['weight']=arrays['weight']/arrays['weight'].sum()
    arrays['design']=np.c_[arrays['hidden'].astype(np.float64),np.ones(len(selected))]
    arrays['log_classes'],arrays['log_individual']=exact_h0_log_probabilities(arrays['logits'],arrays['rate'],arrays['count'])
    return arrays


def readout(data,flat):
    matrix=np.asarray(flat).reshape(7,data['design'].shape[1])
    delta=np.c_[data['design']@matrix.T,np.zeros(len(data['design']))]
    lognorm=logsumexp(data['log_classes']+delta,axis=1)
    nll=-float(data['weight']@(data['log_individual']+delta[np.arange(len(delta)),np.minimum(data['count'],7)]-lognorm))
    return dict(active_anchors=len(delta),active_roots=len(set(data['root_id'])),scenes=len(set(data['scene_id'])),
        exact_individual_current_count_nll=nll,
        arithmetic='float64 evaluation of the supplied (possibly float32-rounded) affine parameters',
        mean_tail_probability=float(data['weight']@np.exp(data['log_classes'][:,-1]-lognorm)),
        offset_weighted_rms=float(np.sqrt(data['weight']@np.square(delta).sum(1))))


def actual_float32_readout(data,model):
    # The cached hidden vectors came from the unchanged CPU1, one-row public
    # forward. Replay the actual one-row F.linear and float32 logit addition.
    changed=[]
    with torch.no_grad():
        for hidden,logits in zip(data['hidden'],data['logits']):
            correction=model.empty_count_affine(torch.as_tensor(hidden[None]))[0].numpy()
            changed.append(logits+np.r_[correction,np.float32(0.)])
    _,individual=exact_h0_log_probabilities(np.asarray(changed),data['rate'],data['count'])
    return dict(exact_individual_current_count_nll=float(-data['weight']@individual),
        arithmetic='analytic individual-count score from actual CPU1 one-row float32 correction/addition at cached unchanged public hidden vectors')


def run(name,initial,split,sidecar,full_fit=False,penalty=.001,maxiter=300):
    if penalty!=.001 or maxiter!=300:raise ValueError('pilot calibration optimizer is predeclared, not a tuning interface')
    allowed={'train_pilot_native_v2':'train_pilot_conditioned','train_campaign_20261023':'train_campaign_20261023'}
    if allowed.get(split)!=sidecar:raise ValueError('only the declared eligible fitting datasets may train this component')
    source,manifest=guarded_source();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    destination=OUT/'experiments'/name
    destination.mkdir(exist_ok=False)
    saved=torch.load(initial,map_location='cpu');model_config=dict(saved['model_config'])
    prepared_sha=sha(OUT/'prepared'/split/'manifest.json')
    if saved['config']['data_manifest_sha256']!=prepared_sha or bool(saved['config']['full_fit'])!=bool(full_fit):
        raise ValueError('base checkpoint and calibration must use the same prepared dataset and fitting fold')
    if model_config.get('total_count_tilt_cutoff')!=7 or model_config.get('empty_history_count_affine',False):
        raise ValueError('initialize from the unchanged eight-potential model')
    model_config['empty_history_count_affine']=True
    model=HumanGraphPredictor(**model_config).cpu().eval();missing=model.load_state_dict(saved['model'],strict=False)
    if missing.unexpected_keys or set(missing.missing_keys)!={'empty_count_affine.weight','empty_count_affine.bias'}:
        raise ValueError('unexpected base-to-affine initialization mismatch')
    model.requires_grad_(False)
    records=load_records(split)
    audits=dict(history=attach_histories(records,sidecar),events=attach_metadata(records,sidecar),
        targets=attach_targets(records,'train_campaign_20261023'))
    fitting=[r for r in records if full_fit or not root_validation(r['root_id'])]
    validation=[r for r in records if root_validation(r['root_id'])]
    if (len({r['root_id'] for r in fitting})!=saved['config']['fit_roots']
            or len(fitting)!=saved['config']['fit_anchors']):
        raise ValueError('calibration fitting Root/anchor boundary differs from its frozen base')
    digest=lambda rows:hashlib.sha256(json.dumps(sorted({r['root_id'] for r in rows}),separators=(',',':')).encode()).hexdigest()
    config=dict(saved['config'],name=name,split=split,sidecar=sidecar,full_fit=full_fit,
        initial_checkpoint=str(initial),initial_sha256=sha(initial),model_config=model_config,
        epochs=1,optimizer='deterministic scipy L-BFGS-B, exact full-H0-anchor objective',
        objective='Exact H0 individual current COUNT NLL plus .5*.001*||affine weights and bias||^2; all original parameters frozen',
        fit_roots=len({r['root_id'] for r in fitting}),fit_anchors=len(fitting),
        validation_roots=len({r['root_id'] for r in validation}),validation_anchors=len(validation),
        validation_role='IN_SAMPLE_NOT_SELECTION' if full_fit else 'EXPOSED_FITTING_ROOT_HELD_DIAGNOSTIC',
        selection_rule='one predeclared completed solver; no held iteration/hyperparameter selection',
        minimum_selection_epoch=1,optimized_parameter_names=['empty_count_affine.weight','empty_count_affine.bias'],
        trainable_parameters=['empty_count_affine.weight','empty_count_affine.bias'],
        count_initialization=None,count_initialization_sha256=None,
        count_initialization_mapping='zero affine correction; all original model tensors copied verbatim',
        learning_rate=None,weight_decay=None,gradient_clip=None,iid_worlds=0,validation_iid_worlds=0,
        fitting_Root_ids_sha256=digest(fitting),validation_Root_ids_sha256=digest(validation),
        data_manifest_sha256=prepared_sha,data_audits=audits,
        penalty=penalty,maxiter=maxiter,gtol=1e-8,ftol=1e-14,
        source_identity=manifest['identity'],source_manifest_sha256=sha(source/'manifest.json'),
        source_hashes={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
        environment=dict(torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,torch_threads=1,device='CPU'))
    config.update(git_commit=subprocess.check_output(['git','-C',str(BASE),'rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','-C',str(BASE),'status','--porcelain'],text=True))
    dump(destination/'config.json',config);dump(destination/'data_join_audit.json',audits)
    started=time.perf_counter();fit=encode_h0(fitting,model);val=encode_h0(validation,model)
    for label,values in [('fit',fit),('validation',val)]:
        np.savez_compressed(destination/(label+'_H0.npz'),**values)
    zero=np.zeros(7*(model.width+1));ledger=[]
    objective=lambda x:affine_objective(x,fit['design'],fit['log_classes'],fit['log_individual'],fit['count'],fit['weight'],penalty)
    def callback(x):
        value,gradient=objective(x);ledger.append(dict(iteration=len(ledger)+1,objective=value,gradient_inf=float(np.max(np.abs(gradient)))))
        np.save(destination/'resumable_affine_iterate.npy',x)
    result=minimize(objective,zero,jac=True,method='L-BFGS-B',callback=callback,
        options=dict(maxiter=maxiter,gtol=1e-8,ftol=1e-14,maxls=50))
    matrix=result.x.reshape(7,model.width+1)
    with torch.no_grad():
        model.empty_count_affine.weight.copy_(torch.as_tensor(matrix[:,:-1],dtype=torch.float32))
        model.empty_count_affine.bias.copy_(torch.as_tensor(matrix[:,-1],dtype=torch.float32))
    for key,value in saved['model'].items():
        if not torch.equal(model.state_dict()[key],value):raise RuntimeError('changed frozen base: '+key)
    rounded=np.c_[model.empty_count_affine.weight.numpy(),model.empty_count_affine.bias.numpy()].astype(np.float64).ravel()
    report=dict(status='TRAINING_COMPLETE_NOT_ACCEPTANCE',source_identity=manifest['identity'],
        solver_success=bool(result.success),solver_message=str(result.message),iterations=int(result.nit),
        gradient_inf=float(np.max(np.abs(result.jac))),
        strong_convexity_objective_gap_bound=float(result.jac@result.jac/(2*penalty)),
        original_parameter_tensors_unchanged=len(saved['model']),
        fitting_before=readout(fit,zero),fitting_after=readout(fit,rounded),
        validation_before=readout(val,zero),validation_after=readout(val,rounded),
        fitting_actual_float32=actual_float32_readout(fit,model),
        validation_actual_float32=actual_float32_readout(val,model),
        all_eligible_fit_roots=len({r['root_id'] for r in fitting}),all_eligible_fit_anchors=len(fitting),
        all_eligible_fit_scenes=len({r['scene_id'] for r in fitting}),
        validation_is_in_sample=bool(full_fit),seconds=time.perf_counter()-started,
        fit_cache_sha256=sha(destination/'fit_H0.npz'),validation_cache_sha256=sha(destination/'validation_H0.npz'))
    checkpoint=dict(model=model.state_dict(),model_config=model_config,config=config,
        conditioned_config=saved['conditioned_config'],epoch=1,step=int(result.nit),
        optimizer=dict(method='scipy L-BFGS-B',solution=result.x,gradient=result.jac,penalty=penalty),
        report=report)
    torch.save(checkpoint,destination/'best.pt');dump(destination/'optimizer_ledger.json',ledger)
    dump(destination/'report.json',report)
    guarded_source()
    dump(destination/'complete.json',dict(status='TRAINING_COMPLETE_NOT_ACCEPTANCE',selected_checkpoint='best.pt',
        checkpoint_sha256=sha(destination/'best.pt'),epochs=1,steps=int(result.nit),seconds=report['seconds'],
        unique_fit_roots=len({r['root_id'] for r in fitting}),fit_anchors=len(fitting),
        active_affine_fit_roots=report['fitting_after']['active_roots'],active_affine_fit_anchors=len(fit['count'])))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--split',default='train_pilot_native_v2');parser.add_argument('--sidecar',default='train_pilot_conditioned')
    parser.add_argument('--full-fit',action='store_true');args=parser.parse_args()
    run(args.name,args.initial,args.split,args.sidecar,args.full_fit)
