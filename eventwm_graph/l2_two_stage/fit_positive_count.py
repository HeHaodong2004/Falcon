"""Resumable proper current-SET calibration of the optional H+ count correction.

Only actual nonempty positive histories have a gradient for this correction.
Root/anchor importance weights preserve the original all-Root objective. H0
anchors remain in every denominator/provenance ledger and have zero correction
gradient. No future loss, association sampling or parameter-dependent cache is
introduced here; the existing current-likelihood loss supplies its proper
accepted-current negative phase on every forward.
"""
import argparse
from dataclasses import dataclass
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
import scipy
import torch

from .association import AssociationGraph
from .common import BASE,OUT,dump,sha
from .conditioned_training import make_batch,loss
from .data import load_records
from .fit_empty_count import guarded_source
from .future_population_targets import attach_targets
from .history_proposal import augment_torch
from .last_event_metadata import attach_metadata
from .mixed_count import legal_context
from .neural import HumanGraphPredictor
from .positive_history import attach_histories
from .tilted_count import LOGITS
from .train import root_validation


HEAD=('positive_count_affine.weight','positive_count_affine.bias')
ALLOWED_DATA={'train_pilot_native_v2':'train_pilot_conditioned',
              'train_campaign_20261023':'train_campaign_20261023'}


def digest_roots(roots):
    return hashlib.sha256(json.dumps(sorted(roots),separators=(',',':')).encode()).hexdigest()


def validate_settings(mode,split,sidecar,epochs=64,seed=413,batch_size=48,
                      learning_rate=.001,worlds=8,validation_worlds=32,full_fit=False):
    if mode not in ('pooled','intercept'):raise ValueError('positive correction mode must be pooled or intercept')
    if ALLOWED_DATA.get(split)!=sidecar:raise ValueError('only declared fitting datasets/sidecars may train')
    if (type(epochs) is not int or epochs!=64 or type(seed) is not int or seed!=413 or
            type(batch_size) is not int or batch_size!=48 or
            type(worlds) is not int or worlds!=8 or
            type(validation_worlds) is not int or validation_worlds!=32 or
            type(learning_rate) not in (int,float) or learning_rate!=.001 or type(full_fit) is not bool):
        raise ValueError('positive-count pilot requires64epochs/seed413/batch48/lr.001/W8/W32')


@dataclass(frozen=True)
class PositiveSchedule:
    all_fit: object
    active_fit: object
    all_validation: object
    validation: tuple
    full_fit: bool

    @property
    def cycle(self):return max(map(len,self.active_fit.values()))

    def epoch(self,epoch):
        if type(epoch) is not int or epoch<1:raise ValueError('epoch is a positive integer')
        ratio=len(self.active_fit)/len(self.all_fit)
        selected=[]
        for root,rows in self.active_fit.items():
            offset=int(hashlib.sha256(('anchor-cycle:'+root).encode()).hexdigest()[:8],16)
            record=rows[(epoch-1+offset)%len(rows)]
            selected.append(dict(record,weight=ratio*len(rows)/len(self.all_fit[root])))
        return selected

    def identities(self):
        def anchors(group):return sum(map(len,group.values()))
        def ledger(group):
            rows=sorted((root,r['scene_id'],int(r['anchor_index']),float(r['timestamp']))
                        for root,records in group.items() for r in records)
            return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()
        return dict(fit_roots=len(self.all_fit),fit_anchors=anchors(self.all_fit),
            active_fit_roots=len(self.active_fit),active_fit_anchors=anchors(self.active_fit),
            inactive_fit_roots=len(self.all_fit)-len(self.active_fit),
            fit_scenes=len({r['scene_id'] for rows in self.all_fit.values() for r in rows}),
            validation_roots=len(self.all_validation),validation_anchors=anchors(self.all_validation),
            active_validation_anchors=len(self.validation),
            active_validation_roots=len({r['root_id'] for r in self.validation}),
            fitting_Root_ids_sha256=digest_roots(self.all_fit),validation_Root_ids_sha256=digest_roots(self.all_validation),
            all_fitting_anchor_ledger_sha256=ledger(self.all_fit),
            all_validation_anchor_ledger_sha256=ledger(self.all_validation),
            minimum_selection_epoch=self.cycle)


def prepare_schedule(records,graphs,full_fit=False):
    if type(full_fit) is not bool:raise ValueError('full_fit must be boolean')
    grouped={};seen=set()
    for record in records:
        root=record['root_id'];key=(root,record['anchor_index'])
        if key in seen:raise ValueError('duplicate prepared Root/anchor')
        seen.add(key);grouped.setdefault(root,[]).append(record)
        graph=graphs[key];graph.validate_certificate()
        actual=int(record['positive_history'].positive_offsets[-1])
        if actual!=graph.detections or (actual>0)!=(graph.minimum_current_count>0):
            raise ValueError('H+ schedule differs from certified complete actual positive history')
    for root,rows in grouped.items():
        rows.sort(key=lambda r:(r['timestamp'],r['anchor_index']))
        if (len({r['scene_id'] for r in rows})!=1 or
                any(b['timestamp']<=a['timestamp'] or b['anchor_index']<=a['anchor_index'] for a,b in zip(rows,rows[1:]))):
            raise ValueError('Root histories/scenes are not causal')
    fit={root:tuple(rows) for root,rows in grouped.items() if full_fit or not root_validation(root)}
    held={root:tuple(rows) for root,rows in grouped.items() if root_validation(root)}
    active={root:tuple(r for r in rows if int(r['positive_history'].positive_offsets[-1])>0)
            for root,rows in fit.items()}
    active={root:rows for root,rows in active.items() if rows}
    validation=tuple(dict(r,weight=1./len(rows)) for rows in held.values()
                     for r in rows if int(r['positive_history'].positive_offsets[-1])>0)
    if not fit or not active or not held or not validation:
        raise ValueError('declared fold requires fitting and held positive-history support')
    return PositiveSchedule(fit,active,held,validation,full_fit)


def validate_base_provenance(saved,prepared_sha,audits,plan,split,sidecar):
    config=saved.get('config',{});identity=plan.identities()
    if (config.get('data_manifest_sha256')!=prepared_sha or config.get('full_fit') is not plan.full_fit or
            config.get('split')!=split or config.get('sidecar')!=sidecar):
        raise ValueError('T base and positive calibration require identical prepared data/fold')
    for field in ('fit_roots','fit_anchors','validation_roots','validation_anchors',
                  'fitting_Root_ids_sha256','validation_Root_ids_sha256'):
        if config.get(field)!=identity[field]:raise ValueError('T base Root/anchor boundary differs: '+field)
    if config.get('data_audits')!=audits:raise ValueError('T base current-target/history/event joins differ')
    if (saved.get('model_config',{}).get('total_count_tilt_cutoff')!=7 or
            saved['model_config'].get('empty_history_count_affine') is not True or
            saved['model_config'].get('positive_history_count_affine') is not None):
        raise ValueError('initialize only from T with its H0 correction and no positive correction')


def initialize_model(saved,mode,device='cuda:0'):
    config=dict(saved['model_config'],positive_history_count_affine=mode)
    model=HumanGraphPredictor(**config).to(device)
    incompatible=model.load_state_dict(saved['model'],strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys)!=set(HEAD):
        raise ValueError('T initialization may miss exactly the two positive-count tensors')
    head=model.positive_count_affine
    if (head.in_features,head.out_features)!=(model.width+3,7):raise ValueError('positive affine shape differs from declared model')
    with torch.no_grad():head.weight.zero_();head.bias.zero_()
    model.requires_grad_(False);head.requires_grad_(True)
    if tuple(name for name,p in model.named_parameters() if p.requires_grad)!=HEAD:
        raise ValueError('only positive-count affine tensors may train')
    for name,value in saved['model'].items():
        if not torch.equal(model.state_dict()[name].cpu(),value.cpu()):raise ValueError('changed T tensor: '+name)
    return model,config


def verify_resume(previous,config):
    if previous.get('resume_exact') is not True:raise ValueError('resume requires a complete epoch checkpoint')
    old=previous.get('config',{})
    ignored={'name','resume','resume_sha256','git_commit','git_status'}
    for key,value in config.items():
        if key not in ignored and old.get(key)!=value:raise ValueError('resume identity mismatch: '+key)
    if type(previous.get('epoch')) is not int or not 0<=previous['epoch']<config['epochs']:
        raise ValueError('resume epoch is outside the declared schedule')


def historical_best_path(directory,learning,cycle):
    """Keep older exact resumes independent of a later overwritten best alias."""
    eligible=[row for row in learning if row['epoch']>=cycle]
    if not eligible:return None
    best=min(eligible,key=lambda row:row['validation_score'])
    path=Path(directory)/('best_epoch%03d.pt'%best['epoch'])
    if not path.is_file():raise ValueError('resume requires the immutable historical selected checkpoint')
    if torch.load(path,map_location='cpu')['epoch']!=best['epoch']:
        raise ValueError('historical selected checkpoint has the wrong epoch')
    return path


def weighted_batch(rows,device='cuda:0'):
    batch=make_batch(rows,device=device)
    weights=np.asarray([r['weight'] for r in rows],np.float64)
    if not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('positive finite Root/anchor weights required')
    batch['weight']=torch.as_tensor(weights,device=device)
    return batch


def run(name,initial,mode,split='train_pilot_native_v2',sidecar='train_pilot_conditioned',
        epochs=64,seed=413,batch_size=48,learning_rate=.001,worlds=8,validation_worlds=32,
        full_fit=False,resume=None,analytic_current_cover=False):
    validate_settings(mode,split,sidecar,epochs,seed,batch_size,learning_rate,worlds,validation_worlds,full_fit)
    if type(analytic_current_cover) is not bool:raise ValueError('analytic_current_cover must be boolean')
    if not name or name in ('.','..') or Path(name).name!=name:raise ValueError('experiment name must be one directory name')
    source,manifest=guarded_source()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('this fitting configuration requires the available CUDA training device')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed);initial=Path(initial);saved=torch.load(initial,map_location='cpu')
    records=load_records(split)
    audits=dict(history=attach_histories(records,sidecar),events=attach_metadata(records,sidecar),
                targets=attach_targets(records,'train_campaign_20261023'))
    print(json.dumps(dict(status='building_positive_history_schedule',all_anchors=len(records))),flush=True)
    graphs={};previous={}
    for record in sorted(records,key=lambda r:(r['root_id'],r['timestamp'],r['anchor_index'])):
        root=record['root_id'];history=record['positive_history']
        graph=previous[root].append(history) if root in previous else AssociationGraph.from_history(history)
        graphs[root,record['anchor_index']]=graph;previous[root]=graph
    plan=prepare_schedule(records,graphs,full_fit);identity=plan.identities()
    if epochs<plan.cycle:raise ValueError('training must complete every active Root anchor cycle')
    prepared_sha=sha(OUT/'prepared'/split/'manifest.json')
    validate_base_provenance(saved,prepared_sha,audits,plan,split,sidecar)
    model,model_config=initialize_model(saved,mode)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=learning_rate,weight_decay=.001)
    conditioned_config=dict(saved['conditioned_config'])
    config=dict(schema='history_conditioned_positive_count_training_1',name=name,mode=mode,
        split=split,sidecar=sidecar,epochs=epochs,seed=seed,batch_size=batch_size,full_fit=full_fit,
        model_config=model_config,conditioned_config=conditioned_config,learning_rate=learning_rate,
        optimizer='AdamW',weight_decay=.001,gradient_clip=5.,iid_worlds=worlds,validation_iid_worlds=validation_worlds,
        current_likelihood_only=True,analytic_empty_normalizer=False,trainable_parameters=list(HEAD),
        analytic_current_cover=analytic_current_cover,
        analytic_current_cover_semantics='When enabled, certify full causal DAG minimum == actual current positive multiplicity and canonical current-source cover; exact logZA0/negative0 with no latent draws. Other rows retain MC. RNG consumption changes explicitly.',
        objective='Normalized current SET-density NLL; positive-count-head gradient equals conditional count NLL. H0 correction gradient is exactly zero.',
        training_weight='active_Roots/all_fit_Roots * Hplus_anchors_in_Root/all_anchors_in_Root; mean over uniform active Root with causal Hplus anchor cycle',
        validation_weight='1/all_anchors_in_held_Root for every original Hplus held anchor',
        validation_score='weighted Hplus current SET-density NLL sum / ALL held Roots; omitted H0 term is parameter-independent constant',
        selection_rule='fixed_last_epoch' if full_fit else 'minimum_held_proper_SET_score_after_full_active_anchor_cycle',
        validation_role='IN_SAMPLE_NOT_SELECTION' if full_fit else 'EXPOSED_FITTING_ROOT_HELD_SELECTION',
        initial_checkpoint=str(initial),initial_sha256=sha(initial),initial_model_tensor_count=len(saved['model']),
        data_manifest_sha256=prepared_sha,data_audits=audits,source_identity=manifest['identity'],
        data_identity_semantics='data_manifest_sha256 identifies prepared training records; each data_audits entry binds its independently joined history/events/native-target manifest',
        source_manifest_sha256=sha(source/'manifest.json'),
        source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0),torch_threads=1,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),**identity)
    config.update(git_commit=subprocess.check_output(['git','-C',str(BASE),'rev-parse','HEAD'],text=True).strip(),
                  git_status=subprocess.check_output(['git','-C',str(BASE),'status','--porcelain'],text=True))
    start_epoch=0;steps=0;learning=[];best=math.inf
    if resume is not None:
        resume=Path(resume);checkpoint=torch.load(resume,map_location='cpu');verify_resume(checkpoint,config)
        if sha(resume.parent/'config.json')!=checkpoint.get('config_sha256'):
            raise ValueError('resume configuration file identity differs from its checkpoint')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch=checkpoint['epoch'];steps=checkpoint['steps'];rng.bit_generator.state=checkpoint['sampling_rng']
        torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state(checkpoint['cuda_rng'])
        np.random.set_state(checkpoint['numpy_rng']);random.setstate(checkpoint['python_rng'])
        learning_path=resume.parent/'learning.json'
        if start_epoch and not learning_path.is_file():raise ValueError('resume requires its completed learning ledger')
        if learning_path.is_file():learning=[r for r in json.loads(learning_path.read_text()) if r['epoch']<=start_epoch]
        best=min((r['validation_score'] for r in learning if r['epoch']>=plan.cycle),default=math.inf)
        config.update(resume=str(resume),resume_sha256=sha(resume))
    destination=OUT/'experiments'/name;destination.mkdir(parents=True,exist_ok=False)
    if not full_fit and math.isfinite(best):
        prior=historical_best_path(resume.parent,learning,plan.cycle)
        shutil.copyfile(prior,destination/'best.pt')
        shutil.copyfile(prior,destination/prior.name)
    dump(destination/'config.json',config);dump(destination/'data_join_audit.json',audits)
    dump(destination/'learning.json',learning)
    (destination/'source').mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copyfile(path,destination/'source'/path.name)
    def save(filename,epoch,exact=True):
        state={name:value.detach().cpu() for name,value in model.state_dict().items()}
        for key,value in saved['model'].items():
            if not torch.equal(state[key],value.cpu()):raise RuntimeError('changed frozen T tensor: '+key)
        checkpoint=dict(model=state,model_config=model_config,conditioned_config=conditioned_config,
            optimizer=optimizer.state_dict(),epoch=epoch,steps=steps,config=config,resume_exact=exact,
            config_sha256=sha(destination/'config.json'),
            sampling_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate())
        temporary=destination/(filename+'.tmp');torch.save(checkpoint,temporary);temporary.replace(destination/filename)
    save('epoch0.pt',start_epoch);started=time.perf_counter();prior_seconds=learning[-1]['seconds'] if learning else 0.
    def forward(rows,sampling_rng,iid_worlds):
        batch=weighted_batch(rows);states=[graphs[r['root_id'],r['anchor_index']] for r in rows]
        contexts=torch.as_tensor(np.asarray([legal_context(r,r['positive_history'],g) for r,g in zip(rows,states)]),device=batch['x'].device)
        parameters=model(batch['x'],batch['graph'],batch['mask'],batch['global_features'],batch['global_graph'],count_context=contexts)
        parameters=augment_torch(parameters,rows,states,conditioned_config['history_fraction'])
        if any(value.requires_grad for key,value in parameters.items() if key!=LOGITS):
            raise RuntimeError('a frozen base output unexpectedly requires gradient')
        return loss(parameters,batch,rows,states,sampling_rng,worlds=iid_worlds,
            guidance=False,current_likelihood_only=True,analytic_empty_normalizer=False,
            analytic_current_cover=analytic_current_cover)
    def weighted_terms(terms,rows):
        weights=np.asarray([r['weight'] for r in rows])
        return {key:float(weights@(value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)))
                for key,value in terms.items()}
    for epoch in range(start_epoch+1,epochs+1):
        rows=[]
        try:
            selected=plan.epoch(epoch);order=np.random.permutation(len(selected));model.train()
            numerator=0.;train_terms={};norms=[]
            for begin in range(0,len(order),batch_size):
                rows=[selected[int(i)] for i in order[begin:begin+batch_size]]
                objective,terms=forward(rows,rng,worlds)
                if not torch.isfinite(objective):raise RuntimeError('nonfinite current SET training score')
                optimizer.zero_grad(set_to_none=True);objective.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.positive_count_affine.parameters(),5.)
                if not torch.isfinite(norm):raise RuntimeError('nonfinite positive-count gradient')
                norms.append(float(norm.detach()));optimizer.step();steps+=1
                numerator+=float(objective.detach())*len(rows)
                for key,value in weighted_terms(terms,rows).items():train_terms[key]=train_terms.get(key,0.)+value
                if steps%25==0:print(json.dumps(dict(status='training',epoch=epoch,steps=steps,
                    active_roots_seen=min(begin+batch_size,len(selected)),seconds=prior_seconds+time.perf_counter()-started)),flush=True)
            model.eval();validation_sum=0.;validation_terms={}
            cpu_state=torch.get_rng_state();cuda_state=torch.cuda.get_rng_state()
            try:
                torch.manual_seed(seed+10000);torch.cuda.manual_seed(seed+10000)
                validation_rng=np.random.default_rng(seed+10000)
                with torch.no_grad():
                    for begin in range(0,len(plan.validation),batch_size):
                        rows=plan.validation[begin:begin+batch_size]
                        objective,terms=forward(rows,validation_rng,validation_worlds)
                        if not torch.isfinite(objective):raise RuntimeError('nonfinite held current SET score')
                        validation_sum+=float(objective)*len(rows)
                        for key,value in weighted_terms(terms,rows).items():validation_terms[key]=validation_terms.get(key,0.)+value
                        if begin and begin%(25*batch_size)==0:print(json.dumps(dict(status='validation',epoch=epoch,
                            active_anchors_seen=min(begin+batch_size,len(plan.validation)),seconds=prior_seconds+time.perf_counter()-started)),flush=True)
            finally:torch.set_rng_state(cpu_state);torch.cuda.set_rng_state(cuda_state)
            score=validation_sum/len(plan.all_validation)
            row=dict(epoch=epoch,steps=steps,training_score=numerator/len(selected),validation_score=score,
                validation_positive_conditional_score=validation_sum/sum(r['weight'] for r in plan.validation),
                training_terms={k:v/len(selected) for k,v in train_terms.items()},
                validation_terms={k:v/len(plan.all_validation) for k,v in validation_terms.items()},
                active_roots_seen=len(selected),seconds=prior_seconds+time.perf_counter()-started,
                preclip_gradient_norm=dict(mean=float(np.mean(norms)),p95=float(np.quantile(norms,.95)),
                    maximum=float(np.max(norms)),clipped_fraction=float(np.mean(np.asarray(norms)>5.))))
            learning.append(row);dump(destination/'learning.json',learning)
            if not full_fit and epoch>=plan.cycle and score<best:
                best=score;filename='best_epoch%03d.pt'%epoch;save(filename,epoch)
                shutil.copyfile(destination/filename,destination/'best.pt')
            save('last.pt',epoch)
            print(json.dumps(row),flush=True)
        except Exception as error:
            save('failed_mid_epoch.pt',epoch,False)
            dump(destination/'failure.json',dict(epoch=epoch,steps=steps,error=repr(error),
                records=[dict(root_id=r['root_id'],anchor=r['anchor_index']) for r in rows],
                exact_resume_checkpoint='last.pt' if epoch>start_epoch+1 else 'epoch0.pt'))
            raise
    guarded_source();selected='last.pt' if full_fit else 'best.pt'
    dump(destination/'complete.json',dict(status='TRAINING_COMPLETE_NOT_ACCEPTANCE',selected_checkpoint=selected,
        checkpoint_sha256=sha(destination/selected),epochs=epochs,steps=steps,seconds=prior_seconds+time.perf_counter()-started,
        config_sha256=sha(destination/'config.json'),
        unique_fit_roots=len(plan.all_fit),fit_anchors=identity['fit_anchors'],active_fit_roots=len(plan.active_fit),
        active_fit_anchors=identity['active_fit_anchors'],original_parameter_tensors_unchanged=len(saved['model']),
        peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--mode',choices=['pooled','intercept'],required=True)
    parser.add_argument('--split',default='train_pilot_native_v2');parser.add_argument('--sidecar',default='train_pilot_conditioned')
    parser.add_argument('--resume',type=Path);parser.add_argument('--full-fit',action='store_true')
    parser.add_argument('--analytic-current-cover',action='store_true')
    run(**vars(parser.parse_args()))
