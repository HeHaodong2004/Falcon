#!/usr/bin/env python3
"""Read-only sealed V4 research snapshot and unchanged EventQueryWM scaling.

Production APIs are deliberately never imported. Live SQLite connections use
mode=ro and query_only; only finalized, independently verified roots are frozen.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from eventquerywm.inputs import ModelInput, tensorize, collate, input_hash, INPUT_VERSION
from eventquerywm.model import EventQueryWM, ModelConfig
from eventquerywm.sr_metrics import metrics
from eventquerywm.v2.labels import inspect_social_root, outcome_target
from collector.core import load_wm_input, digest

PROJECT = Path('/home/haodong/workspace/projects/eventwm-falcon')
DATA = Path('/tmp/eventwm_v4_300k')
VERIFIED = PROJECT/'output/eventwm_300k_combined_index/verified.sqlite3'
DEFAULT_OUT = PROJECT/'output/eventwm_v4_200k_research_20261012'
SPLIT_SEED = 20261012
BANK_INDEX = PROJECT/'output/first_training_checkout/output/first_training/index.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def ro(path):
    db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True, timeout=10)
    db.execute('PRAGMA query_only=ON')
    return db


def production_sample():
    with ro(DATA/'pipeline.sqlite3') as db:
        db.execute('BEGIN')
        m = json.loads(db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()[0])
        p = json.loads(db.execute("SELECT value FROM meta WHERE key='progress'").fetchone()[0])
    pid = m['runner']['pid']
    return dict(timestamp=time.time(),progress=p,runner_pid=pid,
                runner_alive=Path(f'/proc/{pid}').exists(),state=m['state'],
                native_limit=read(DATA/'native_limit.json')['limit'],
                source_commit=m['frozen_commit'],pipeline_settings=m['pipeline_settings'])


def key(value):
    return hashlib.sha256(f'{SPLIT_SEED}/{value}'.encode()).hexdigest()


def strict_splits(blocks, validation_fraction=.1, extra_protected=()):
    """Preserve reference roles; quarantine their scenes from optimization."""
    protected = {b['root']['physical_scene_id'] for b in blocks if b['root']['role'] != 'train'} | set(extra_protected)
    eligible = sorted({b['root']['physical_scene_id'] for b in blocks
                       if b['root']['role'] == 'train'}-protected, key=key)
    val = set(eligible[:max(1, round(len(eligible)*validation_fraction))])
    for b in blocks:
        root = b['root']
        role, scene = root['role'], root['physical_scene_id']
        split = (role if role != 'train' else 'quarantine_reference_scene'
                 if scene in protected else 'validation' if scene in val else 'train')
        b['split'] = split
        for r in b['rows']:
            r['split'] = split
    sets = {s:{b['root']['physical_scene_id'] for b in blocks if b['split'] == s}
            for s in ('train','validation','natural_reference','unseen_evaluation')}
    roots = {s:{b['root']['root_id'] for b in blocks if b['split'] == s} for s in sets}
    for a in sets:
        for c in sets:
            if a != c and (sets[a]&sets[c] or roots[a]&roots[c]):
                raise ValueError('Scene or root leakage: '+a+'/'+c)
    for field in ('trajectory_id','episode_id','fixture_id'):
        groups = {s:{r[field] for b in blocks for r in b['rows']
                     if b['split'] == s and r.get(field)} for s in sets}
        for a in groups:
            for c in groups:
                if a != c and groups[a]&groups[c]:
                    raise ValueError('Fixture/trajectory leakage: '+field)
    return sets


def reserve_bank(args):
    """Finalize splits before preprocessing, including the saved selector bank."""
    out=args.output
    assert not (out/'cache/manifest.json').exists() and not (out/'runs').exists()
    manifest=read(out/'snapshot.json')
    assert not manifest['audit'].get('reserved_selector_bank')
    index=read(BANK_INDEX)
    bank=[r for r in index['rows'] if r['split']=='validation']
    bank_scenes={r['scene'] for r in bank}
    strict_splits(manifest['blocks'],extra_protected=bank_scenes)
    blocks=manifest['blocks']
    manifest['audit']['splits']={s:dict(roots=sum(b['split']==s for b in blocks),queries=sum(len(b['rows']) for b in blocks if b['split']==s),
                          scenes=len({b['root']['physical_scene_id'] for b in blocks if b['split']==s}))
                    for s in sorted({b['split'] for b in blocks})}
    eligible=sorted([b['root']['root_id'] for b in blocks if b['split']=='train'],key=key)
    manifest['subsets']={name:eligible[:min(len(eligible),n)] for name,n in [('30k',round(30000/33)),('50k',round(50000/33)),('all',len(eligible))]}
    manifest['audit']['requested_sizes_feasible']={s:len(eligible)*33>=s for s in (30000,100000,200000)}
    manifest['audit']['reserved_selector_bank']=dict(index=str(BANK_INDEX),sha256=sha(BANK_INDEX),queries=len(bank),roots=len({r['root_id'] for r in bank}),scenes=len(bank_scenes),
            policy='All legacy development evaluation scenes excluded from V4 optimization and validation; benchmark is historical for old model, scene-held-out for new model')
    manifest['audit']['split_policy']+='; additionally reserve every frozen legacy selector-bank scene'
    # Keep the initial audited census for transparent provenance. Neither version
    # has been used for training; this is the final pre-experiment split freeze.
    (out/'snapshot.json').rename(out/'audited_census_before_bank_reservation.json')
    save(out/'snapshot.json',manifest)
    save(out/'audit.json',manifest['audit'])
    save(out/'snapshot_identity.json',dict(sha256=sha(out/'snapshot.json'),source_index=str(VERIFIED),frozen_at=manifest['audit']['cutoff_unix'],bank_reserved_at=time.time()))
    print(json.dumps(dict(stage='bank_reserved',splits=manifest['audit']['splits'])),flush=True)


def snapshot(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'snapshot.json').exists():
        raise ValueError('Snapshot already exists; preserve its cutoff')
    torch.set_num_threads(1)
    started = time.time()
    before = production_sample()
    save(out/'production_before.json', before)
    # Short transactions release the WAL reader before any filesystem hashing.
    with ro(VERIFIED) as db:
        db.execute('BEGIN')
        verified = db.execute('SELECT root_id,path,role,physical_scene,seed,queries,stats FROM roots WHERE dataset=? ORDER BY root_id', (str(DATA),)).fetchall()
        legacy = db.execute('SELECT dataset,role,SUM(queries) FROM roots WHERE dataset!=? GROUP BY dataset,role', (str(DATA),)).fetchall()
    with ro(DATA/'pipeline.sqlite3') as db:
        db.execute('BEGIN')
        m = json.loads(db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()[0])
        progress = json.loads(db.execute("SELECT value FROM meta WHERE key='progress'").fetchone()[0])
        roots = [json.loads(db.execute('SELECT value FROM roots WHERE id=?',(r[0],)).fetchone()[0]) for r in verified]
    epochs = {e['commit'] for e in m['production_source_epochs']}
    blocks, excluded = [], []
    census = defaultdict(Counter)
    all_ids, all_seeds = set(), set()
    for i,(v,root) in enumerate(zip(verified,roots)):
        try:
            assert root['state'] == 'completed'
            assert (root['root_id'],root['path'],root['role'],root['physical_scene_id'],root['seed'],root['stats']['primary_queries']) == v[:6]
            assert root['source_commit'] in epochs
            assert root['seed'] not in all_seeds
            rows = inspect_social_root(DATA,root)
            assert len(rows) == 33 and not all_ids.intersection(r['query_id'] for r in rows)
            path = (DATA/root['path']).resolve()
            assert path.is_dir()
            labels = read(path/'labels/labels.json')
            primary = [l for l in labels if l['branch_id'].startswith('b')]
            queries = read(path/'queries/queries.json')
            history = read(path/'legal_observations/history.json')
            for row in rows:
                row.update(source_commit=root['source_commit'],root_path=str(path))
            all_ids.update(r['query_id'] for r in rows)
            all_seeds.add(root['seed'])
            seals = read(DATA/root['seal_file'])
            evidence = {n:seals[n] for n in ('VALIDATED.json','legal_observations/history.json','queries/queries.json','labels/labels.json','branch_truth/branches.jsonl.gz') if n in seals}
            for l in primary:
                assert l['sr_contract_version'] == 'binary-execution-safety-v1'
                assert bool(l['static_collision']) == bool(l['static_contact'])
                assert not l['static_collision'] and not l['static_feasibility_violation']
                assert l['sr_label'] == int(not l['native_human_collision'])
                census['static_checker'][l['static_checker_version']] += 1
                census['sr'][str(l['sr_label'])] += 1
                census['contact']['human'] += int(l['native_human_collision'])
                census['contact']['static'] += int(l['static_collision'])
                census['termination'][l['termination_reason']] += 1
                census['label_protocol'][l['protocol_version']] += 1
            for q in queries:
                census['query_schema'][str(q['query']['schema_version'])] += 1
            census['source_epoch'][root['source_commit']] += len(rows)
            census['role'][root['role']] += len(rows)
            census['legal_category'][root['stats']['legal_state']['category']] += len(rows)
            census['history_length'][str(len(history))] += 1
            census['successor_files']['sealed_branch_truth'] += int('branch_truth/branches.jsonl.gz' in evidence)*len(rows)
            blocks.append(dict(root=root,rows=rows,evidence=evidence,archive_path=str(path),
                               complete_sha256=sha(path/'COMPLETE.json'),seal_sha256=sha(DATA/root['seal_file'])))
        except (AssertionError,ValueError,KeyError,OSError) as e:
            excluded.append(dict(root_id=root['root_id'],queries=v[5],reason=type(e).__name__+': '+str(e)))
        if (i+1)%250 == 0:
            print(json.dumps(dict(stage='audit',roots=i+1,total=len(roots),excluded=len(excluded),seconds=time.time()-started)),flush=True)
    scenes = strict_splits(blocks)
    split_counts = {s:dict(roots=sum(b['split']==s for b in blocks),queries=sum(len(b['rows']) for b in blocks if b['split']==s),
                          scenes=len({b['root']['physical_scene_id'] for b in blocks if b['split']==s}))
                    for s in sorted({b['split'] for b in blocks})}
    eligible_roots = sorted([b['root']['root_id'] for b in blocks if b['split']=='train'],key=key)
    sizes = {'30k':min(len(eligible_roots),round(30000/33)), '50k':min(len(eligible_roots),round(50000/33)), 'all':len(eligible_roots)}
    subsets = {name:eligible_roots[:n] for name,n in sizes.items()}
    audit = dict(cutoff_unix=started,verified_roots=len(verified),verified_queries=sum(v[5] for v in verified),
                 accepted_roots=len(blocks),accepted_queries=sum(len(b['rows']) for b in blocks),
                 distinct_scenes=len({b['root']['physical_scene_id'] for b in blocks}),
                 invalid_excluded_queries=sum(r['queries'] for r in excluded),invalid_excluded_roots=excluded,
                 production_progress_at_cutoff=progress,legacy_reference_counts=legacy,
                 unfinalized_or_unverified_retained_queries=max(0,progress['primary_queries']-sum(v[5] for v in verified)),
                 census={k:dict(v) for k,v in census.items()},splits=split_counts,
                 split_policy='Preserve every reference root; quarantine TRAIN scenes shared with any reference; SHA256(20261012/physical_scene_id) first 10% remaining scenes for validation; nested root hashes for training subsets',
                 requested_sizes_feasible={s:len(eligible_roots)*33>=s for s in (30000,100000,200000)},
                 schema={k:m[k] for k in ('schema_version','protocol_version','sr_contract_version','prefix_truth_encoding','branch_truth_encoding','contact_serialization','checker')},
                 input_version=INPUT_VERSION,raw_data_copied=False,production_mutated=False,
                 validity_scope='Existing continuous deep-seal verification plus independent completion/seal, validation, legal-history/query/label hashes, query digests, index equality, authoritative SR/social checks. No native replay or re-certification of archived geometry.',
                 seconds=time.time()-started)
    manifest = dict(version='eventwm-v4-research-snapshot-v1',audit=audit,subsets=subsets,blocks=blocks)
    save(out/'snapshot.json',manifest)
    save(out/'audit.json',audit)
    save(out/'snapshot_identity.json',dict(sha256=sha(out/'snapshot.json'),source_index=str(VERIFIED),frozen_at=started))
    save(out/'production_after_snapshot.json',production_sample())
    print(json.dumps(dict(stage='snapshot_complete',splits=split_counts,invalid_excluded=audit['invalid_excluded_queries'])),flush=True)


def prepare(args):
    out = args.output
    manifest = read(out/'snapshot.json')
    assert manifest['audit'].get('reserved_selector_bank'), 'Finalize selector bank reservation first'
    assert sha(out/'snapshot.json') == read(out/'snapshot_identity.json')['sha256']
    destination = out/'cache'
    destination.mkdir(exist_ok=True)
    if (destination/'manifest.json').exists():
        raise ValueError('Cache already finalized')
    shards, buffered, entries = {}, [], []
    started = time.time()
    canonical_parities = 0
    epochs_checked = set()
    reuse_path=getattr(args,'reuse_cache',None)
    reused={b['root_id']:b for b in Cache(reuse_path).blocks} if reuse_path else {}
    reused_roots=0
    def flush():
        if not buffered:
            return
        name = f'roots_{len(shards):04d}.pt'
        torch.save(buffered,destination/name)
        shards[name] = sha(destination/name)
        for offset,b in enumerate(buffered):
            entries.append(dict(root_id=b['root_id'],split=b['split'],shard=name,offset=offset,queries=len(b['rows'])))
        buffered.clear()
    for i,b in enumerate(manifest['blocks']):
        # Quarantined examples remain in the frozen audit, never in model data.
        if b['split'].startswith('quarantine'):
            continue
        path = Path(b['archive_path'])
        for name,h in b['evidence'].items():
            if name.startswith('branch_truth/'):
                continue
            assert sha(path/name) == h
        rows = b['rows']
        old=reused.get(b['root']['root_id'])
        if old:
            assert [r['query_id'] for r in rows]==[r['query_id'] for r in old['rows']]
            assert [r['sr_label'] for r in rows]==[r['sr_label'] for r in old['rows']]
            base,query_tensor=old['base'],old['query'];reused_roots+=1
        else:
            legal = load_wm_input(path,rows[0]['query_id'])
            base = tensorize(legal)
            queries = {q['query']['query_id']:q['query'] for q in read(path/'queries/queries.json')}
            tensors = []
            # Query interpolation depends only on the last decision pose.
            for r in rows:
                q = tensorize(dict(history=[legal['history'][-1]],history_images=None,query=queries[r['query_id']])).query
                tensors.append(q)
            query_tensor = torch.stack(tensors)
        if b['root']['source_commit'] not in epochs_checked:
            for j in (0,15,32):
                full = tensorize(load_wm_input(path,rows[j]['query_id']))
                candidate = ModelInput(base.humans,base.human_padding,base.robot,base.history_padding,query_tensor[j])
                assert input_hash(full) == input_hash(candidate)
                canonical_parities += 1
            epochs_checked.add(b['root']['source_commit'])
        buffered.append(dict(root_id=b['root']['root_id'],split=b['split'],base=base,query=query_tensor,rows=rows))
        if len(buffered)>=64:
            flush()
        if (i+1)%250 == 0:
            print(json.dumps(dict(stage='cache',source_roots=i+1,total=len(manifest['blocks']),shards=len(shards),seconds=time.time()-started)),flush=True)
    flush()
    save(destination/'manifest.json',dict(version='eventwm-v4-compact-legal-v1',entries=entries,shards=shards,
         snapshot_sha256=sha(out/'snapshot.json'),canonical_input_hash_parities=canonical_parities,
         reused_roots=reused_roots,
         scientific_epochs_checked=sorted(epochs_checked),bytes=sum((destination/n).stat().st_size for n in shards),seconds=time.time()-started))
    print(json.dumps(dict(stage='cache_complete',roots=len(entries),shards=len(shards),parities=canonical_parities,seconds=time.time()-started)),flush=True)


class Cache:
    def __init__(self,out):
        self.out=Path(out)
        self.manifest=read(self.out/'cache/manifest.json')
        assert sha(self.out/'snapshot.json')==self.manifest['snapshot_sha256']
        self.blocks=[]
        for name,h in self.manifest['shards'].items():
            path=self.out/'cache'/name
            assert sha(path)==h
            self.blocks.extend(torch.load(path,map_location='cpu'))
        self.samples=[]
        self.rows=[]
        self.splits=defaultdict(list)
        self.byroot=defaultdict(list)
        for b in self.blocks:
            for i,row in enumerate(b['rows']):
                ix=len(self.samples)
                self.samples.append((b,i))
                self.rows.append(row)
                self.splits[b['split']].append(ix)
                self.byroot[b['root_id']].append(ix)

    def batch(self,indices,mode='full',device='cpu'):
        values=[]
        for ix in indices:
            b,j=self.samples[int(ix)]
            v=b['base']
            human_padding=v.human_padding
            if mode=='query_only':
                human_padding=torch.ones_like(human_padding)
            q=torch.zeros_like(b['query'][j]) if mode=='history_only' else b['query'][j]
            values.append(ModelInput(v.humans,human_padding,v.robot,v.history_padding,q))
        return collate(values).to(device)


def counterfactual(rows,z,root_level=False):
    groups=defaultdict(list)
    for row,score in zip(rows,z):
        if row['variant']!='WAIT':
            groups[row['root_id'] if root_level else row['group_id']].append((row['sr_label'],-score))
    wins=pairs=0
    macro=[]
    for members in groups.values():
        positive=np.array([p for y,p in members if y==0])
        negative=np.array([p for y,p in members if y==1])
        if not len(positive) or not len(negative):
            continue
        diff=positive[:,None]-negative[None,:]
        w=float((diff>0).sum()+.5*(diff==0).sum())
        pairs+=diff.size
        wins+=w
        macro.append(w/diff.size)
    return dict(pair_concordance=wins/pairs if pairs else None,macro_group_concordance=float(np.mean(macro)) if macro else None,
                contrasting_pairs=pairs,mixed_groups=len(macro),scope='root' if root_level else 'root/spatial_path')


def score(cache,indices,z):
    rows=[cache.rows[i] for i in indices]
    y=np.array([r['sr_label'] for r in rows])
    p=1/(1+np.exp(-np.clip(z,-700,700)))
    result=metrics(y,p,z)
    result['false_safe_rate_given_collision']=1-result['collision_recall'] if result['collision_recall'] is not None else None
    safe_predictions=p>.5
    result['collision_rate_among_predicted_safe']=float(np.mean(y[safe_predictions]==0)) if safe_predictions.any() else None
    result['within_group_ranking']=counterfactual(rows,z)
    result['within_root_ranking']=counterfactual(rows,z,True)
    result['categories']={}
    for category in sorted({r['category'] for r in rows}):
        keep=np.array([r['category']==category for r in rows])
        result['categories'][category]=metrics(y[keep],p[keep],z[keep])
    return result


@torch.no_grad()
def infer(model,cache,indices,mode,device,batch_size=128):
    model.eval()
    result=[]
    for start in range(0,len(indices),batch_size):
        ix=indices[start:start+batch_size]
        result.append(model(cache.batch(ix,mode,device)).cpu().numpy())
    return np.concatenate(result)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def smoke(args):
    torch.set_num_threads(1)
    seed_all(args.seed)
    with ro(VERIFIED) as db:
        rid,path=db.execute('SELECT root_id,path FROM roots WHERE dataset=? AND role=? ORDER BY root_id LIMIT 1',(str(DATA),'train')).fetchone()
    queries=read(DATA/path/'queries/queries.json')
    values=[tensorize(load_wm_input(DATA/path,q['query']['query_id'])) for q in queries[:4]]
    model=EventQueryWM(ModelConfig.family('M',dropout=.2)).to(args.device)
    value=collate(values).to(args.device)
    started=time.time()
    for i in range(4):
        model.zero_grad(set_to_none=True)
        z=model(value)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.ones_like(z))
        loss.backward()
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    report=dict(device=args.device,cuda_available=torch.cuda.is_available(),gpu_name=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
                parameters=sum(p.numel() for p in model.parameters()),config=asdict(model.config),root_id=rid,
                humans_shape=list(value.humans.shape),robot_shape=list(value.robot.shape),query_shape=list(value.query.shape),
                seconds_4_forward_backward=time.time()-started,finite_loss=bool(torch.isfinite(loss)),
                finite_gradients=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                memory_free_total=torch.cuda.mem_get_info(args.device) if args.device.startswith('cuda') else None,
                production=production_sample())
    save(args.output/'gpu_smoke.json',report)
    print(json.dumps(report),flush=True)


def train(args):
    import math
    torch.set_num_threads(1)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; never silently fall back to long CPU training')
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
    seed_all(args.seed)
    out=args.output
    run=out/'runs'/getattr(args,'name',f'{args.subset}_{args.mode}_seed{args.seed}')
    resume=bool(getattr(args,'resume',False)) and (run/'training_state.pt').exists()
    run.mkdir(parents=True,exist_ok=resume)
    cache=getattr(args,'cache',None) or Cache(out)
    snapshot=read(out/'snapshot.json')
    allowed=set(snapshot['subsets'][args.subset])
    indices=[i for i in cache.splits['train'] if cache.rows[i]['root_id'] in allowed]
    val=cache.splits['validation']
    assert len(indices) and len(val)
    config=ModelConfig.family(getattr(args,'size','M'),dropout=getattr(args,'dropout',.2))
    model=EventQueryWM(config).to(args.device)
    assert all(p.requires_grad for p in model.parameters()), 'Full-model baseline must train the encoder'
    lr=getattr(args,'lr',1e-4);weight_decay=getattr(args,'weight_decay',.01)
    schedule=getattr(args,'schedule','constant')
    foreach=bool(getattr(args,'foreach',False))
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay,foreach=foreach)
    amp=bool(getattr(args,'amp',False))
    scaler=torch.cuda.amp.GradScaler(enabled=amp)
    exposure_epochs=getattr(args,'exposure_epochs',0)
    epoch_steps=math.ceil(len(indices)/64)
    exposure_steps=round(exposure_epochs*epoch_steps)
    if exposure_epochs:args.steps=max(args.fixed_steps,exposure_steps)
    protocol=dict(seed=args.seed,subset=args.subset,mode=args.mode,config=asdict(config),
                  parameters=sum(p.numel() for p in model.parameters()),train_queries=len(indices),train_roots=len(allowed),
                  train_scenes=len({cache.rows[i]['scene'] for i in indices}),unique_train_queries=len({cache.rows[i]['query_id'] for i in indices}),
                  snapshot_sha256=sha(out/'snapshot.json'),loss='unweighted BCEWithLogitsLoss, safe positive',
                  optimizer='AdamW',lr=lr,weight_decay=weight_decay,batch_size=64,gradient_clip=1,
                  schedule=schedule,warmup_steps=256 if schedule=='cosine' else 0,
                  final_lr_fraction=.1 if schedule=='cosine' else 1.,
                  full_model_training=True,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  mixed_precision=amp,data_loader=getattr(args,'loader','legacy_cpu_collate'),
                  foreach_optimizer=foreach,
                  source_commit=getattr(args,'source_commit',subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),
                  sampling='Uniform retained queries without replacement per shuffled epoch',
                  max_steps=args.steps,fixed_comparison_step=args.fixed_steps,
                  exposure_epochs=exposure_epochs,exposure_steps=exposure_steps,
                  eval_every=args.eval_every,early_stopping_patience=args.patience,
                  early_stopping_min_steps=args.fixed_steps,
                  checkpoint_selection='minimum full held-out-root validation BCE',
                  early_stopping_enabled=not bool(exposure_epochs),
                  device=args.device,torch=torch.__version__,dataloader_workers=0,torch_threads=1)
    save(run/'configuration.json',protocol)
    save(run/'production_start.json',production_sample())
    monitor=ProductionGuard(run)
    generator=np.random.default_rng(args.seed+999)
    order=np.array([],dtype=int)
    offset=0
    best=float('inf')
    best_step=0
    stale=0
    curve=[]
    started=time.time()
    presentations=0
    start_step=0
    elapsed_before=0.
    if resume:
        state=torch.load(run/'training_state.pt',map_location=args.device)
        if state['protocol']!=protocol:raise ValueError('Resume requires identical source, snapshot and configuration')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state.get('scaler',{}))
        generator.bit_generator.state=state['generator_state']
        order=state['order'];offset=state['offset'];presentations=state['presentations']
        best=state['best'];best_step=state['best_step'];stale=state['stale'];curve=state['curve']
        start_step=state['step'];elapsed_before=state['elapsed_seconds']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if args.device.startswith('cuda'):torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    def save_state(step,status='running'):
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),protocol=protocol,
            step=step,order=order,offset=offset,presentations=presentations,
            generator_state=generator.bit_generator.state,best=best,best_step=best_step,stale=stale,curve=curve,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
            elapsed_seconds=elapsed_before+time.time()-started)
        temporary=run/'training_state.tmp.pt';torch.save(state,temporary);temporary.replace(run/'training_state.pt')
        save(run/'progress.json',dict(pid=os.getpid(),status=status,step=step,max_steps=args.steps,
             epochs=presentations/len(indices),best_step=best_step,best_validation_bce=best,
             source_commit=protocol['source_commit'],snapshot_sha256=protocol['snapshot_sha256']))
    probe=sorted(indices,key=lambda i:key(cache.rows[i]['query_id']))[:min(4096,len(indices))]
    def evaluate(step):
        nonlocal best,best_step,stale
        vz=infer(model,cache,val,args.mode,args.device)
        tz=infer(model,cache,probe,args.mode,args.device)
        record=dict(step=step,train_probe=score(cache,probe,tz),validation=score(cache,val,vz),
                    wall_seconds=elapsed_before+time.time()-started,sample_presentations=presentations,epochs=presentations/len(indices))
        record['unique_gradient_queries_seen']=min(len(indices),presentations)
        record['unique_gradient_roots_seen']=len(allowed) if presentations>=len(indices) else len({cache.rows[int(i)]['root_id'] for i in order[:offset]})
        curve.append(record)
        save(run/'curve.json',curve)
        ck=dict(model=model.state_dict(),config=asdict(config),step=step,protocol=protocol,
                sample_presentations=presentations,unique_gradient_queries_seen=record['unique_gradient_queries_seen'],
                unique_gradient_roots_seen=record['unique_gradient_roots_seen'])
        if record['validation']['bce'] < best:
            best=record['validation']['bce'];best_step=step;stale=0
            torch.save(ck,run/'best.pt')
        else:
            stale+=1
        if step==args.fixed_steps:
            torch.save(ck,run/'fixed.pt')
        if exposure_epochs and step==exposure_steps:
            torch.save(ck,run/'exposure.pt')
        if exposure_epochs>=2 and step==2*epoch_steps:
            torch.save(ck,run/'two_epochs.pt')
        print(json.dumps(dict(run=run.name,step=step,train_bce=record['train_probe']['bce'],val_bce=record['validation']['bce'],
                             collision_ap=record['validation']['collision_pr_auc_average_precision'],seconds=time.time()-started)),flush=True)
        save_state(step)
        monitor.check()
    if not resume:evaluate(0)
    step=start_step
    with (run/'steps.jsonl').open('a' if resume else 'w',buffering=1) as stream:
        for step in range(start_step+1,args.steps+1):
            if (out/'STOP_RESEARCH').exists():
                save_state(step-1,'stopped');return
            if offset>=len(order):
                order=generator.permutation(indices);offset=0
            ix=order[offset:offset+64];offset+=len(ix)
            presentations+=len(ix)
            model.train()
            value=cache.batch(ix,args.mode,args.device)
            label=cache.targets(ix) if hasattr(cache,'targets') else torch.tensor([cache.rows[int(i)]['sr_label'] for i in ix],dtype=torch.float32,device=args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loss=torch.nn.functional.binary_cross_entropy_with_logits(model(value),label)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            if schedule=='cosine':
                warmup=256
                fraction=step/warmup if step<=warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,args.steps-warmup)))
                for group in optimizer.param_groups:group['lr']=lr*fraction
            scaler.step(optimizer);scaler.update()
            stream.write(json.dumps(dict(step=step,loss=float(loss),norm=float(norm),lr=optimizer.param_groups[0]['lr'],batch_size=len(ix),presentations=presentations,epochs=presentations/len(indices),seconds=elapsed_before+time.time()-started))+'\n')
            if step%args.eval_every==0 or step==args.steps or step==args.fixed_steps or (exposure_epochs and step%epoch_steps==0):
                evaluate(step)
                if not exposure_epochs and step>=args.fixed_steps and stale>=args.patience:
                    break
            if step%64==0:
                monitor.check()
    save_state(step,'evaluating')
    torch.save(dict(model=model.state_dict(),config=asdict(config),step=step,protocol=protocol),run/'last.pt')
    reports={}
    for name in (('fixed','best','two_epochs','exposure') if exposure_epochs>=2 else ('fixed','best','exposure') if exposure_epochs else ('fixed','best')):
        ck=torch.load(run/(name+'.pt'),map_location='cpu')
        model.load_state_dict(ck['model'])
        reports[name]=dict(step=ck['step'],splits={})
        for split in ('train','validation','natural_reference','unseen_evaluation'):
            ix=indices if split=='train' else cache.splits[split]
            z=infer(model,cache,ix,args.mode,args.device)
            reports[name]['splits'][split]=score(cache,ix,z)
            np.savez_compressed(run/f'{name}_{split}.npz',indices=np.array(ix),logits=z)
        if name=='best' and args.mode=='full' and args.subset=='all':
            for ablation in ('query_only','history_only'):
                z=infer(model,cache,val,ablation,args.device)
                reports[name]['splits']['validation_inference_'+ablation]=score(cache,val,z)
    save(run/'results.json',dict(protocol=protocol,stopped_at=step,best_step=best_step,wall_seconds=elapsed_before+time.time()-started,
                               results=reports,production_end=production_sample(),peak_gpu_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None))
    save_state(step,'completed')


def suite(args):
    from copy import copy
    planned=[('30k','full',20261012),('50k','full',20261012),('all','full',20261012),
             ('30k','full',20261013),('all','full',20261013),
             ('all','query_only',20261012),('all','history_only',20261012)]
    subsets=read(args.output/'snapshot.json')['subsets']
    if subsets['50k']==subsets['all']:
        planned=[r for r in planned if r[0]!='50k']
    save(args.output/'suite_configuration.json',dict(runs=planned,max_steps=args.steps,fixed_steps=args.fixed_steps,
           policy='Sequential on GPU 1, 1 CPU thread, zero loader workers; every run has publication guard; already completed runs preserved'))
    for subset,mode,seed in planned:
        name=f'{subset}_{mode}_seed{seed}'
        if (args.output/'runs'/name/'results.json').exists():
            continue
        a=copy(args)
        a.subset,a.mode,a.seed=subset,mode,seed
        train(a)


class ProductionGuard:
    """Stops research only after sustained publication degradation or PID change."""
    def __init__(self,run):
        self.run=run
        self.last=0
        self.samples=[]
        self.bad=0
        self.start=production_sample()
        self.baseline=self.historical_rate()
        save(run/'production_guard.json',dict(baseline_qph=self.baseline,window_seconds=600,degradation_fraction=.25,
                                             consecutive_checks=3,check_interval=60,actions='raise and stop this research process only'))

    def historical_rate(self):
        log=PROJECT/'output/eventwm_300k_logs/progress.jsonl'
        if not log.exists():
            return None
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-8*1024*1024))
            rows=[]
            for line in f.read().splitlines():
                try:
                    r=json.loads(line)
                    if r.get('source_commit')==self.start['source_commit'] and r.get('runner',{}).get('pid')==self.start['runner_pid']:
                        rows.append(r)
                except ValueError:
                    pass
        now=self.start['timestamp']
        rates={}
        for seconds in (600,1800):
            candidates=[r for r in rows if seconds<=now-r['timestamp']<=seconds+180]
            if candidates:
                r=min(candidates,key=lambda x:abs(now-x['timestamp']-seconds))
                rates[str(seconds)]=(self.start['progress']['primary_queries']-r['new_retained'])/(now-r['timestamp'])*3600
        save(self.run/'publication_baselines.json',dict(timestamp=now,rates_qph=rates,
            policy='Use the lower contemporaneous 10/30 minute baseline; preserve both windows. Publication is bursty and scene admission varies, so this prevents attributing an existing decline to training. A later sustained additional decline still stops research.'))
        return min(rates.values()) if rates else None

    def check(self):
        now=time.time()
        if now-self.last<60:
            return
        self.last=now
        sample=production_sample()
        self.samples.append(sample)
        earlier=[s for s in self.samples if now-s['timestamp']>=600]
        rate=None
        if earlier:
            old=earlier[-1]
            rate=(sample['progress']['primary_queries']-old['progress']['primary_queries'])/(now-old['timestamp'])*3600
        self.bad=self.bad+1 if rate is not None and self.baseline and rate<.75*self.baseline else 0
        with (self.run/'production_monitor.jsonl').open('a') as f:
            f.write(json.dumps(dict(**sample,qph_10min=rate,baseline_qph=self.baseline,consecutive_degraded=self.bad))+'\n')
        if sample['runner_pid']!=self.start['runner_pid'] or not sample['runner_alive'] or self.bad>=3:
            raise RuntimeError('Production guard stopping RESEARCH: runner changed/unavailable or sustained publication degradation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('snapshot','reserve_bank','prepare','train','suite','smoke','status'))
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--subset',choices=('30k','50k','all'),default='all')
    p.add_argument('--mode',choices=('full','query_only','history_only'),default='full')
    p.add_argument('--seed',type=int,default=20261012)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--steps',type=int,default=2048)
    p.add_argument('--fixed-steps',type=int,default=1024)
    p.add_argument('--eval-every',type=int,default=256)
    p.add_argument('--patience',type=int,default=4)
    args=p.parse_args()
    args.output=args.output.resolve()
    if args.output==DATA or DATA in args.output.parents or '/production/' in str(args.output):
        raise ValueError('Protected production path')
    if args.command=='status':
        print(json.dumps(production_sample()))
    else:
        globals()[args.command](args)


if __name__=='__main__':
    main()
