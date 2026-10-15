#!/usr/bin/env python3
"""WM-only, immutable V4 census and full-model training using validated utilities.

No collector controls, Habitat simulation, planner, or privileged model inputs.
The live verified index is read in a transaction; all output belongs to research.
"""
import argparse
from collections import Counter, defaultdict
from copy import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
from types import SimpleNamespace

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import torch
import eventwm_v4_research as r
from eventwm_v4_scaling import block, fixture
from eventquerywm.inputs import INPUT_VERSION, ModelInput, collate, input_hash, tensorize
from eventquerywm.v2.labels import inspect_social_root, supervision
from collector.core import load_wm_input, digest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = r.PROJECT/'output/eventwm_full_v4_wm_20261013'
REFERENCE = r.PROJECT/'output/eventwm_v4_only_20261013'
REFERENCE_SHA = '06eb6a9fac4e8ebc684d4eb17c0d85b2cbee4d583c467a340fccdad22e32e1b2'
INDEX = r.PROJECT/'output/eventwm_v4_only_index/verified.sqlite3'
RESERVATIONS = r.PROJECT/'output/eventwm_closed_loop_improvement_20261012/manifests'
DEBUG = r.PROJECT/'output/eventwm_closed_loop_publish_checkout/configs/eventwm_closed_loop_20261012.manifest.json'
SPLITS = ('train', 'validation', 'natural_reference', 'unseen_evaluation')


def source_sha():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def branch_labels(labels):
    """A diagnostic WAIT can share query_id; only (branch_id,query_id) is unique."""
    joined = {}
    for label in labels:
        key = label['branch_id'], label['query_id']
        if key in joined:
            raise ValueError('Duplicate branch/query label')
        joined[key] = label
    return joined


def episode_signature(episode):
    path = '/'+episode['scene_id'].lower()
    family = 'MP3D' if '/mp3d/' in path else 'HM3D' if '/hm3d/' in path else 'other'
    count = episode['info']['human_num']
    points = {k:v for k,v in episode['info'].items()
              if any(k.startswith(f'human_{i}_waypoint_') for i in range(count))}
    if count and not points:
        raise ValueError('Missing physical trajectory metadata')
    body = dict(scene=family+':'+Path(episode['scene_id']).name.split('.')[0],
                human_count=count, waypoints=points)
    if not count:
        body.update(start=episode['start_position'], goals=episode['goals'])
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def reservation_groups():
    groups = set()
    for name in ('development128', 'confirmation256'):
        groups.update(row['physical_trajectory_id'] for row in
                      r.read(RESERVATIONS/name/'episodes.json')['rows'])
    for row in r.read(DEBUG)['rows']:
        with gzip.open(row['source_file'], 'rt') as f:
            episode = next(e for e in json.load(f)['episodes']
                           if str(e['episode_id']) == str(row['episode_id']))
        groups.add(episode_signature(episode))
    return groups


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), min=float(a.min()), max=float(a.max()), mean=float(a.mean()),
                quantiles={str(q):float(np.quantile(a,q)) for q in (0,.1,.25,.5,.75,.9,1)})


def freeze(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'snapshot.json').exists(), 'Never overwrite a frozen snapshot'
    assert r.sha(REFERENCE/'snapshot.json') == REFERENCE_SHA
    parent = r.read(REFERENCE/'snapshot.json')
    previous = {b['root']['root_id']:b for b in parent['blocks']}
    before = r.production_sample()
    # The index's own verifier deeply seals complete roots before publication.
    # Backup is a consistent SQLite snapshot, not a copy of raw trajectories.
    if not (out/'verified.sqlite3').exists():
        with r.ro(args.index) as source, sqlite3.connect(out/'verified.sqlite3') as target:
            source.backup(target)
    with r.ro(out/'verified.sqlite3') as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert db.execute('SELECT DISTINCT dataset FROM roots').fetchall() == [(str(r.DATA),)]
        verified = db.execute('SELECT root_id,path,role,physical_scene,queries FROM roots ORDER BY root_id').fetchall()
        provenances = {rid:(sha,complete,commit) for rid,sha,complete,commit in
                       db.execute('SELECT root_id,seal_sha256,complete_sha256,source_commit FROM provenance')}
        expected_sr = dict(db.execute('SELECT sr,COUNT(*) FROM examples GROUP BY sr'))
    with r.ro(r.DATA/'pipeline.sqlite3') as db:
        db.execute('BEGIN')
        meta = json.loads(db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()[0])
        roots = {v[0]:json.loads(db.execute('SELECT value FROM roots WHERE id=?',(v[0],)).fetchone()[0]) for v in verified}
    fixed_validation = {b['root']['root_id'] for b in parent['blocks'] if b['split']=='validation'}
    reserved = reservation_groups()
    bank = parent['audit']['reserved_selector_bank']
    assert r.sha(bank['index']) == bank['sha256']
    bank_roots = {row['root_id'] for row in r.read(bank['index'])['rows'] if row['split']=='validation'}
    blocks, excluded = [], []
    counts = defaultdict(Counter)
    durations, speeds, observed = [], [], []
    history_lengths, history_spans, history_dts = [], [], []
    diagnostic_aliases = invalid_diagnostic_waits = 0
    schema = Counter()
    started = time.time()
    guard = WMGuard(out)
    for number, v in enumerate(verified):
        guard.check()
        if (out/'STOP_RESEARCH').exists():
            raise RuntimeError('Research freeze stopped; collector untouched')
        root = roots[v[0]]
        try:
            assert root['state']=='completed' and root['path']==v[1] and root['role']==v[2]
            assert root['physical_scene_id']==v[3]
            # Retain the published path, whose archive symlink survives migration.
            # Finalization is established by immutable seals, not directory names.
            path = Path(os.path.abspath(r.DATA/root['path']))
            seal_hash, complete_hash, commit = provenances[v[0]]
            assert r.sha(r.DATA/root['seal_file']) == seal_hash
            assert r.sha(path/'COMPLETE.json') == complete_hash
            root['source_commit'] = commit
            # Reinspect every small legal/query/label record independently.
            # Canonical reader filters primary branches before any query-id join.
            rows = inspect_social_root(r.DATA,root)
            assert len(rows)==v[4]==33
            group = fixture(r.DATA,root)
            split = 'validation' if v[0] in fixed_validation else root['role']
            seals = r.read(r.DATA/root['seal_file'])
            labels = branch_labels(r.read(path/'labels/labels.json'))
            public = {q['query']['query_id']:q['query'] for q in r.read(path/'queries/queries.json')}
            primaries = {row['query_id'] for row in rows}
            diagnostic = [l for (bid,qid),l in labels.items() if not bid.startswith('b') and qid in primaries]
            diagnostic_aliases += len(diagnostic)
            invalid_diagnostic_waits += sum(not l['sr_valid'] for l in diagnostic)
            legal = load_wm_input(path,rows[0]['query_id'])
            tensor = tensorize(legal)
            times = np.array([h['timestamp'] for h in legal['history']])
            history_lengths.append(len(times));history_spans.append(times[-1]-times[0])
            history_dts.extend(np.diff(times).tolist())
            category = root['stats']['legal_state']['category']
            visible = sum(t['visible_now'] and t['valid'] for t in legal['history'][-1]['human_tracks'])
            counts['roots']['zero_visible_now'] += int(visible==0)
            counts['roots']['never_detected_humans'] += int(tensor.humans.shape[0]==0)
            counts['storage']['archived' if 'eventwm_v4_300k_archive' in str(path.resolve()) else 'sealed_persistent_root'] += 1
            for row in rows:
                label = labels[row['branch_id'],row['query_id']]
                assert label['sr_valid'] and label['sr_label']==row['sr_label']
                target = supervision(label)
                assert row['sr_label'] in (0,1) and type(label['sr_label']) is int
                human = bool(label['native_human_collision']); static = bool(label['static_collision'])
                assert (row['sr_label']==0) == (human or static)
                if row['sr_label']==1:
                    assert label['completed'] and label['full_horizon_observed']
                    assert label['termination_reason']=='query_completed'
                if human:
                    assert target['human_collision_time_valid']
                    assert 0<=target['human_collision_time_s']<=target['observed_duration_s']+1e-5
                q = public[row['query_id']]
                row.update(**group,cohort='V4',split=split,source_commit=commit,
                           category=category,human_count=root['human_count'],visible_now=visible,
                           ever_tracks=tensor.humans.shape[0],speed_mps=q['generation_config']['speed']*q['speed_scale'])
                row['root_path'] = str(path)
                counts['queries']['safe' if row['sr_label'] else 'collision'] += 1
                counts['queries']['human_collision'] += human
                counts['queries']['static_collision'] += static
                counts['queries']['noncollision_invalid_or_censored'] += int(not human and not static and not label['completed'])
                counts['categories'][category] += 1
                counts['human_counts'][str(root['human_count'])] += 1
                durations.append(q['planned_duration']);observed.append(label['observed_duration'])
                speeds.append(row['speed_mps'])
                schema[f"query={q['schema_version']};protocol={q['protocol_version']};SR={label.get('sr_contract_version')}"] += 1
            b = dict(root=root,rows=rows,split=split,cohort='V4',archive_path=str(path),
                     complete_sha256=complete_hash,seal_sha256=seal_hash,
                     evidence={n:seals[n] for n in ('VALIDATED.json','legal_observations/history.json',
                         'queries/queries.json','labels/labels.json','branch_truth/branches.jsonl.gz') if n in seals})
            blocks.append(b)
        except (OSError,ValueError,AssertionError,KeyError) as exc:
            import traceback
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            excluded.append(dict(root_id=v[0],queries=v[4],reason=repr(exc),check=frame.line))
        if (number+1)%500==0:
            print(json.dumps(dict(stage='independent_V4_audit',roots=number+1,total=len(verified),
                                  excluded=len(excluded),seconds=time.time()-started)),flush=True)
    held = {b['root']['root_id'] for b in blocks if b['split']!='train'} | bank_roots
    identifiers = defaultdict(set)
    for b in blocks:
        if b['rows'][0].get('trajectory_id') in reserved:
            held.add(b['root']['root_id'])
        for field in ('episode_id','fixture_id','trajectory_id'):
            if b['rows'][0].get(field):
                identifiers[field,b['rows'][0][field]].add(b['root']['root_id'])
    while True:
        size = len(held)
        for ids in identifiers.values():
            if ids & held: held.update(ids)
        if len(held)==size: break
    for b in blocks:
        if b['split']=='train' and b['root']['root_id'] in held:
            b['split']='quarantine_reserved_trajectory'
        for row in b['rows']: row['split']=b['split']
    leakage = {}
    for field in ('root_id','query_id','episode_id','fixture_id','trajectory_id'):
        groups = {s:{row[field] for b in blocks if b['split']==s for row in b['rows'] if row.get(field)} for s in SPLITS}
        leakage[field] = {a+'/'+b:len(groups[a]&groups[b]) for i,a in enumerate(SPLITS) for b in SPLITS[i+1:]}
        assert not any(leakage[field].values()), (field,leakage[field])
    all_queries = [row['query_id'] for b in blocks for row in b['rows']]
    assert len(all_queries)==len(set(all_queries)), 'Duplicate V4 query'
    train = [b for b in blocks if b['split']=='train']
    assert not reserved & {b['rows'][0].get('trajectory_id') for b in train}
    assert not bank_roots & {b['root']['root_id'] for b in train}
    unseen = {b['root']['physical_scene_id'] for b in blocks if b['split']=='unseen_evaluation'}
    assert not unseen & {b['root']['physical_scene_id'] for b in train}
    assert fixed_validation == {b['root']['root_id'] for b in blocks if b['split']=='validation'}
    splits = {s:dict(roots=sum(b['split']==s for b in blocks),
                    queries=sum(len(b['rows']) for b in blocks if b['split']==s),
                    scenes=len({b['root']['physical_scene_id'] for b in blocks if b['split']==s}))
              for s in sorted({b['split'] for b in blocks})}
    if not excluded:
        assert expected_sr=={0:counts['queries']['collision'],1:counts['queries']['safe']}
    audit = dict(scope='V4-only; archived, finalized, independently validated at immutable index cutoff',
        source_commit=source_sha(),cutoff=before['timestamp'],source_index=str(args.index),
        frozen_index_sha256=r.sha(out/'verified.sqlite3'),parent_snapshot_sha256=REFERENCE_SHA,
        source_finalized_queries=sum(v[4] for v in verified),source_finalized_roots=len(verified),
        live_collection_at_cutoff=before,splits=splits,counts=dict(counts),leakage=leakage,
        unique_scenes=len({b['root']['physical_scene_id'] for b in blocks}),include_v3=False,
        invalid_exclusions=excluded,invalid_excluded_queries=sum(x['queries'] for x in excluded),
        reserved_selector_bank=bank,reserved_navigation_trajectories=len(reserved),reserved_navigation_overlap=0,
        reference_validation_unchanged=True,unseen_scene_training_overlap=0,
        duration_s=distribution(durations),commanded_speed_mps=distribution(speeds),observed_duration_s=distribution(observed),
        history_steps=distribution(history_lengths),history_span_s=distribution(history_spans),history_dt_s=distribution(history_dts),
        schemas=dict(schema),diagnostic_alias_labels=diagnostic_aliases,invalid_diagnostic_alias_labels=invalid_diagnostic_waits,
        label_join='(branch_id,query_id), never query_id alone; canonical primary branch selector',
        source_epochs=dict(Counter(b['root']['source_commit'] for b in blocks)),raw_copied=False,
        static_capability='No retained static-positive supervision: do not claim learned static collision prediction',
        successor_source='Sealed branch_truth has future physical humans and legal future observations; supervision only',
        future_target_duration_coverage={str(h):sum(x>=h for x in observed) for h in (1,3,5)},
        elapsed_seconds=time.time()-started)
    ordered = sorted([b['root']['root_id'] for b in train], key=r.key)
    r.save(out/'snapshot.json',dict(version='eventwm-v4-research-snapshot-v1',blocks=blocks,audit=audit,
                                   subsets={'all':ordered,'30k':ordered[:909]}))
    identity = dict(sha256=r.sha(out/'snapshot.json'),source_index_sha256=audit['frozen_index_sha256'])
    r.save(out/'snapshot_identity.json',identity);r.save(out/'audit.json',audit)
    print(json.dumps(dict(stage='frozen',identity=identity,splits=splits,excluded=excluded)),flush=True)


def prepare(args):
    snapshot = r.read(args.output/'snapshot.json')
    assert r.sha(args.output/'snapshot.json')==r.read(args.output/'snapshot_identity.json')['sha256']
    active = {b['root']['root_id']:b for b in snapshot['blocks'] if b['split'] in SPLITS}
    old = r.read(REFERENCE/'cache/manifest.json')
    assert old['snapshot_sha256']==REFERENCE_SHA
    destination=args.output/'cache';destination.mkdir(exist_ok=True)
    assert not (destination/'manifest.json').exists(), 'Cache already frozen'
    hashes, reused, buffered, seen = {}, 0, [], set()
    guard=WMGuard(args.output)
    for name,expected in old['shards'].items():
        guard.check()
        source=REFERENCE/'cache'/name
        assert r.sha(source)==expected
        part=torch.load(source,map_location='cpu')
        kept=[]
        for b in part:
            if b['root_id'] not in active: continue
            metadata=active[b['root_id']]
            assert [x['query_id'] for x in b['rows']]==[x['query_id'] for x in metadata['rows']]
            assert [x['sr_label'] for x in b['rows']]==[x['sr_label'] for x in metadata['rows']]
            assert b['split']==metadata['split']
            kept.append(b);seen.add(b['root_id'])
        if len(kept)==len(part):
            (destination/name).symlink_to(source.resolve());hashes[name]=expected;reused+=len(part)
        elif kept:
            torch.save(kept,destination/name);hashes[name]=r.sha(destination/name)
    def flush():
        if not buffered: return
        name=f'new_roots_{len(hashes):04d}.pt'
        torch.save(buffered,destination/name);hashes[name]=r.sha(destination/name);buffered.clear()
    for rid,b in active.items():
        if rid in seen: continue
        path=Path(b['archive_path']);legal=load_wm_input(path,b['rows'][0]['query_id']);base=tensorize(legal)
        queries={q['query']['query_id']:q['query'] for q in r.read(path/'queries/queries.json')}
        query=torch.stack([tensorize(dict(history=[legal['history'][-1]],history_images=None,
                          query=queries[row['query_id']])).query for row in b['rows']])
        for j in (0,15,32):
            assert input_hash(tensorize(load_wm_input(path,b['rows'][j]['query_id'])))==input_hash(
                ModelInput(base.humans,base.human_padding,base.robot,base.history_padding,query[j]))
        buffered.append(dict(root_id=rid,split=b['split'],base=base,query=query,rows=b['rows']))
        seen.add(rid)
        if len(buffered)>=64: flush();guard.check()
    flush()
    assert seen==set(active)
    r.save(destination/'manifest.json',dict(version='eventwm-v4-compact-legal-v1',shards=hashes,
        snapshot_sha256=r.sha(args.output/'snapshot.json'),input_version=INPUT_VERSION,
        roots=len(seen),queries=sum(len(b['rows']) for b in active.values()),linked_roots=reused,
        raw_copied=False,include_v3=False))
    print(json.dumps(dict(stage='cache_frozen',roots=len(seen),linked_roots=reused,shards=len(hashes))),flush=True)


class FrozenCache(r.Cache):
    """Reuse immutable tensors; authoritative metadata normalized by this audit."""
    def __init__(self,out):
        super().__init__(out)
        snapshot=r.read(Path(out)/'snapshot.json')
        rows={x['query_id']:x for b in snapshot['blocks'] for x in b['rows']}
        for row in self.rows:
            authoritative=rows[row['query_id']]
            assert row['root_id']==authoritative['root_id'] and row['branch_id']==authoritative['branch_id']
            assert row['sr_label']==authoritative['sr_label'] and row['split']==authoritative['split']
            row.clear();row.update(authoritative)


class GPUCache:
    """Same legal tensors, root history stored once; gather batches on GPU.

    No new features or labels enter ModelInput. Additional right padding has
    identical masks; an explicit parity check precedes timing/training.
    """
    def __init__(self,cache,device):
        self.original=cache
        self.rows=cache.rows;self.splits=cache.splits;self.blocks=cache.blocks
        self.samples=cache.samples;self.byroot=cache.byroot
        self.device=device
        roots=collate([b['base'] for b in cache.blocks])
        self.root_shapes=dict(humans=list(roots.humans.shape),robot=list(roots.robot.shape))
        # Each root base's query is irrelevant; candidate queries remain unique.
        self.root_inputs=roots.to(device)
        self.queries=torch.cat([b['query'] for b in cache.blocks]).to(device)
        self.root_index=torch.tensor([j for j,b in enumerate(cache.blocks) for _ in b['rows']],device=device)
        self.labels=torch.tensor([row['sr_label'] for row in cache.rows],dtype=torch.float32,device=device)
        self.lengths=np.asarray([b['base'].robot.shape[0] for b,j in cache.samples])
        self.human_counts=np.asarray([b['base'].humans.shape[0] for b,j in cache.samples])

    def batch(self,indices,mode='full',device=None):
        if str(device)!=str(self.device): raise ValueError('GPU cache device mismatch')
        ix=torch.as_tensor(indices,dtype=torch.long,device=self.device)
        value=self.root_inputs.select(self.root_index[ix])
        # Match legacy per-batch padding without a GPU-to-CPU synchronization.
        t=int(self.lengths[indices].max());n=max(1,int(self.human_counts[indices].max()))
        value.humans=value.humans[:,:n,:t];value.human_padding=value.human_padding[:,:n]
        value.robot=value.robot[:,:t];value.history_padding=value.history_padding[:,:t]
        value.query=self.queries[ix]
        if mode=='query_only': value.human_padding=torch.ones_like(value.human_padding)
        if mode=='history_only': value.query=torch.zeros_like(value.query)
        return value

    def targets(self,indices):
        return self.labels[torch.as_tensor(indices,dtype=torch.long,device=self.device)]


def benchmark(args):
    """Bounded GPU-only fit smoke; no trained checkpoint is selected here."""
    from eventquerywm.model import EventQueryWM,ModelConfig
    assert torch.cuda.is_available(), 'CUDA failed: never fall back to CPU training'
    assert args.device.startswith('cuda'), 'Substantial training must use CUDA'
    source=args.cache_source or args.output
    torch.cuda.set_device(args.device)
    cache=FrozenCache(source)
    r.seed_all(20261013)
    model=EventQueryWM(ModelConfig.family(args.size,dropout=.2)).to(args.device)
    indices=np.random.default_rng(20261013).permutation(cache.splits['train']).tolist()
    batches=[indices[i:i+64] for i in range(0,64*100,64)]
    gpu=GPUCache(cache,args.device)
    model.eval()
    with torch.no_grad():
        legacy=model(cache.batch(batches[0],'full',args.device))
        resident=model(gpu.batch(batches[0],'full',args.device))
        parity=float((legacy-resident).abs().max())
    assert parity<1e-4, ('GPU-cache padding parity failed',parity)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    results=[]
    for loader,amp,foreach in ((cache,False,False),(gpu,False,False),(gpu,False,True),(gpu,True,True)):
        model.load_state_dict(initial);model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,foreach=foreach)
        scaler=torch.cuda.amp.GradScaler(enabled=amp)
        loading=0.;cpu_start=time.process_time();torch.cuda.reset_peak_memory_stats(args.device)
        start=time.perf_counter()
        for step,ix in enumerate(batches):
            before=time.perf_counter();value=loader.batch(ix,'full',args.device)
            label=loader.targets(ix) if hasattr(loader,'targets') else torch.tensor(
                [cache.rows[i]['sr_label'] for i in ix],dtype=torch.float32,device=args.device)
            loading+=time.perf_counter()-before
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits=model(value)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,label)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update()
        torch.cuda.synchronize(args.device)
        elapsed=time.perf_counter()-start
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        results.append(dict(loader='gpu_resident' if loader is gpu else 'legacy_cpu_collate',mixed_precision=amp,foreach_optimizer=foreach,
            steps=len(batches),seconds=elapsed,steps_per_second=len(batches)/elapsed,
            cpu_seconds=time.process_time()-cpu_start,cpu_one_core_fraction=(time.process_time()-cpu_start)/elapsed,
            loading_enqueue_seconds=loading,loading_enqueue_fraction=loading/elapsed,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(args.device),last_loss=float(loss)))
    # Select purely on measured throughput and stability; no validation tuning.
    winner=max(results,key=lambda x:x['steps_per_second'])
    report=dict(source_commit=source_sha(),snapshot_sha256=r.sha(source/'snapshot.json'),
        device=args.device,cuda_available=torch.cuda.is_available(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_name=torch.cuda.get_device_name(args.device),gpu_free_total_bytes=torch.cuda.mem_get_info(args.device),
        cache_parity_max_logit_difference=parity,root_shapes=gpu.root_shapes,measurements=results,
        recommendation=winner,
        gpu_utilization='NVML unavailable due existing mismatch; CUDA timings and memory measured directly',
        production=r.production_sample())
    report['size']=args.size
    args.output.mkdir(parents=True,exist_ok=True);r.save(args.output/('gpu_throughput.json' if args.size=='M' else f'gpu_throughput_{args.size}.json'),report)
    print(json.dumps(report,indent=2),flush=True)


class WMGuard(r.ProductionGuard):
    def historical_rate(self):
        rate=super().historical_rate()
        if rate: return rate
        heartbeat=Path('/home/haodong/workspace/datasets/eventwm-falcon/state/v4_continuous/supervisor.json')
        if heartbeat.exists():
            s=r.read(heartbeat)
            if s.get('collector_pid')==self.start['runner_pid'] and time.time()-s.get('timestamp',0)<120:
                return s.get('throughput',{}).get('queries_per_hour')
        return None


def suite(args):
    assert torch.cuda.is_available(), 'No system repairs; CUDA must work before training'
    assert args.device.startswith('cuda'), 'Substantial training must use CUDA'
    index=torch.device(args.device).index or 0
    if index>=torch.cuda.device_count():
        raise RuntimeError(f'Requested {args.device} but only {torch.cuda.device_count()} devices visible; CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}')
    assert torch.tensor([1.,2.],device=args.device).sum().item()==3.
    r.save(args.output/f'cuda_preflight_seed{args.seed_filter}.json',dict(
        source_commit=source_sha(),device=args.device,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        device_count=torch.cuda.device_count(),gpu_name=torch.cuda.get_device_name(args.device),
        free_total_bytes=torch.cuda.mem_get_info(args.device),actual_tensor_operation_passed=True))
    r.Cache=FrozenCache
    r.ProductionGuard=WMGuard
    cache=FrozenCache(args.output)
    throughput=r.read(args.throughput_profile or args.output/'gpu_throughput.json')
    assert throughput['snapshot_sha256']==r.sha(args.output/'snapshot.json')
    chosen=throughput['recommendation']
    if chosen['loader']=='gpu_resident':cache=GPUCache(cache,args.device)
    plans=[dict(name=f'constant4_seed{s}',size='M',seed=s,epochs=4,lr=1e-4,schedule='constant')
           for s in (20261012,20261013)]
    plans += [dict(name=f'cosine4_seed{s}',size='M',seed=s,epochs=4,lr=1e-4,schedule='cosine')
              for s in (20261012,20261013)]
    plans += [dict(name=f'L_cosine4_seed{s}',size='L',seed=s,epochs=4,lr=1e-4,schedule='cosine')
              for s in (20261012,20261013)]
    r.save(args.output/'experiment_matrix.json',dict(source_commit=source_sha(),plans=plans,
        architecture='Existing EventQueryWM families M and L; every parameter trainable',
        inference_only_ablations='Supporting diagnostic, not retrained architecture comparisons',
        selection='Minimum validation BCE, fixed validation membership; do not choose on natural/unseen'))
    for plan in plans:
        if args.seed_filter is not None and plan['seed']!=args.seed_filter:continue
        if (args.output/'STOP_RESEARCH').exists(): break
        if (args.output/'runs'/plan['name']/'results.json').exists(): continue
        a=copy(args)
        for k,v in plan.items(): setattr(a,k,v)
        a.subset='all';a.mode='full';a.cache=cache;a.exposure_epochs=a.epochs
        a.fixed_steps=1024;a.steps=2048;a.eval_every=512;a.patience=4
        a.source_commit=source_sha();a.weight_decay=.01;a.dropout=.2;a.resume=True
        a.amp=chosen['mixed_precision'];a.loader=chosen['loader']
        a.foreach=chosen.get('foreach_optimizer',False)
        r.train(a)


def control(args):
    """Bounded current-architecture diagnostic; no new predictive module."""
    assert torch.cuda.is_available() and args.device.startswith('cuda')
    r.Cache=FrozenCache;r.ProductionGuard=WMGuard
    cache=FrozenCache(args.output)
    profile=r.read(args.throughput_profile or args.output/'benchmark_v2/gpu_throughput.json')
    assert profile['snapshot_sha256']==r.sha(args.output/'snapshot.json')
    choice=profile['recommendation']
    if choice['loader']=='gpu_resident':cache=GPUCache(cache,args.device)
    a=copy(args);a.name=f'{args.recipe}_seed{args.seed_filter or 20261012}'
    if (args.output/'runs'/a.name/'results.json').exists():
        print('Completed control preserved: '+a.name);return
    a.seed=args.seed_filter or 20261012;a.size='M';a.subset='all'
    a.mode='query_only' if args.recipe=='query_only' else 'full'
    a.dropout=.35 if args.recipe=='regularization' else .2
    a.weight_decay=.1 if args.recipe=='regularization' else .01
    a.lr=1e-4;a.schedule='cosine';a.exposure_epochs=4;a.fixed_steps=1024
    a.steps=15112;a.eval_every=512;a.patience=4;a.cache=cache;a.resume=True
    a.source_commit=source_sha();a.amp=choice['mixed_precision'];a.loader=choice['loader']
    a.foreach=choice.get('foreach_optimizer',False)
    r.train(a)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('freeze','prepare','benchmark','suite','control','status','stop','resume'))
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--index',type=Path,default=INDEX)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--cache-source',type=Path)
    p.add_argument('--size',choices=('M','L'),default='M')
    p.add_argument('--throughput-profile',type=Path)
    p.add_argument('--seed-filter',type=int,choices=(20261012,20261013))
    p.add_argument('--recipe',choices=('regularization','query_only'),default='regularization')
    args=p.parse_args();args.output=args.output.resolve()
    if args.output==r.DATA or r.DATA in args.output.parents or '/production/' in str(args.output):
        raise ValueError('Protected production output')
    torch.set_num_threads(1)
    if args.command=='stop':
        (args.output/'STOP_RESEARCH').write_text('Stop this WM research only at the next checkpoint boundary.\n')
    elif args.command=='status':
        print(json.dumps(dict(stopped=(args.output/'STOP_RESEARCH').exists(),
            runs=[dict(name=q.parent.name,summary=r.read(q)) for q in (args.output/'runs').glob('*/progress.json')])))
    else:
        if args.command=='resume':
            (args.output/'STOP_RESEARCH').unlink(missing_ok=True);args.command='suite'
        globals()[args.command](args)


if __name__=='__main__': main()
