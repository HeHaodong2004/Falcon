#!/usr/bin/env python3
"""One-root correctness collector. No multi-scene/scale mode by design."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))
from collector.core import Query, build_labels, digest, read, write


def launch(args, worker_id, query_id=None, limit=None):
    dest = args.output/'audits/workers'/worker_id
    dest.mkdir(parents=True,exist_ok=True)
    cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'worker','--output',str(args.output),
           '--worker-id',worker_id,'--storage-profile',getattr(args,'storage_profile','validated')]
    if query_id:
        cmd += ['--query-id',query_id]
    if limit is not None:
        cmd += ['--limit-steps',str(limit)]
    start = time.time()
    with (dest/'run.log').open('w') as log:
        process = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=REPO,timeout=180)
    write(dest/'execution.json',dict(command=cmd,returncode=process.returncode,seconds=time.time()-start))
    if process.returncode:
        write(dest/'collection_error.json',dict(termination_reason='collection_error',returncode=process.returncode))
        raise RuntimeError(f'Worker {worker_id} failed; see {dest}/run.log')
    print('COMPLETE',worker_id,round(time.time()-start,2),'s',flush=True)
    if query_id:
        branch=read(dest/'branch.json')
        from collector.v2 import PROTOCOL, human_event_labels
        if read(args.output/'roots/replay.json').get('protocol_version') == PROTOCOL:
            q=next(x['query'] for x in read(args.output/'queries/queries.json') if x['query']['query_id']==query_id)
            label=human_event_labels(branch,q)
            if label['static_feasibility_violation']:
                write(args.output/'audits/STATIC_FEASIBILITY_VIOLATION.json',label)
                write(args.output/'labels/quarantined_branch.json',label)
                raise RuntimeError('STATIC_FEASIBILITY_VIOLATION: stopping all collection')
        return branch


def collect(args):
    from collector.v2 import guard_output, PROTOCOL, QueryConfig
    guard_output(args.output)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new output directory; refusing to overwrite a collection')
    for d in ('roots','legal_observations','queries','branch_truth','labels','audits','diagnostics'):
        (args.output/d).mkdir(parents=True,exist_ok=True)
    recipe = read(args.recipe) if getattr(args, 'recipe', None) else dict(case='002_castle_h1',seed=2301,
          mechanism='fresh_process_seeded_episode_plus_exact_prefix',prefix_controls=[[0.,0.]]*40)
    if getattr(args,'protocol','v2') == 'v2':
        recipe.update(protocol_version=PROTOCOL)
        recipe.setdefault('query_config',dict(movement_queries=getattr(args,'movement_queries',8)))
        recipe.setdefault('static_config',{})
    write(args.output/'roots/replay.json',recipe)
    sources = [Path(__file__),*sorted((REPO/'scripts/collector').glob('*.py')),
               *sorted((REPO/'scripts/controlled').glob('*.py'))]
    write(args.output/'audits/provenance.json',dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          storage_profile=getattr(args,'storage_profile','validated'),
          python=sys.executable,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
          source_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    launch(args,'root')
    queries = read(args.output/'queries/queries.json')
    if len(queries)+3 > getattr(args, 'max_branches', 36):
        raise ValueError('Root branch budget exceeded before branch execution')
    records = []
    for i,row in enumerate(queries):
        records.append(launch(args,f'b{i:03}',row['query']['query_id']))
    # Reverse a pair after all other branches; each query is replayed in a new
    # process, proving the orchestrator never carries live human state forward.
    for i in (1,0):
        records.append(launch(args,f'reverse_{i}',queries[i]['query']['query_id']))
    records.append(launch(args,'truncated_wait',queries[-1]['query']['query_id'],limit=10))
    with (args.output/'branch_truth/branches.jsonl').open('w') as f:
        import json
        for row in records:
            f.write(json.dumps(row,allow_nan=False)+'\n')
    # Images grouped into one archive per root; no per-timestep/query NPZ files in
    # the logical dataset. Worker files are retained as execution audit artifacts.
    import zipfile
    if getattr(args,'storage_profile','validated')!='production':
        with zipfile.ZipFile(args.output/'legal_observations/branch_images.zip','w') as archive:
            for b in records:
                archive.write(args.output/'audits/workers'/b['branch_id']/'branch_images.npz',b['branch_id']+'.npz')
    labels(args)
    validate(args)
    if getattr(args,'storage_profile','validated')=='production':
        from collector.storage import compact_workers
        compact_workers(args.output)
    elif not getattr(args, 'no_render', False):
        render(args)


def branches(output):
    import json
    return [json.loads(line) for line in (output/'branch_truth/branches.jsonl').read_text().splitlines()]


def labels(args):
    from collector.v2 import PROTOCOL, human_event_labels
    recipe=read(args.output/'roots/replay.json')
    if recipe.get('protocol_version') == PROTOCOL:
        qs={q['query']['query_id']:q['query'] for q in read(args.output/'queries/queries.json')}
        case=next(c for c in read(REPO/'evaluation/controlled50/cases.json') if c['tag']==recipe['case'])
        labs=[human_event_labels(b,qs[b['query_id']],case['humans']) for b in branches(args.output)]
    else: labs=[build_labels(b) for b in branches(args.output)]
    write(args.output/'labels/labels.json',labs)


def validate(args):
    import numpy as np
    from collector.core import LegalRoot, legal_observation, load_wm_input
    rows = branches(args.output)
    qs = read(args.output/'queries/queries.json')
    byid = {q['query']['query_id']:q for q in qs}
    from collector.v2 import PROTOCOL, human_event_labels
    v2=read(args.output/'roots/replay.json').get('protocol_version') == PROTOCOL
    audits = {}
    def audit(name,passed,details):
        audits[name] = dict(status='PASS' if passed else 'FAIL',details=details)
    audit('query_immutability',all(b['query_hash_before']==b['query_hash_after']==byid[b['query_id']]['sha256']==
          digest(byid[b['query_id']]['query']) for b in rows),'Canonical SHA256 before/after every branch and persisted query')
    audit('same_root',len({b['root_fingerprint'] for b in rows})==1,'Exact root physical/controller/RNG/history/image fingerprint in every fresh process')
    pairs = [(next(b for b in rows if b['branch_id']==f'b{i:03}'),
              next(b for b in rows if b['branch_id']==f'reverse_{i}')) for i in (0,1)]
    reproducible = all(digest(a['truth'])==digest(b['truth']) and digest(a['controls'])==digest(b['controls']) and
                       digest(a['legal_observations'])==digest(b['legal_observations']) for a,b in pairs)
    audit('same_query_reproducibility',reproducible,'Two queries replayed exactly, including all native truth, controls and legal metadata')
    audit('branch_order_independence',reproducible,'Original order 0,1,...; repeated order 1,0 in independent fresh processes')
    # Adversarial poison of all extra root fields is stripped at the boundary.
    clean = read(args.output/'roots/legal_root.json')
    dirty = dict(clean,hidden_human_position=[999,999,999],human_future=[-999],human_goal='poison')
    audit('generator_privileged_boundary',LegalRoot.load(clean)==LegalRoot.load(dirty),
          'Typed whitelist excludes adversarial GT/future/goal fields; runtime generation additionally uses a separate static PathFinder')
    fake = {'agent_0_articulated_agent_jaw_rgb':np.zeros((2,2,3)),
            'agent_0_articulated_agent_jaw_depth':np.ones((2,2,1))}
    a = legal_observation(fake,[0,0,0],0,0,[0,0])
    fake.update(hidden_human_position=np.ones(3)*999,oracle_humanoid_future_trajectory=np.ones((10,3)))
    b = legal_observation(fake,[0,0,0],0,0,[0,0])
    wm = load_wm_input(args.output,qs[0]['query']['query_id'])
    from collector.observation import TRACK_FIELDS
    audit('hidden_gt_not_in_legal_input',a[0]==b[0] and all(np.array_equal(a[1][k],b[1][k]) for k in a[1]) and
          all(set(t)==set(TRACK_FIELDS) for h in wm['history'] for t in h['human_tracks']),
          'Injected raw hidden/future sensors excluded; structured tracks have an explicit nested whitelist; visibility and tracking audited separately')
    case=next(c for c in read(REPO/'evaluation/controlled50/cases.json') if c['tag']==read(args.output/'roots/replay.json')['case'])
    derived = [human_event_labels(b,byid[b['query_id']]['query'],case['humans']) if v2 else build_labels(b) for b in rows]
    audit('offline_label_recomputation',derived==read(args.output/'labels/labels.json'),'All labels recomputed solely from saved raw truth')
    truncated = next(x for x in derived if x['branch_id']=='truncated_wait')
    audit('truncation_censoring',truncated['censored'] and truncated['full_query_safe'] is None,
          {'truncated_wait':truncated})
    temporal = []
    for b in rows:
        q = byid[b['query_id']]['query']; t0 = b['truth'][0]['timestamp']
        ts = np.array([r['timestamp'] for r in b['truth']])
        temporal.append(bool(np.all(np.diff(ts)>0) and np.max(np.diff(ts))<=.02400001 and
            all(abs(c['t_start']-(t0+c['index']*q['control_dt']))<q['clock_quantization_tolerance'] and
                abs(c['t_end']-c['t_start']-q['control_dt'])<q['clock_quantization_tolerance'] and
                c['issued_command']==q['executable_controls'][c['index']] for c in b['controls']) and
            (not b['completed'] or abs(b['observed_duration']-q['planned_duration'])<q['clock_quantization_tolerance'])))
    audit('timestamps',all(temporal),'Nominal dt = task physics request + four native substeps; <=8 ms clock quantization; recording gaps <=24 ms; control integration 1/120 s')
    # Re-run the exact generator without importing a live simulator environment.
    from collector.runtime import StaticPlanner
    from collector.core import generate_queries
    scenes = read(REPO/'evaluation/controlled50/scenes.json')
    recipe = read(args.output/'roots/replay.json')
    case = next(c for c in read(REPO/'evaluation/controlled50/cases.json') if c['tag']==recipe['case'])
    navmesh = REPO/'evaluation/controlled50'/next(s['navmesh'] for s in scenes if s['name']==case['scene'])
    if not v2:
        regen,_ = generate_queries(LegalRoot.load(dirty),StaticPlanner(navmesh))
        audit('static_planner_human_blind',[q.sha256 for q in regen]==[q['sha256'] for q in qs],
              'Regenerated with poisoned privileged fields and standalone static navmesh')
    else:
        audit('static_feasibility_certificates',all(q['query']['static_precheck']['static_feasible'] for q in qs),
              'Every immutable fixed variant checked before branch execution')
        audit('zero_unexpected_static_collisions',not any(l['static_feasibility_violation'] for l in derived),
              'Static contact invalidates any human outcome and stops collection')
        audit('zero_human_invariant',case['humans']!=0 or not any(l['human_event_label']==1 for l in derived),
              'Zero humans cannot have a positive human event')
        primary=[l for l in derived if l['branch_id'].startswith('b')]
        audit('v2_primary_labels',all(l['human_event_valid'] for l in primary),'All primary outcomes determined under human-event contract')
    ref = next(b for b in rows if b['branch_id']==f'b{len(qs)-1:03}')
    comparisons = []
    for b in rows[:len(qs)-1] if case['humans'] else []:
        n = min(len(b['truth']),len(ref['truth']))
        hp = np.array([r['humans'][0]['position'] for r in b['truth'][:n]])
        hr = np.array([r['humans'][0]['position'] for r in ref['truth'][:n]])
        dt = np.diff([r['timestamp'] for r in b['truth'][:n]])
        speed_delta = np.abs(np.linalg.norm(np.diff(hp,axis=0),axis=1)/dt-np.linalg.norm(np.diff(hr,axis=0),axis=1)/dt)
        comparisons.append(dict(branch_id=b['branch_id'],query_id=b['query_id'],reference_branch=ref['branch_id'],
            compared_seconds=b['truth'][n-1]['timestamp']-b['truth'][0]['timestamp'],
            max_human_deviation_m=float(np.linalg.norm(hp-hr,axis=1).max()),
            mean_absolute_speed_change_mps=float(speed_delta.mean()),
            human_travel_difference_m=float(np.linalg.norm(np.diff(hp,axis=0),axis=1).sum()-np.linalg.norm(np.diff(hr,axis=0),axis=1).sum()),
            minimum_clearance=build_labels(b)['minimum_human_clearance']))
    reacting = max((c['max_human_deviation_m'] for c in comparisons), default=0.)>1e-5
    audit('human_reactivity',reacting,dict(max_deviation_m=max((c['max_human_deviation_m'] for c in comparisons), default=0.),
          minimum_active_clearance=min((c['minimum_clearance'] for c in comparisons), default=None),
          interpretation='Observed robot-dependent human motion' if reacting else
          'No robot reactivity observed. Native avoidance excludes agent 0; semantics unchanged.'))
    write(args.output/'diagnostics/comparisons.json',comparisons)
    write(args.output/'audits/audits.json',audits)
    print('AUDITS', {k:v['status'] for k,v in audits.items()},flush=True)
    # Reactivity is a scientific result, not a collector-integrity assertion.
    if any(v['status']=='FAIL' for k,v in audits.items() if k!='human_reactivity'):
        raise RuntimeError('Collector integrity audit failed; see audits/audits.json')


def render(args):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    rows = branches(args.output); qs = read(args.output/'queries/queries.json')
    labels_byid = {l['branch_id']:l for l in read(args.output/'labels/labels.json')}
    # Pick closest approach spatial path offline, never use it to alter queries.
    selected = min(rows[:len(qs)-1],key=lambda b:labels_byid[b['branch_id']]['minimum_human_clearance'])
    direction = next(q['query']['direction'] for q in qs if q['query']['query_id']==selected['query_id'])
    ids = {q['query']['query_id'] for q in qs if q['query']['direction'] in (direction,'WAIT')}
    group = [b for b in rows[:len(qs)] if b['query_id'] in ids]
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for b in group:
        q = next(q['query'] for q in qs if q['query']['query_id']==b['query_id'])
        label = f"{b['branch_id']} wait={q['wait_duration']:.2f} speed={q['speed_scale']}"
        p=np.array([r['robot']['position'] for r in b['truth']]); t=np.array([r['timestamp'] for r in b['truth']]); h=np.array([r['humans'][0]['position'] for r in b['truth']])
        axes[0].plot(p[:,0],p[:,2],label=label)
        axes[1].plot(t-t[0],np.linalg.norm((p-h)[:,[0,2]],axis=1)-.55,label=label)
    axes[0].plot(h[:,0],h[:,2],'k--',label='reference human')
    axes[0].set(xlabel='world x (m)',ylabel='world z (m)',title='Same root: nominal timing variants / WAIT'); axes[0].axis('equal')
    axes[1].axhline(0,color='red'); axes[1].axhline(.5,color='gray',linestyle=':')
    axes[1].set(xlabel='actual simulator seconds after root',ylabel='disc clearance (m)',title='Actual consequences')
    for ax in axes: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.output/'diagnostics/comparison.png'); plt.close(fig)
    video_ids = [group[0]['branch_id'],group[1]['branch_id'],group[3]['branch_id'],group[-1]['branch_id']]
    stacks = [np.load(args.output/'audits/workers'/bid/'branch_images.npz')['rgb'] for bid in video_ids]
    import cv2
    with imageio.get_writer(args.output/'diagnostics/comparison.mp4',fps=1/qs[0]['query']['control_dt'],codec='libx264') as writer:
        for i in range(max(map(len,stacks))):
            panels=[]
            for bid,stack in zip(video_ids,stacks):
                frame=stack[min(i,len(stack)-1)].copy()
                cv2.putText(frame,bid+(' ended' if i>=len(stack) else ''),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,0),1)
                panels.append(frame[:,:,:3])
            writer.append_data(np.concatenate(panels,axis=1))
    write(args.output/'diagnostics/render.json',dict(branches=video_ids,selection='offline closest-approach path; now / wait .5 / slow / reference WAIT',ended_frames='held and marked ended'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['collect','worker','labels','validate','render'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--storage-profile',choices=['validated','production'],default='validated')
    p.add_argument('--recipe',type=Path)
    p.add_argument('--max-branches',type=int,default=36)
    p.add_argument('--no-render',action='store_true')
    p.add_argument('--protocol',choices=['v1','v2'],default='v2')
    p.add_argument('--movement-queries',type=int,default=8)
    p.add_argument('--worker-id',default='root'); p.add_argument('--query-id'); p.add_argument('--limit-steps',type=int)
    args=p.parse_args(); args.output=args.output.absolute()
    from collector.v2 import guard_output
    guard_output(args.output)
    if args.command=='worker':
        from collector.runtime import worker
        worker(args)
    else:
        globals()[args.command](args)


if __name__=='__main__':
    main()
