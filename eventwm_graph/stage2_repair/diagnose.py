"""Read-only source/data inventory and independent real-window checks.

Run through scripts/stage2_segment_run.py; never imports the simulator.
Artifacts (including source observations and weights) stay under output/.
"""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from eventwm_graph.stage2_hf.common import OUT as HF, CKPT, PREP, read, save, sha
from eventwm_graph.stage2_hf.packed import load_arrays, get_segment
from eventwm_graph.stage2.data import source_rows
from eventwm_graph.stage2_segment.data import load, replay, read as member

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/stage2_repair_20261025'


def inventory():
    train, tm = source_rows('train')
    dev, dm = source_rows('dev')
    p = read(HF / 'protocol.json')
    assert not ({r['scene_id'] for r in train} & {r['scene_id'] for r in dev})
    assert not (set(p['opened_protected_roots']) & {r['root_id'] for r in train + dev})
    records = {}
    for name, path in [('train', tm), ('dev', dm), ('protocol', HF/'protocol.json')]:
        records[name] = dict(path=str(path), sha256=sha(path))
    records['counts'] = dict(train_roots=len(train), dev_roots=len(dev),
        training_scenes=len({r['scene_id'] for r in train}),
        development_scenes=len({r['scene_id'] for r in dev}),
        opened_confirmation_roots=len(p['opened_protected_roots']))
    for split in ['pilot_fit', 'dev_pilot', 'full']:
        for kind in ['packed', 'temporal', 'time_prepared']:
            path = HF/kind/split/'manifest.json'
            if path.exists():
                m = read(path)
                records[kind+'/'+split] = dict(path=str(path.resolve()), sha256=sha(path),
                    **{k:v for k,v in m.items() if k in ['roots','scenes','segments','sampled_cells','counts','failures']})
    records['baseline'] = dict(path=str(CKPT.resolve()), sha256=sha(CKPT))
    for name in ['time_twohead', 'time_zero_control', 'spatial_bce']:
        path = HF/'training'/name/'epoch02.pt'
        if path.exists():
            records[name] = dict(path=str(path.resolve()), sha256=sha(path), config=read(path.parent/'config.json'))
    records['load'] = subprocess.check_output(['ps','-eo','pid,ppid,etime,pcpu,pmem,args','--sort=-pcpu'],text=True).splitlines()[:27]
    records['gpu'] = dict(torch=torch.__version__, available=torch.cuda.is_available(),
        devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    records['train_example_row'] = train[0]
    records['dev_example_row'] = dev[0]
    records['source_manifest_keys'] = list(read(tm))
    save(OUT/'inventory.json', records)
    print(records['counts'], records['gpu'], flush=True)
    print('training row', train[0], flush=True)


def reference_update(state, observation):
    """Dictionary implementation independent of PartialGraph and FastPartialGraph."""
    cells = state['cells']
    blocked = {tuple(u['cell']) for u in observation['static_updates'] if u['state']==2}
    for u in observation['static_updates']:
        key = tuple(u['cell'])
        if u['source']=='occupied_ego_base_support' or (u['state']==1 and key in blocked):
            continue
        first = cells[key]['first_seen'] if key in cells else float(observation['timestamp'])
        cells[key] = dict(state=int(u['state']),height=float(u['height']),first_seen=first,
            last_seen=float(observation['timestamp']),source=u['source'])
    pose = observation['robot_pose']
    if not state['visited'] or np.linalg.norm((np.array(pose[:3])-state['visited'][-1]['position'])[[0,2]])>=.55:
        state['visited'].append(dict(position=list(pose[:3]),timestamp=float(observation['timestamp'])))
    state['timestamp'] = float(observation['timestamp'])
    state['ego_support'] = list(pose[:3])


def reference_state(legal, nodes, sid):
    root = legal['partial_graph']
    state = dict(cells={tuple(c['cell']):{k:v for k,v in c.items() if k!='cell'} for c in root['cells']},
        visited=copy.deepcopy(root['visited']), timestamp=root['timestamp'],ego_support=root['ego_support'])
    ancestors = []
    while sid is not None:
        ancestors.append(sid)
        sid = nodes[sid]['parent']
    for sid in reversed(ancestors):
        reference_update(state,nodes[sid]['observation'])
    return state


def trace():
    """24 training windows, independent restart-per-window ancestry, exact keys."""
    from eventwm_graph.stage2_segment.features import SegmentFeatures
    from eventwm_graph.stage2_segment.fused import transform
    from eventwm_graph.stage2_segment.api import SegmentPredictor
    from eventwm_graph.stage2_hf.spatial import build
    train, _ = source_rows('train')
    index = {r['root_id']:r for r in train}
    packed = read(HF/'packed/pilot_fit/manifest.json')
    selected = []
    # Pick by real movement and target composition, before inspecting predictions.
    counts = Counter()
    for row in packed['rows']:
        a = load_arrays(row)
        for i, meta in enumerate(row['segments']):
            v = get_segment(a,i); p=v[3]
            movement=float(np.linalg.norm(np.diff(p[:,[1,3]],axis=0),axis=1).sum())
            dy=np.diff(p[:,4]);rotation=float(np.abs(np.arctan2(np.sin(dy),np.cos(dy))).sum())
            motion='mixed' if movement>1e-4 and rotation>1e-4 else 'translation' if movement>1e-4 else 'rotation' if rotation>1e-4 else 'stationary'
            kind='zero' if not len(v[4]) else 'blocked' if np.any(v[5]==2) else 'free'
            key=motion+':'+kind
            if counts[key]>=2 or (len(selected)>=24):continue
            selected.append((row,i,motion,kind));counts[key]+=1
        if len(selected)>=24:break
    # Preserve selected example identities; no development labels are fitting inputs.
    records=[];predictor=SegmentPredictor(CKPT,device='cuda:0',execution='optimized',future_format='native',feature_threads=1)
    source_cache={};replays={};quantized=[]
    for row,i,motion,kind in selected:
        rid=row['root_id'];a=load_arrays(row);v=get_segment(a,i);meta=row['segments'][i]
        if rid not in source_cache:
            source_cache[rid]=load(index[rid]);replays[rid]={(e['start_node'],e['end_node']):e for e in replay(index[rid]) if 'audit' not in e}
        legal,queries,outcomes,nodes,children=source_cache[rid]
        ex=replays[rid][(meta['start_node'],meta['end_node'])]
        before=reference_state(legal,nodes,meta['start_node']);after=reference_state(legal,nodes,meta['end_node'])
        actual={tuple(c['cell']):{k:v for k,v in c.items() if k!='cell'} for c in ex['graph']['cells']}
        assert before['cells']==actual
        for k in ['timestamp','visited','ego_support']:assert before[k]==ex['graph'][k],k
        target={k:(c['state'],c['height']) for k,c in after['cells'].items() if k not in before['cells']}
        assert target==ex['target']
        assert ex['known_terminal_differences']==sum(c['state']!=after['cells'][k]['state'] or abs(c['height']-after['cells'][k]['height'])>1e-8 for k,c in before['cells'].items())
        np.testing.assert_array_equal(v[0],np.array([c['cell'] for c in ex['graph']['cells']]))
        np.testing.assert_array_equal(v[1],[c['state'] for c in ex['graph']['cells']])
        np.testing.assert_allclose(v[2],[c['height'] for c in ex['graph']['cells']],rtol=0,atol=2e-6)
        np.testing.assert_array_equal(v[3],ex['trajectory'])
        assert {tuple(k):int(s) for k,s in zip(v[4],v[5])}=={k:v[0] for k,v in target.items()}
        start_ob=legal['history'][-1] if meta['start_node'] is None else nodes[meta['start_node']]['observation']
        np.testing.assert_array_equal(v[3][0],[start_ob['timestamp'],*start_ob['robot_pose']])
        ctx=SegmentFeatures(ex['graph'],ex['trajectory']);keys=ctx.candidates('both')
        dense=build(*v[:4],v[-1],*v[4:7],make_query=False,domain_check=False)
        np.testing.assert_array_equal(keys,dense['keys'])
        y=np.array([target.get(tuple(k),(0,0))[0] for k in keys],np.uint8)
        ci=keys-dense['lo'];np.testing.assert_array_equal(y,dense['labels'][ci[:,0],ci[:,1]])
        # Match CPU features against CUDA, including cell identity (not labels alone).
        check=keys[np.linspace(0,len(keys)-1,min(256,len(keys))).astype(int)]
        cpu,h=ctx.transform(check,'both');gpu,gh=transform(ctx,check,cuda=predictor.cuda,device_output=True);gpu=gpu.cpu().numpy()
        delta=predictor.predict(partial_graph=ex['graph'],trajectory=ex['trajectory'])
        np.testing.assert_array_equal(keys,delta['candidate_cells'])
        p=delta['probabilities'];hard=np.where((p[:,1]+p[:,2]).astype(np.float64)>=.2,np.where(p[:,1]>=p[:,2],1,2),0).astype(np.uint8)
        cm=np.bincount(3*y.astype(np.int64)+hard,minlength=9).reshape(3,3)
        rec=dict(root_id=rid,scene_id=row['scene_id'],index=i,start_node=meta['start_node'],end_node=meta['end_node'],motion=motion,kind=kind,
            poses=len(v[3]),duration=float(v[3][-1,0]-v[3][0,0]),resolution=float(v[-1][0]),known=len(v[0]),target=len(v[4]),new_free=int((v[5]==1).sum()),new_blocked=int((v[5]==2).sum()),
            known_terminal_differences=ex['known_terminal_differences'],candidates=len(keys),out_of_domain=dense['out_of_domain_positive'],cpu_cuda_max_error=float(np.max(abs(cpu[:,16:]-gpu))),
            cm=cm.tolist(),exact_replay=True,exact_candidate_keys=True)
        records.append(rec)
        path=OUT/'examples'/('%02d'%len(records));path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,keys=keys,y=y,probability=p,prediction=hard,path=v[3],gkeys=v[0],gstates=v[1],gheights=v[2],geometry=v[-1])
        print(rec,flush=True)
    save(OUT/'trace.json',dict(examples=records,strata=dict(counts),checks='Independent ancestry restart and update dictionary; exact cell identities, metadata, cache, CPU/CUDA, and target comparisons.'))


def baseline():
    from eventwm_graph.stage2_hf.spatial_eval import score
    from eventwm_graph.stage2_hf.light_eval import aggregate
    from eventwm_graph.stage2_segment.api import SegmentPredictor
    predictor=SegmentPredictor(CKPT,device='cuda:0',execution='optimized',future_format='native',feature_threads=1)
    manifest=read(HF/'light_eval_cache/manifest.json');packed=read(HF/'packed/dev_pilot/manifest.json');pm={r['root_id']:r for r in packed['rows']}
    records=[];pooled=np.zeros((14,3,3),np.int64);maxerror=0.;windows=0;zero_horizon=0;start=time.perf_counter()
    for row in manifest['rows']:
        a=load_arrays(pm[row['root_id']]);look={(e['start_node'],e['end_node']):i for i,e in enumerate(pm[row['root_id']]['segments'])}
        with np.load(row['path'],allow_pickle=False)as z:p=z['baseline'];y=z['y'];x=z['x']
        cms=[];scores=[];empt=np.zeros(14,int);zero=0
        for meta in row['segments']:
            v=get_segment(a,look[(meta['start_node'],meta['end_node'])]);sl=slice(meta['start'],meta['end']);gt=y[sl]
            # Independent network recomputation on saved full precision covariates.
            with torch.inference_mode():
                prob=torch.cat([predictor.cuda.net(torch.from_numpy(x[b:min(b+65536,sl.stop)]).to('cuda:0'))[:,:3].softmax(-1).cpu() for b in range(sl.start,sl.stop,65536)]).numpy()
            if len(v[3])==1:prob.fill(0);prob[:,0]=1;zero_horizon+=1
            maxerror=max(maxerror,float(np.max(abs(prob-p[sl]))))
            cm,sc,ee=score(prob,gt);cms.append(cm);scores.append(sc);pooled+=cm;windows+=1
            if not meta['new_cells']:zero+=1;empt+=ee
        records.append(dict(scene_id=row['scene_id'],root_id=row['root_id'],cm=np.mean(cms,axis=0),scores=np.mean(scores,axis=0),empty=empt,zero_cases=zero))
    summary=aggregate(records);summary.update(windows=windows,zero_horizon=zero_horizon,network_cache_max_error=maxerror,checkpoint_sha256=sha(CKPT),cache_manifest_sha256=sha(HF/'light_eval_cache/manifest.json'),pooled_confusion=pooled.tolist(),seconds=time.perf_counter()-start)
    save(OUT/'baseline_reproduced.json',summary)
    print('BASELINE',summary['rules']['threshold_0.20'], 'windows', windows,'zero_horizon',zero_horizon,'maxprob_error',maxerror,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['inventory','trace','baseline']);args=parser.parse_args()
    globals()[args.action]()
