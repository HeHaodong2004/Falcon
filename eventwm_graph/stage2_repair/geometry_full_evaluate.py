"""All original development cells, revised geometry/camera input contract.

Scene geometry and trial camera arrays are the only expanded inputs. Actual
observation updates remain scoring targets outside GeometryPredictor. Each
original candidate key and label vector is checked against baseline archives.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .diagnose import OUT, HF, read, save, sha, source_rows, member
from .geometry_api import GeometryPredictor
from .geometry_reference import trial_cameras
from .full_evaluate import BASELINE, labels, summarize, codes
from eventwm_graph.stage2_segment.data import replay
from eventwm_graph.stage2_segment.evaluate import groups


def run(name='geometry_selected',device='cuda:0',shard=0,shards=1,pilot=False):
    torch.set_num_threads(1);tick=time.perf_counter();rows,manifest=source_rows('dev')
    if pilot:
        selected=set(read(HF/'protocol.json')['pilot_selection_roots']);rows=[r for r in rows if r['root_id']in selected]
    scene_rows=defaultdict(list)
    for r in rows:scene_rows[r['scene_id']].append(r)
    dest=OUT/'full_evaluation'/name;dest.mkdir(parents=True,exist_ok=True)
    sources={name:sha(Path(__file__).with_name(name))for name in ['geometry_api.py','geometry_cuda.py','geometry_scene.py','geometry_reference.py','cuda_module.py']}
    model_sha=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    save(dest/('identity_%02d.json'%shard),dict(contract='REVISED current graph, trajectory, supplied trial camera poses, static scene collision mesh/navmesh. Static-only geometry, no active-human model.',source_manifest_sha256=sha(manifest),sources=sources,evaluation_source_sha256=sha(__file__),model_sha256=model_sha,device=device,shard=shard,shards=shards,pilot=pilot,original_roots=len(rows)))
    completed=[]
    for scene in sorted(scene_rows)[shard::shards]:
        model=None;asset=None
        try:
            for row in scene_rows[scene]:
                path=dest/(row['root_id']+'.json.gz')
                if path.exists():
                    value=json.loads(gzip.decompress(path.read_bytes()))
                    if value['checkpoint_sha256']!=model_sha:raise ValueError('stale geometry evaluation receipt')
                    completed.append(value);continue
                private=member(row,'privileged_initialization.json.gz');ep=private['episode'];legal=member(row,'legal_root.json.gz');nodes=member(row,'future_observations.json.gz');signature=(ep['scene_id'],ep['info']['graph_navmesh'],legal['partial_graph']['resolution'])
                if signature!=asset:
                    if model is not None:model.close()
                    model=GeometryPredictor(*signature,legal['sensor'],device=device);asset=signature
                    print('SCENE',scene,'certificates',len(model.asset.arrays['keys']),'asset_prepare_seconds',model.asset.prepare_seconds,flush=True)
                frozen=json.loads(gzip.decompress((BASELINE/(row['root_id']+'.json.gz')).read_bytes()));assert frozen['checkpoint_sha256']=='1ba6844d8b7eb336c46b41125eefae16e413c4ba408da9b0a1dc47ebcef66bfd'
                old={(e['start_node'],e['end_node']):e for e in frozen['records']}
                with np.load(BASELINE/(row['root_id']+'.probabilities.npz'),allow_pickle=False)as z:old_y=z['y'];offsets=z['offsets']
                old_slices={(e['start_node'],e['end_node']):slice(offsets[i],offsets[i+1])for i,e in enumerate(frozen['records'])};records=[];audit=None;stamp=time.perf_counter()
                for ex in replay(row,'mixed'):
                    if 'audit'in ex:audit=ex['audit'];continue
                    p,o,r=trial_cameras(legal,nodes,ex['start_node'],ex['end_node']);np.testing.assert_array_equal(p,ex['trajectory'])
                    delta=model.predict(partial_graph=ex['graph'],trajectory=p,camera_origins=o,camera_rotations=r);keys=delta['candidate_cells'];hard=delta['states'];y,th,missing=labels(keys,ex['target']);previous=old[ex['start_node'],ex['end_node']]
                    assert hashlib.sha256(keys.astype('<i4').tobytes()).hexdigest()==previous['domain_sha256']
                    np.testing.assert_array_equal(y,old_y[old_slices[ex['start_node'],ex['end_node']]])
                    assert np.count_nonzero(y)+len(missing)==previous['true_counts'][0]
                    cm=np.bincount(3*y.astype(np.int64)+hard,minlength=9).reshape(3,3)
                    for s,h in missing:cm[s,0]+=1
                    future=delta['future_partial_graph'];columns=future.cell_columns;prepared=model.prepare_graph(ex['graph']);base=prepared.future_base.table;fc=codes(columns['cell']);ki=np.searchsorted(fc,codes(base.keys));np.testing.assert_array_equal(columns['cell'][ki],base.keys)
                    for field,expected in [('state',base.states),('height',base.heights),('first_seen',base.first),('last_seen',base.last)]:np.testing.assert_array_equal(columns[field][ki],expected)
                    chosen=np.flatnonzero(hard);ni=np.searchsorted(fc,codes(keys[chosen]));np.testing.assert_array_equal(columns['cell'][ni],keys[chosen]);np.testing.assert_array_equal(columns['state'][ni],hard[chosen]);np.testing.assert_array_equal(columns['height'][ni],delta['static_updates'].heights);assert len(columns['cell'])==len(base.keys)+len(chosen)and future['hypothetical']is True
                    correct=(y>0)&(hard==y);height=[float(abs(delta['heights'][correct]-th[correct]).sum()),int(correct.sum())]
                    records.append(dict(start_node=ex['start_node'],end_node=ex['end_node'],cm=cm.tolist(),baseline_cm=previous['confusion'][6],scores=None,groups=groups(ex),new_cells=len(ex['target']),poses=len(p),empty_prediction=not bool(len(chosen)),candidates=len(keys),out_of_domain=len(missing),known_terminal_differences=ex['known_terminal_differences'],height_error=height,future_verified=True))
                assert audit is not None and len(records)==len(old)
                value=dict(root_id=row['root_id'],scene_id=row['scene_id'],human_count=ep['info']['human_num'],checkpoint_sha256=model_sha,scene_asset_sha256=model.identity,records=records,audit=audit,seconds=time.perf_counter()-stamp)
                temporary=path.with_suffix('.tmp');temporary.write_bytes(gzip.compress(json.dumps(value,separators=(',',':')).encode()));temporary.replace(path);completed.append(value)
                if len(completed)%16==0:print(dict(shard=shard,roots=len(completed),seconds=time.perf_counter()-tick),flush=True)
        finally:
            if model is not None:model.close()
    result=summarize(completed);result.update(model_sha256=model_sha,shard=shard,shards=shards,seconds=time.perf_counter()-tick,contract='REVISED geometry/camera inputs; deterministic static certificate engine; active-human occlusion omitted; probability scores not computed, primary accuracy is exact confusion.')
    save(dest/('summary_%02d.json'%shard),result)
    if shards==1:
        if not pilot:assert result['roots']==2110 and result['windows']==47128
        save(dest/'summary.json',result)
    print('SUMMARY',result['roots'],result['windows'],result['hierarchical'],flush=True)


def merge(name):
    dest=OUT/'full_evaluation'/name;rows=[json.loads(gzip.decompress(p.read_bytes()))for p in dest.glob('*.json.gz')];r=summarize(rows);assert r['roots']==2110 and r['windows']==47128
    assert len({x['checkpoint_sha256']for x in rows})==1;r['model_sha256']=rows[0]['checkpoint_sha256'];r['contract']='REVISED static scene geometry + supplied trial camera poses, evaluated on all original development windows/cells.';save(dest/'summary.json',r);print('FULL SUMMARY',r['hierarchical'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='geometry_selected');p.add_argument('--device',default='cuda:0');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--pilot',action='store_true');p.add_argument('--merge',action='store_true');a=vars(p.parse_args());do_merge=a.pop('merge')
    if do_merge:merge(a['name'])
    else:run(**a)
