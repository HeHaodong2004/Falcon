"""Extra cost diagnostic on actual intermediate graphs and longer legal histories.

It supplements the original fixed32 workloads, never replaces their results or
changes the accuracy universe. Every query is a recorded continuation/prefix;
only ancestor/current sensor images are exposed to the predictor.
"""
from collections import defaultdict
import hashlib
import numpy as np
from .diagnose import OUT,HF,read,save,source_rows,member
from .range_history import ancestor_frames,validate
from eventwm_graph.stage2_segment.data import replay


def workloads(sensor_directory='range_branches',name='branch_cost_inventory'):
    source,_=source_rows('dev');source={r['root_id']:r for r in source};selected=[];inventory=[]
    for row in read(HF/'light_eval_cache/manifest.json')['rows']:
        rid=row['root_id'];raw=source[rid];groups=defaultdict(list)
        for e in replay(raw,'mixed'):
            if 'audit'not in e and e['start_node']is not None:groups[e['start_node']].append(e)
        sensor_path=OUT/sensor_directory/(rid+'.npz');receipt=read(sensor_path.with_suffix('.json'))
        from .diagnose import sha
        if receipt['sha256']!=sha(sensor_path)or receipt['identity']['files']!=raw['files']:raise ValueError('sensor source identity changed')
        with np.load(sensor_path)as z:sensor={k:z[k]for k in z.files}
        candidates=[]
        for sid,examples in groups.items():
            unique={}
            for e in examples:
                p=e['trajectory']
                for length in range(2,len(p)+1):
                    q=p[:length].copy();unique[hashlib.sha256(q.tobytes()).hexdigest()]=q
            take=ancestor_frames(sensor['parents'],sensor['history_count'],sid)
            if len(unique)>=6:candidates.append((len(take),len(examples[0]['graph']['cells']),int(sid),examples[0]['graph'],unique,take))
        inventory.append(dict(root_id=rid,intermediate_groups=len(groups),groups_with_six_distinct_continuations=len(candidates)))
        if not candidates:continue
        _,_,sid,graph,unique,take=max(candidates,key=lambda c:c[:3])
        paths=sorted(unique.items(),key=lambda item:(len(item[1]),item[0]));indices=np.linspace(0,len(paths)-1,6).astype(int);paths=[dict(trajectory=paths[i][1].tolist())for i in indices]
        packet=dict(ranges=sensor['depth'][take],origins=sensor['origins'][take],rotations=sensor['rotations'][take],timestamps=sensor['timestamps'][take],sensor_protocol='ideal_current_jaw_range_dynamic_occlusion_1')
        validate(packet['ranges'],packet['origins'],packet['rotations'],packet['timestamps'],graph['timestamp']);assert abs(float(packet['timestamps'][-1])-graph['timestamp'])<1.e-7
        legal=member(raw,'legal_root.json.gz');nodes=member(raw,'future_observations.json.gz');tracks=[o['tracks']for o in legal['history']]+[n['observation']['tracks']for n in nodes]
        for p in paths:assert p['trajectory'][0][0]==graph['timestamp']and len(p['trajectory'])>1
        selected.append((dict(root_id=rid,scene_id=row['scene_id'],start_node=sid,graph=graph,paths=paths),packet,[tracks[i]for i in take]))
    save(OUT/(name+'.json'),dict(role='Extra intermediate-graph cost diagnostic only; original accuracy and32-map benchmark universes retained.',sensor_directory=sensor_directory,selection='One eligible intermediate graph per originalpilotRoot: most legal past frames, then known-cell count, then nodeid. Six distinct recorded continuations/prefixes spread over available path lengths. No future sensor data.',roots=inventory,selected=[dict(root_id=e['root_id'],start_node=e['start_node'],known_cells=len(e['graph']['cells']),range_frames=len(p['ranges']),path_lengths=[len(v['trajectory'])for v in e['paths']])for e,p,t in selected]))
    return selected


if __name__=='__main__':
    value=workloads();print('EXTRA_BRANCH_WORKLOADS',len(value),'range_frames',[len(p['ranges'])for e,p,t in value],flush=True)
