"""Model-independent extraction from unscored, frozen-boundary dev roots only."""
import collections,gzip,hashlib,importlib.util,io,json,os,shutil,sqlite3,sys,time
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from eventwm_graph.l2_final.common import OUT,OLD,CORE,sha,dump
spec=importlib.util.spec_from_file_location('native_visibility_reference',BASE/'scripts/l2_final_native_census.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)

def main():
    dest=OUT/'reobservation_supplement';dest.mkdir(exist_ok=False)
    protocol=BASE/'docs/l2/final_800k_20261023/REOBSERVATION_SUPPLEMENT_PROTOCOL.md';protocol_sha=sha(protocol)
    original=json.loads((OLD/'snapshot/manifest.json').read_text());dev={r['root_id'] for r in original['rows'] if r['split']=='dev'}
    scenes={r['scene_id'] for r in original['rows'] if r['split']=='dev'}
    train=json.loads((OUT/'snapshot/manifest.json').read_text());extra=json.loads((OUT/'supplement/manifest.json').read_text())
    train_keys={(r['scene_id'],r['seed']) for r in train['rows']+extra['rows']};train_ids={r['root_id'] for r in train['rows']+extra['rows']}
    c=sqlite3.connect('file:'+str(OUT/'snapshot/journal.sqlite3')+'?mode=ro',uri=True);jobs=c.execute("SELECT seed,scene,result FROM jobs WHERE state='sealed'").fetchall();c.close()
    candidates=[]
    for seed,scene,result in jobs:
        r=json.loads(result);rid=r['root_id'];h=int(hashlib.sha256(('l2-explore-20261021:'+scene).encode()).hexdigest()[:8],16)%100
        if scene in scenes and 70<=h<85 and rid not in dev and rid not in train_ids and (scene,seed) not in train_keys:candidates.append((rid,seed,scene,r['folder']))
    candidates.sort(key=lambda r:hashlib.sha256(('l2-final-reobservation-development-20261023:'+r[0]).encode()).hexdigest())
    selected=[];ledger=[];start=time.perf_counter()
    for rid,seed,scene,folder in candidates:
        folder=Path(folder);m=json.loads((folder/'manifest.json').read_text())
        if m['split']!='train':ledger.append(dict(root_id=rid,status='source_protected_metadata_only'));continue
        assert m['sealed'] and m['root_id']==m['future_id']==rid and m['scene_id']==scene and m['independent_futures']==1
        assert all(m['version'].get(k)==train['core_reference'][k] for k in CORE)
        def payload(name):
            raw=(folder/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==m['files'][name];return raw
        legal=native.decode_payload(gzip.decompress(payload('legal_root.json.gz')),True)
        queries=json.loads(gzip.decompress(payload('queries.json.gz')));outcomes=json.loads(gzip.decompress(payload('outcomes.json.gz')))
        nodes=native.decode_payload(gzip.decompress(payload('future_observations.json.gz')))
        with np.load(io.BytesIO(payload('privileged_future.npz')),allow_pickle=False) as f:times=f['timestamps'].copy();positions=f['positions'].copy()
        qi=max(range(len(queries)),key=lambda i:(len(queries[i]['compiled']['actions']),queries[i]['query_id']))
        obs=[nodes[j]['observation'] for j in outcomes[qi]['node_ids']];rootob=legal['history'][-1];now=float(rootob['timestamp']);state={};event=None
        for index,ob in enumerate([rootob]+obs):
            t=float(ob['timestamp'])
            if t>now+3.06+1e-8:break
            ids,clean,failure=native.match_physical(ob,times,positions)
            if not native.available(ob) or not clean or ids is None:state={};continue
            for actor in list(state):
                if actor not in ids and state[actor][2] is None:state[actor]=(state[actor][0],state[actor][1],t)
            for actor in sorted(ids):
                if actor in state:
                    previous,source_id,miss=state[actor];gap=t-previous
                    if miss is not None and .5-1e-8<=gap<=2.+1e-8 and times[-1]>=t+.5-1e-8:
                        event=dict(root_id=rid,seed=seed,scene_id=scene,physical=actor,index=index-1,timestamp=t,last_positive=previous,first_miss=miss,
                                   time_since_last_certified_positive=gap,discrete_gap_duration_bounds=[t-miss,gap],source_id_changed=source_id!=ids[actor],
                                   native_remaining_horizon=float(times[-1]-t),query_id=queries[qi]['query_id'])
                        break
                state[actor]=(t,ids[actor],None)
            if event:break
        ledger.append(dict(root_id=rid,scene_id=scene,status='selected' if event else 'no_strict_eligible_event'))
        if event:
            target=dest/'roots'/rid;target.mkdir(parents=True);files={**m['files'],'manifest.json':sha(folder/'manifest.json')}
            for name,expected in files.items():
                assert Path(name).name==name and sha(folder/name)==expected;shutil.copyfile(folder/name,target/name)
                assert sha(target/name)==expected and sha(folder/name)==expected;os.chmod(target/name,0o444)
            os.chmod(target,0o555);event.update(files=files,source=str(folder));selected.append(event)
        if len(ledger)%250==0:print(json.dumps(dict(inspected=len(ledger),selected=len(selected),seconds=time.perf_counter()-start)),flush=True)
        if len(selected)>=256:break
    assert sha(protocol)==protocol_sha
    dump(dest/'manifest.json',dict(schema='l2_final_strict_native_reobservation_development_1',protocol_sha256=protocol_sha,
         journal_sha256=sha(OUT/'snapshot/journal.sqlite3'),candidate_roots=len(candidates),inspected_roots=len(ledger),roots=len(selected),
         scenes=len({r['scene_id'] for r in selected}),rows=selected,inspection_ledger=ledger,protected_payloads_opened=False,seconds=time.perf_counter()-start))
    os.chmod(dest/'manifest.json',0o444)
    if (dest/'roots').exists():os.chmod(dest/'roots',0o555)
    print(json.dumps(dict(complete=True,inspected=len(ledger),roots=len(selected),scenes=len({r['scene_id'] for r in selected}),seconds=time.perf_counter()-start)),flush=True)
if __name__=='__main__':main()
