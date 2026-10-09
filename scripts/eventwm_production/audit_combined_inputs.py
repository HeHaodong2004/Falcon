from settings import PROJECT as PROJECT_PATH, DATA as DATA_PATH, SCIENCE as SCIENCE_PATH, TARGET as TARGET_QUERIES
"""Audit actual model-input reads across the verified old/new combined index."""
import collections,json,pathlib,sqlite3,sys,time
from active_release import active_repo
P=PROJECT_PATH;sys.path.insert(0,str(active_repo()/'scripts'))
from collector import core
from collector.observation import TRACK_FIELDS
started=time.time();counts=collections.Counter();ids=set();opened=set();original=core.read
with sqlite3.connect('file:'+str(P/'output/eventwm_300k_combined_index/verified.sqlite3')+'?mode=ro',uri=True) as db:
    roots=list(db.execute('SELECT root_id,dataset,path,role,queries FROM roots ORDER BY root_id'))
for rid,dataset,path,role,expected in roots:
    base=pathlib.Path(dataset)/path;cache={};allowed={'legal_observations/history.json','queries/queries.json','legal_observations/image_storage.json'}
    def captured(file):
        relative=str(pathlib.Path(file).relative_to(base));assert relative in allowed
        opened.add(relative)
        if relative not in cache:cache[relative]=original(file)
        return cache[relative]
    root=original(base/'roots/legal_root.json');queries=original(base/'queries/queries.json')
    core.read=captured
    try:
        for q in queries:
            qid=q['query']['query_id'];assert qid not in ids;ids.add(qid)
            value=core.load_wm_input(base,qid)
            assert set(value)=={'history','history_images','query'} and value['history_images'] is None
            assert value['query']['root_id']==rid and value['query']['generation_config']['human_blind']
            for observation in value['history']:
                assert observation['timestamp']<=root['timestamp']+1e-8
                for track in observation['human_tracks']:
                    assert set(track)==set(TRACK_FIELDS)
                    assert track['first_seen_time']<=track['last_seen_time']<=observation['timestamp']+1e-8
                    assert track['num_observations']>=1 and track['observation_age']>=-1e-8
    finally:core.read=original
    assert len(queries)==expected
    counts[role]+=expected
result=dict(passed=True,timestamp=time.time(),verified_roots=len(roots),primary_query_inputs=len(ids),roles=dict(counts),opened_model_input_artifacts=sorted(opened),no_future_or_privileged_truth_in_model_input=True,seconds=time.time()-started)
(P/'output/eventwm_72h_combined_input_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
