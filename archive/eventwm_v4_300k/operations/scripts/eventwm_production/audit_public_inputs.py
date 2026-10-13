from settings import PROJECT as PROJECT_PATH, DATA as DATA_PATH, SCIENCE as SCIENCE_PATH, TARGET as TARGET_QUERIES
import collections,hashlib,json,pathlib,sqlite3,sys,time
from active_release import active_repo
P=PROJECT_PATH;R=active_repo();sys.path.insert(0,str(R/'scripts'))
from collector import core
from collector.observation import TRACK_FIELDS
D=DATA_PATH;start=time.time();counts=collections.Counter();audit_counts=collections.defaultdict(collections.Counter);ids=set();opened=set()
with sqlite3.connect((DATA_PATH/'pipeline.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
    roots=[json.loads(v) for v, in db.execute("SELECT value FROM roots WHERE json_extract(value,'$.state')='completed' ORDER BY ordinal")]
original=core.read
for root in roots:
    base=D/root['path'];cache={};allowed={'legal_observations/history.json','queries/queries.json','legal_observations/image_storage.json'}
    def captured(path):
        relative=str(pathlib.Path(path).relative_to(base));assert relative in allowed
        opened.add(relative)
        if relative not in cache:cache[relative]=original(path)
        return cache[relative]
    core.read=captured
    queries=original(base/'queries/queries.json');legal=original(base/'roots/legal_root.json')
    for q in queries:
        value=core.load_wm_input(base,q['query']['query_id']);assert set(value)=={'history','history_images','query'} and value['history_images'] is None
        assert value['query']['root_id']==root['root_id'] and value['query']['generation_config']['human_blind']
        assert q['query']['query_id'] not in ids;ids.add(q['query']['query_id'])
        for h in value['history']:
            assert h['timestamp']<=legal['timestamp']+1e-8
            for t in h['human_tracks']:
                assert set(t)==set(TRACK_FIELDS) and t['first_seen_time']<=t['last_seen_time']<=h['timestamp']+1e-8
                assert t['num_observations']>=1 and t['observation_age']>=-1e-8
    core.read=original
    counts['role_'+root['role']]+=root['stats']['primary_queries'];counts['category_'+root['stats']['legal_state']['category']]+=1
    counts['human_count_'+str(root['human_count'])]+=1;counts['static_violations']+=root['stats']['static_violations'];counts['invalid_retained']+=root['stats']['invalid']
    for name,v in original(base/'audits/audits.json').items():
        audit_counts[name][v['status']]+=1
        if name!='human_reactivity':assert v['status']=='PASS'
assert counts['static_violations']==counts['invalid_retained']==0
report=dict(passed=True,timestamp=time.time(),roots=len(roots),primary_query_inputs=len(ids),opened_input_artifacts=sorted(opened),
            counts=dict(counts),audits={k:dict(v) for k,v in audit_counts.items()},no_future_or_privileged_truth_opened=True,
            human_reactivity_note='Diagnostic FAIL means no robot-dependent human motion was observed; it is not an integrity failure.',
            seconds=time.time()-start,source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
            versioned_geometry_roots=sum(bool(r.get('recipe',{}).get('episode_sampler')) for r in roots))
(P/'output/eventwm_72h_public_input_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
