from settings import PROJECT as PROJECT_PATH, DATA as DATA_PATH, SCIENCE as SCIENCE_PATH, TARGET as TARGET_QUERIES
"""Incremental static proposal strata accounting; no acceptance from labels."""
import collections,json,pathlib,sqlite3,time
P=PROJECT_PATH
D=DATA_PATH;OUT=P/'output/eventwm_72h_v4_sampler_stats.json'
cursor=-1;counts=collections.Counter();regions=collections.Counter();clearance=collections.Counter();scene_counts=collections.Counter()
while True:
    with sqlite3.connect((DATA_PATH/'pipeline.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
        rows=[(n,json.loads(v)) for n,v in db.execute('SELECT ordinal,value FROM roots WHERE ordinal>? ORDER BY ordinal LIMIT 512',(cursor,))]
    for n,r in rows:
        if r['state'] not in ('completed','failed'):break
        if r['state']=='failed' and len(r['attempts'])<2 and r['attempts'][-1].get('reason') in ('worker_timeout','root_timeout','interrupted_before_commit'):break
        sampler=r.get('recipe',{}).get('episode_sampler')
        if sampler:
            component=sampler['component'];category=r['recipe']['sampling'].get('target_category','natural');h=r['human_count']
            assert r['role']=='train'
            counts['attempted']+=1;counts['attempted_'+component]+=1;counts['attempted_category_'+category]+=1;counts['attempted_humans_'+str(h)]+=1
            scene_counts['attempted_'+r['scene']]+=1
            if r['state']=='completed':
                counts['accepted']+=1;counts['accepted_'+component]+=1;counts['accepted_category_'+category]+=1;counts['accepted_humans_'+str(h)]+=1
                scene_counts['accepted_'+r['scene']]+=1
                regions[r['scene']+'/'+r['start_region']]+=1
            ep=D/'catalogue/episodes'/f"{r['root_id']}.json"
            if ep.exists():
                audit=json.loads(ep.read_text()).get('info',{}).get('geometry_preparation',{})
                clearance.update(audit.get('clearance_histogram',{}))
                counts['static_candidate_attempts']+=audit.get('start_attempts',0);counts['static_feasible_candidates']+=audit.get('start_feasible',0)
        cursor=n
    report=dict(timestamp=time.time(),new_cursor=cursor,counts=dict(counts),accepted_regions=dict(regions),scene_strata=dict(scene_counts),static_candidate_clearance_0_1m_bins=dict(clearance),importance_weights='Not estimated; densities of coverage-conditioned sampler unavailable. Do not assume V3-distribution equivalence.',mixture_probabilities={'v3_support':.125,'geometry_feasible':.875},reference_episodes='Original V3 seeded shadow coverage; natural/unseen roles unchanged')
    temp=OUT.with_suffix('.tmp');temp.write_text(json.dumps(report));temp.replace(OUT)
    time.sleep(1 if len(rows)==512 and cursor==rows[-1][0] else 90)
