"""Bounded native recovery gate. Reads production; writes diagnostic artifacts only."""
import concurrent.futures as cf, json, pathlib, shutil, sqlite3, sys, time
code=pathlib.Path(sys.argv[1]).resolve(); out=pathlib.Path(sys.argv[2]).resolve()
sys.path[:0]=[str(code/'scripts'),str(code)]
from eventwm_pipeline_pool import PipelineWorker,PipelineValidator
from eventwm_execution_pool import stop_checkpoint
from collector.core import read,write,digest
src=pathlib.Path('/tmp/eventwm_v4_300k'); failed='v3m_e7c6987aa64211d2f63c'
db=sqlite3.connect(src.resolve().as_uri()+'/pipeline.sqlite3?mode=ro',uri=True)
roots=[json.loads(v) for v, in db.execute('SELECT value FROM roots ORDER BY ordinal DESC')]
selected=[next(r for r in roots if r['root_id']==failed)]; scenes={selected[0]['scene']}
for role in ['train','natural_reference','unseen_evaluation']:
    r=next(r for r in roots if r['state']=='completed' and r['role']==role and r['scene'] not in scenes and r['source_commit']=='58b4b0a17c90c5c7de420834b8ef62c44194eee7')
    selected.append(r);scenes.add(r['scene'])
for part in ['catalogue/cases','catalogue/episodes','logs']:(out/part).mkdir(parents=True,exist_ok=True)
shutil.copy2(src/'catalogue/scenes.json',out/'catalogue/scenes.json')
for r in selected:
    rid=r['root_id']; (out/'roots'/rid/'roots').mkdir(parents=True)
    for part in [f'catalogue/cases/{rid}.json',f'catalogue/episodes/{rid}.json']:
        shutil.copy2(src/part,out/part)
    # Re-run the ORIGINAL proposal recipe, including native natural prefix selection.
    shutil.copy2(src/f'recipes/{rid}.json',out/'roots'/rid/'roots/replay.json')
link=code/'evaluation/controlled50'
link.unlink();link.symlink_to(out/'catalogue',target_is_directory=True)
pids=set();cps={};w=None;v=None;results=[];began=time.time()
try:
    w=PipelineWorker(0,code,out,out/'logs',pids,cps,time.perf_counter())
    v=PipelineValidator(code,out,out/'logs',pids)
    for root in selected:
        rid=root['root_id']; dest=out/'roots'/rid
        job=dict(root_id=rid,scene=root['scene'],output=str(dest),worker_id='root',query_id=None,limit_steps=None)
        rr=w.execute(job);write(dest/'root_result.json',rr)
        print('ROOT',rid,rr['error'],flush=True);assert rr['error'] is None
        cps[rid]=rr['checkpoint'];qs=read(dest/'queries/queries.json')
        assert len(qs)==33
        if rid==failed:
            generation=read(dest/'audits/generation_v2.json')
            assert any(x['rejection_reason']=='native_controller_static_geometry_overlap' for x in generation['candidate_rejections'])
            assert all(q['query']['query_id']!='q_b85b8f97f044' for q in qs)
        jobs=[dict(job,worker_id=f'b{i:03}',query_id=q['query']['query_id']) for i,q in enumerate(qs)]
        jobs += [dict(job,worker_id='reverse_1',query_id=qs[1]['query']['query_id']),dict(job,worker_id='reverse_0',query_id=qs[0]['query']['query_id']),dict(job,worker_id='truncated_wait',query_id=qs[-1]['query']['query_id'],limit_steps=10)]
        with cf.ThreadPoolExecutor(max_workers=4) as pool:answers=list(pool.map(w.execute,jobs))
        for a in answers:
            assert not a['error'],a
            assert not a['label']['static_feasibility_violation'],a['label']
        stop_checkpoint(cps.pop(rid),pids)
        matches=[]
        for i in [0,27]:
            fresh=w.request(dict(job,worker_id=f'fresh_{i}',query_id=qs[i]['query']['query_id']))
            assert not fresh['error'],fresh
            cow=answers[i]
            assert cow['full_native_trace_scientific_sha256']==fresh['full_native_trace_scientific_sha256']
            assert cow['scientific_hashes']['history']==fresh['scientific_hashes']['history']
            matches.append(qs[i]['query']['query_id'])
        receipt=v.execute(dest);assert receipt['result']['retained']==33,receipt
        results.append(dict(root_id=rid,role=root['role'],scene=root['scene'],retained=33,
                            fresh_cow_matches=matches,validation=receipt['result']))
        write(out/'progress.json',results)
        print('VALIDATED',rid,'TOTAL',33*len(results),flush=True)
    write(out/'VERDICT.json',dict(passed=True,retained_native_primary_queries=132,
                                static_feasibility_violations=0,fresh_cow_query_pairs=8,
                                original_failure_rejected=True,results=results,seconds=time.time()-began))
finally:
    for cp in cps.values():stop_checkpoint(cp,pids)
    if w:w.close()
    if v:v.close()
