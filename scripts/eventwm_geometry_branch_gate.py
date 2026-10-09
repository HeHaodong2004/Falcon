"""Complete native branch/audit gate for first four TRAIN successes in fixed128.

Selection is fixed from the completed contiguous root-only experiment, before
branch outcomes exist. No gate samples contribute to production counts.
"""
import collections, concurrent.futures as cf, json, pathlib, sqlite3, subprocess, time
import eventwm_geometry_v4 as geometry
import eventwm_mass_collection as mass
from collector.core import read, write, digest
from eventwm_pipeline_io import prepare_episode, reserve_attempt
from eventwm_pipeline_pool import PipelineWorker, PipelineValidator
from eventwm_pipeline_runner import preflight
from eventwm_execution_pool import stop_checkpoint
from nonvisual_runtime import scientific_root, scientific_history


def main():
    release=preflight();repo=pathlib.Path(__file__).resolve().parents[1]
    project=repo.parents[1];source=project/'output/eventwm_72h_geometry_native_v2'
    out=project/'output/eventwm_72h_v4_branch_gate';out.mkdir(exist_ok=True)
    assert not (out/'VERDICT.json').exists(),'Use a fresh gate directory'
    with sqlite3.connect('file:/tmp/eventwm_v4_300k/pipeline.sqlite3?mode=ro',uri=True) as db:
        scenes=json.loads(db.execute("SELECT value FROM meta WHERE key='scenes'").fetchone()[0])
    experiment=read(source/'RESULT.json')
    chosen=sorted([r for r in experiment['rows'] if r['arm']=='v4_geometry' and r['role']=='train' and r.get('accepted')],key=lambda r:r['sequence'])[:4]
    assert len(chosen)==4
    write(out/'catalogue/scenes.json',[s['scene'] for s in scenes]);mass.catalog_link(out)
    begin=time.perf_counter();pids=set();checkpoints={};rows=[];errors=[];jobs=[]
    workers=[];validator=None
    try:
        workers=[PipelineWorker(i,repo,out,out/'logs',pids,checkpoints,begin) for i in range(2)]
        validator=PipelineValidator(repo,out,out/'logs/validator',pids)
        for previous in chosen:
            root=geometry.spec(previous['sequence'],scenes,experiment['master_seed'])
            root.update(attempts=[],_v3_preparation_covered={})
            scene=next(s for s in scenes if s['name']==root['scene'])
            prep=prepare_episode(out,scene,root,{})
            assert prep['error'] is None,prep
            root.update(episode_sha256=prep['episode_sha256'],start_region=prep['start_region'])
            path=reserve_attempt(out,root,out/'logs',lambda r:None)
            def job(wid,qid=None,limit=None):
                return dict(root_id=root['root_id'],scene=root['scene'],output=str(path),worker_id=wid,query_id=qid,limit_steps=limit)
            result=workers[0].execute(job('root'));jobs.append(result)
            assert result['error'] is None,result['error']
            assert result.get('checkpoint');checkpoints[root['root_id']]=result['checkpoint']
            oldpath=source/'v4_geometry/roots'/previous['root_id']/'attempt_01'
            oldhistory=read(oldpath/'legal_observations/history.json');history=read(path/'legal_observations/history.json')
            assert scientific_history(oldhistory)==scientific_history(history),'Native legal history changed'
            oldstate=read(oldpath/'roots/privileged_root.json')['state'];state=read(path/'roots/privileged_root.json')['state']
            assert scientific_root(oldstate,oldhistory)==scientific_root(state,history),'Native root physics changed'
            queries=read(path/'queries/queries.json');assert len(queries)==33
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                # Each worker owns at most one request; no stale response reuse.
                for start in range(0,33,2):
                    pending=[pool.submit(workers[k].execute,job(f'b{i:03}',queries[i]['query']['query_id'])) for k,i in enumerate(range(start,min(33,start+2)))]
                    for f in pending:
                        answer=f.result();jobs.append(answer)
                        assert answer['error'] is None and not answer['label']['static_feasibility_violation'],answer
            for wid,i,limit in [('reverse_1',1,None),('reverse_0',0,None),('truncated_wait',32,10)]:
                answer=workers[0].execute(job(wid,queries[i]['query']['query_id'],limit));jobs.append(answer)
                assert answer['error'] is None,answer
            stop_checkpoint(checkpoints.pop(root['root_id']),pids)
            reply=validator.execute(path);assert reply['error'] is None
            audits=read(path/'audits/audits.json')
            labels=read(path/'labels/labels.json')
            write(out/'native_jobs.json',jobs)
            rows.append(dict(root_id=root['root_id'],source_sequence=previous['sequence'],role=root['role'],human_count=root['human_count'],primary_queries=33,preparation=prep,audits=audits,validation=reply,bytes=sum(p.stat().st_size for p in path.rglob('*') if p.is_file())))
            print('VALIDATED',root['root_id'],33,flush=True)
    except BaseException as exc:
        errors.append(repr(exc));raise
    finally:
        for cp in list(checkpoints.values()):stop_checkpoint(cp,pids)
        for worker in workers:worker.close()
        if validator:validator.close()
        report=dict(passed=not errors and len(rows)==4,errors=errors,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),geometry_version=geometry.VERSION,retained_native_primary_queries=len(rows)*33,counted_in_production=False,rows=rows,elapsed_seconds=time.perf_counter()-begin,native_cpu_seconds=sum(r.get('cpu_seconds',0) for r in jobs),selection='First four TRAIN root successes of completed contiguous128 proposal experiment; no branch outcome selection',physics_history_matches=True if len(rows)==4 else None)
        write(out/'VERDICT.json',report)
        print('GATE',report['passed'],report['retained_native_primary_queries'],flush=True)


if __name__=='__main__':main()
