"""Predeclared extremal saved-horizon supplement; never replaces primary gates."""
import gc
import gzip
import hashlib
import json
import pickle
import resource
import time
import numpy as np
from eventwm_graph.stage2_segment.common import BASE,OUT,dump,sha,snapshot


def order(ex):
    return (-float(ex['duration']),ex['root_id'],str(ex['start_node']),str(ex['end_node']))


def accepted_identity():
    primary=json.loads((OUT/'benchmarks/release_single_v1/protocol.json').read_text())
    manifest=json.loads((Path(primary['source']['path'])/'manifest.json').read_text())
    for rel,expected in manifest['files'].items():assert sha(BASE/rel)==expected,rel
    assert sha(primary['checkpoint'])==primary['checkpoint_sha256']
    assert sha(OUT/'benchmark_inputs/manifest.json')==primary['input_manifest_sha256']
    return dict(source=primary['source'],checkpoint_sha256=primary['checkpoint_sha256'],
                primary_protocol_sha256=sha(OUT/'benchmarks/release_single_v1/protocol.json'))


def input_digest(ex):
    digest=hashlib.sha256(pickle.dumps(ex['graph'],protocol=5))
    path=np.asarray(ex['trajectory'],dtype='<f8');digest.update(str(path.shape).encode());digest.update(path.tobytes(order='C'))
    return digest.hexdigest()


def no_primary_benchmark():
    import os
    for process in Path('/proc').glob('[0-9]*/cmdline'):
        if process.parent.name==str(os.getpid()):continue
        try:args=process.read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert 'eventwm_graph.stage2_segment.benchmark' not in args,'Competing benchmark: '+process.parent.name


def main():
    if not (OUT/'stability_release_dede_v1/COMPLETE.json').exists():raise RuntimeError('Wait for sustained service completion')
    from eventwm_graph.stage2_segment.api import SegmentPredictor
    from eventwm_graph.stage2_segment.benchmark import BaselineComplete,reference_identity,host_state,GC,gc_event
    from eventwm_graph.stage2_segment.data import replay
    from eventwm_graph.stage2.data import source_rows
    from eventwm_graph.stage2.benchmark import quantiles,wire
    from eventwm_graph.stage2_segment.common import CHECKPOINT
    config_path=OUT/'supplemental_tail_protocol.json';config=json.loads(config_path.read_text());dest=OUT/'supplemental_extreme_tail_v1';dest.mkdir(exist_ok=False)
    prepared=OUT/'prepared/full_mixed_ray';assert sha(prepared/'manifest.json')==config['fitting_manifest_sha256']
    bound_path=BASE/'docs/l2/stage2_segment_20261024/artifacts/training_shard_manifest.json.gz'
    with gzip.open(bound_path,'rt') as f:bound=json.load(f)
    assert bound['source_manifest_sha256']==config['fitting_manifest_sha256']
    expected={Path(s['sidecar']).resolve():s['sidecar_sha256'] for s in bound['shards']}
    paths=sorted(p.resolve() for p in prepared.glob('[0-9]*_[0-9]*.json'))
    assert len(expected)==3308 and set(paths)==set(expected)
    selected=[];seen=set();segments=0;metadata=hashlib.sha256();tick=time.perf_counter()
    for path in paths:
        digest=sha(path);assert digest==expected[path],str(path)
        metadata.update(path.name.encode());metadata.update(digest.encode());report=json.loads(path.read_text());best={}
        for ex in report['segments']:
            segments+=1;root=ex['root_id']
            if root not in best or order(ex)<order(best[root]):best[root]=ex
        for root,ex in best.items():
            assert root not in seen;seen.add(root);selected.append(ex)
        selected=sorted(selected,key=order)[:config['roots']]
    assert len(seen)==105838 and segments==2482285 and len(selected)==12
    dump(dest/'selection.json',dict(config_sha256=sha(config_path),selected=selected,roots_scanned=len(seen),windows_scanned=segments,
        metadata_identity=metadata.hexdigest(),sealed_shard_manifest_sha256=sha(bound_path),every_sidecar_authenticated=True,
        selection_seconds=time.perf_counter()-tick,scope='Fitting metadata used only to select real extreme runtime workloads'))
    rows,source=source_rows('train');lookup={r['root_id']:r for r in rows};cases=[]
    for spec in selected:
        found=None
        for ex in replay(lookup[spec['root_id']],'mixed'):
            if 'audit' not in ex and ex['start_node']==spec['start_node'] and ex['end_node']==spec['end_node']:found=ex
        assert found is not None and abs(found['trajectory'][-1,0]-found['trajectory'][0,0]-spec['duration'])<1e-9
        cases.append((spec,found))
    state=host_state();assert not state['competing_research'] and gc.isenabled();accepted=accepted_identity();gc.callbacks.append(gc_event)
    no_primary_benchmark()
    digests={spec['root_id']:input_digest(ex) for spec,ex in cases}
    checkpoint=OUT/'training/final/last.pt';models=dict(original=BaselineComplete(CHECKPOINT),final=SegmentPredictor(checkpoint,execution='optimized',future_format='native',device='cuda:1'))
    reference=SegmentPredictor(checkpoint);proofs=[];startup=[];holders={}
    dump(dest/'protocol.json',dict(config=config,source=snapshot(),reference_runtime=reference_identity(),host_state=state,
        source_manifest_sha256=sha(source),checkpoint_sha256=sha(checkpoint),gc_enabled=True,
        accepted_primary_identity=accepted,initial_input_digests=digests,
        scope='Additional actual extreme-horizon runtime cohort; primary frozen acceptance definitions/results unchanged'))
    for spec,ex in cases:
        for name,model in models.items():
            tick=time.perf_counter();holders[name]=model.predict(partial_graph=ex['graph'],trajectory=ex['trajectory'])
            startup.append(dict(root_id=spec['root_id'],model=name,seconds=time.perf_counter()-tick))
        expected=reference.predict(partial_graph=ex['graph'],trajectory=ex['trajectory']);actual=holders['final']
        for field in ('candidate_cells','states','unknown_cells'):np.testing.assert_array_equal(actual[field],expected[field])
        for field in ('probabilities','heights'):np.testing.assert_allclose(actual[field],expected[field],atol=2e-6,rtol=0)
        af=actual['future_partial_graph'].to_snapshot();ef=expected['future_partial_graph']
        assert {k:v for k,v in af.items() if k!='cells'}=={k:v for k,v in ef.items() if k!='cells'}
        assert len(af['cells'])==len(ef['cells'])
        for ac,ec in zip(af['cells'],ef['cells']):
            assert {k:v for k,v in ac.items() if k!='height'}=={k:v for k,v in ec.items() if k!='height'}
            assert abs(ac['height']-ec['height'])<=2e-6
        proofs.append(dict(root_id=spec['root_id'],hard_states_exact=True,
            full_future_graph_tolerance_pass=True,
            max_probability_error=float(np.max(abs(actual['probabilities']-expected['probabilities']))) if len(actual['states']) else 0.,
            max_height_error=float(np.max(abs(actual['heights']-expected['heights']))) if len(actual['states']) else 0.))
        assert input_digest(ex)==digests[spec['root_id']]
    del reference;calls=[]
    for index,(spec,ex) in enumerate(cases):
        for condition in config['conditions']:
            for repeat in range(config['repeats_per_condition']):
                for name in (('original','final') if (index+repeat)%2==0 else ('final','original')):
                    model=models[name]
                    if condition=='map_cold':model.clear_cache()
                    begin_gc=len(GC);cpu=time.process_time();tick=time.perf_counter()
                    holders[name]=model.predict(partial_graph=ex['graph'],trajectory=ex['trajectory'])
                    end=time.perf_counter();process_cpu=time.process_time()-cpu;delta=holders[name];before_wire=time.perf_counter();encoded=wire(delta);wire_seconds=time.perf_counter()-before_wire
                    calls.append(dict(model=name,condition=condition,repeat=repeat,root_id=spec['root_id'],duration=spec['duration'],poses=len(ex['trajectory']),
                        known_cells=len(ex['graph']['cells']),candidates=len(delta['states']),updates=len(delta['static_updates']),total_seconds=end-tick,
                        cpu_seconds=process_cpu,wire_seconds=wire_seconds,wire_bytes=len(encoded),profile=dict(model.last_profile),
                        gc_events=[g for g in GC[begin_gc:] if g['start']<end and g['end']>tick]))
    assert accepted_identity()==accepted
    for spec,ex in cases:assert input_digest(ex)==digests[spec['root_id']]
    dump(dest/'calls.json',calls);dump(dest/'equivalence.json',proofs)
    dump(dest/'metrics.json',dict(models={name:{condition:quantiles([r['total_seconds'] for r in calls if r['model']==name and r['condition']==condition])
        for condition in config['conditions']} for name in models},calls=len(calls),startup_calls=startup,
        minimum_duration=min(s['duration'] for s in selected),maximum_duration=max(s['duration'] for s in selected),
        maximum_saved_poses=max(len(e['trajectory']) for _,e in cases),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        memory_scope='Joint initialized process with both models, real inputs and JIT; not model-only memory',all_reference_proofs_pass=True,
        gc_enabled=gc.isenabled(),primary_gates_redefined=False))


if __name__=='__main__':main()
