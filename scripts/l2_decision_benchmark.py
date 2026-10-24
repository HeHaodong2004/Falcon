"""Reuse the frozen complete structural workload, adding the planning packet.

Original benchmark and evidence remain unchanged. This is an unbounded timing
run, not a way to bypass online cooperative deadline semantics. All event
endpoints and four selected successor candidates remain in the matched path.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import l2_partial_structural_benchmark as benchmark
from eventwm_graph.l2_partial.planning import packet_from_batch, PACKET_VERSION
from eventwm_graph.l2_partial.repair_profile import _read

SCRIPT_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
benchmark.OUTPUT_ROOT=benchmark.REPO/'output/l2_decision_ready_20261021'
_make_models=benchmark.make_models
_causal_input=benchmark._causal_input
_publish=benchmark._publish


def make_models():
    models,elapsed=_make_models()
    for name in ('structural_bound','structural_bound_unoptimized'):
        model=models[name];original=model.online_call
        def wrapped(belief,actual,queries,*args,_original=original,**kwargs):
            started=time.perf_counter()
            b,batch=_original(belief,actual,queries,*args,**kwargs)
            packet=packet_from_batch(b,queries,batch,pointgoal=actual.get('_planning_pointgoal',[3.,0.,0.]))
            batch.planning_packet=packet
            batch.timings_ms.update(packet=packet.timings_ms['packet'],complete_with_packet=(time.perf_counter()-started)*1000.)
            return b,batch
        model.online_call=wrapped
    return models,elapsed


def causal_input(model,snapshot,row):
    previous,actual,workload,queries,elapsed,history=_causal_input(model,snapshot,row)
    legal=_read(snapshot/'roots'/row['root_id']/'legal_root.json.gz')
    actual={**actual,'_planning_pointgoal':legal['pointgoal']}
    return previous,actual,workload,queries,elapsed,history


def publish(path,value):
    value['planning_packet_version']=PACKET_VERSION
    value['decision_benchmark_wrapper_sha256']=SCRIPT_HASH
    value['packet_included_in_outer_complete']=True
    value['workload']['packet_creation']='typed adapter includes graph/future references, distributions, per-world responses and all requested successors'
    return _publish(path,value)


if __name__=='__main__':
    benchmark.make_models=make_models
    benchmark._causal_input=causal_input
    benchmark._publish=publish
    benchmark.main()
