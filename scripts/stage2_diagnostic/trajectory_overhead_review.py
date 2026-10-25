"""Cached raw timing quantile/scope review; no inference or new timing."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'output/stage2_diagnostic_20261025/visibility'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=read(BASE/'full_adapter_protocol_v3.json');cp=read(BASE/'decision_protocol_v3.json');full=read(BASE/'full_adapter_latency_v3/metrics.json');fc=read(BASE/'full_adapter_latency_v3/calls.json');component=read(BASE/'decision_overhead_v3/metrics.json');cc=read(BASE/'decision_overhead_v3/calls.json')
 for path,value in p['source'].items():assert sha(ROOT/path)==value
 assert sha(ROOT/'scripts/stage2_diagnostic/decision_overhead.py')==cp['source_sha256']
 assert full['protocol_sha256']==sha(BASE/'full_adapter_protocol_v3.json') and component['protocol_sha256']==sha(BASE/'decision_protocol_v3.json')
 assert len(fc)==320 and len(cc)==1120 and full['unique_inputs']==16 and component['cases']==16
 assert full['gc_enabled'] and component['gc_enabled'] and full['actual_future_graphs_inside_timer']
 assert not full['production_changed'] and not full['probabilities_changed'] and not full['primary_gates_redefined']
 for cohort in full['workloads']:
  for rule,m in full['workloads'][cohort].items():
   values=np.asarray([r['total_seconds']*1000 for r in fc if r['cohort']==cohort and r['rule']==rule]);assert len(values)==m['calls']==80
   np.testing.assert_array_equal(np.quantile(values,[.5,.95,.99]),[m[k]for k in ('p50_ms','p95_ms','p99_ms')])
 for rule,m in component['rules'].items():
  values=np.asarray([r['total_seconds']*1000 for r in cc if r['rule']==rule]);assert len(values)==m['calls']==80
  np.testing.assert_array_equal(np.quantile(values,[.5,.95,.99]),[m[k]for k in ('p50_ms','p95_ms','p99_ms')])
 proofs=read(BASE/'full_adapter_latency_v3/proofs.json');assert len(proofs)==16
 assert all(r['probabilities_unchanged']and r['states_match_frozen_rule']and r['future_graph_materialized']for r in proofs)
 result=dict(status='PASS_WITH_DECLARED_SCOPE',review_source_sha256=sha(__file__),protocol_sha256=sha(BASE/'full_adapter_protocol_v3.json'),component_protocol_sha256=sha(BASE/'decision_protocol_v3.json'),full_calls=320,component_calls=1120,unique_inputs=16,raw_quantiles_independently_recomputed_exact=True,
             reviewed_timer_boundary='external_predict includes original complete predict, extra public graph lookup, decision, sparse updates, and construction of second complete future graph; startup and proof snapshots separate. Component timing excludes original model/map prep and is not end-to-end.',
             full_metrics=full['workloads'],verdict='Secondary retained-cache diagnostic only; no substitute primary gate or adapter promotion.',
             limitations=['16 fixed representative/dense examples and80calls per cohort/rule do not establish the185example original gate, map-cold, K2/4/8, integrated latency, concurrency, or robust rare-tail behavior.',
                         'Original argmax graph is redundantly constructed before threshold graph; results are conservative for this external adapter and not an optimized production threshold implementation.',
                         'No changed threshold probability calibration; decision outputs have more updates and lower FREE precision. Timing does not validate hard-map reliability.',
                         'Startup is separately recorded, not hidden; joint process RSS is not per-rule or model-only memory. Adapter GPU peak and component standalone RSS unmeasured.',
                         'v1/v2/source/guard failure artifacts preserved; v3 admission helper checks frozen source/checkpoint/input identities without executing historical helper.',
                         'No independent new timing or map reconstruction was executed for this review; static construction boundary and saved proof receipts inspected.'])
 dest=ROOT/'output/stage2_diagnostic_20261025/trajectory/overhead_independent_review.json'
 with dest.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k]for k in ('status','full_calls','component_calls','unique_inputs','raw_quantiles_independently_recomputed_exact')},indent=2))
if __name__=='__main__':main()
