"""Small matched CPU-reference/GPU threshold check, not a timing experiment."""
import json
import numpy as np
from .common import OUT,CHECKPOINT,THRESHOLDS,make_model,digest,save
from .decision_overhead import decision_states,_case_digest

def main():
    from eventwm_graph.stage2_segment.benchmark import inputs
    from eventwm_graph.stage2_segment.api import SegmentPredictor
    dest=OUT/'numerical/backend_threshold_v1';dest.mkdir(parents=True,exist_ok=True)
    protocol=dest/'protocol.json'
    if protocol.exists():raise FileExistsError('backend check protocol already exists')
    save(protocol,dict(scope='First8 frozen representative and8dense inputs; GPU vsCPUreference; no labels/no timingclaim',
        thresholds=THRESHOLDS.tolist(),precision='F32sum thenF64comparison',source_sha256=digest(__file__),
        caveat='Empirical16cohortinput check, NOT proof of universal backend hardthresholdidentity'))
    gpu=make_model();cpu=SegmentPredictor(CHECKPOINT,execution='reference',future_format='snapshot',threads=1)
    corpus=inputs('single');rows=[]
    for group in ('representative','dense'):
        for ex in corpus[group][:8]:
            graph=ex['graph'];path=ex['trajectory'];before=_case_digest(graph,path)
            a=gpu.predict(partial_graph=graph,trajectory=path);b=cpu.predict(partial_graph=graph,trajectory=path)
            np.testing.assert_array_equal(a['candidate_cells'],b['candidate_cells'])
            np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=2e-6,rtol=0)
            if _case_digest(graph,path)!=before:raise ValueError('backend check input changed')
            differences={f'{t:.2f}':int(np.count_nonzero(decision_states(a['probabilities'],t)!=decision_states(b['probabilities'],t))) for t in THRESHOLDS}
            rows.append(dict(cohort=group,input_sha256=before,candidates=len(a['states']),
                max_class_probability_error=float(abs(a['probabilities']-b['probabilities']).max()),
                max_revelation_probability_error=float(abs((a['probabilities'][:,1]+a['probabilities'][:,2]).astype(np.float64)-(b['probabilities'][:,1]+b['probabilities'][:,2]).astype(np.float64)).max()),
                argmax_disagreements=int(np.count_nonzero(a['states']!=b['states'])),threshold_disagreements=differences))
    save(dest/'results.json',dict(cases=len(rows),unique_inputs=len({r['input_sha256'] for r in rows}),rows=rows,
        total_disagreements={f'{t:.2f}':sum(r['threshold_disagreements'][f'{t:.2f}'] for r in rows) for t in THRESHOLDS},
        source_sha256=digest(__file__),checkpoint_unchanged=True,production_changed=False))
    print(json.dumps({'cases':len(rows),'selected_0.20_disagreements':sum(r['threshold_disagreements']['0.20'] for r in rows),
        'max_probability_error':max(r['max_class_probability_error'] for r in rows)}),flush=True)

if __name__=='__main__':main()
