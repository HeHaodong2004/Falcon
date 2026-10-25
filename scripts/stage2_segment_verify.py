"""Scientific equivalence of runtime packing on frozen real segment inputs."""
import argparse
import json
import time
import numpy as np
from eventwm_graph.stage2_segment.api import SegmentPredictor
from eventwm_graph.stage2_segment.benchmark import inputs
from eventwm_graph.stage2_segment.features import SegmentFeatures, legal_lines, legal_lines_reference
from eventwm_graph.stage2_segment.fused import transform
from eventwm_graph.stage2_segment.common import OUT, dump, sha, snapshot


def verify(checkpoint, name='runtime_equivalence_pilot', limit=None,future_format='snapshot',device='cpu',feature_threads=1):
    source = snapshot(); reference = SegmentPredictor(checkpoint, execution='reference'); optimized = SegmentPredictor(checkpoint, execution='optimized',future_format=future_format,device=device,feature_threads=feature_threads)
    from numba import set_num_threads
    set_num_threads(feature_threads)
    workload = inputs('single'); examples = {}; rows = []; maximum = np.zeros(88); tick = time.perf_counter()
    for items in workload.values():
        for ex in items:
            examples.setdefault((ex['root_id'],ex['start_node'],ex['end_node']),ex)
    selected = list(examples.values())[:limit]
    for index, ex in enumerate(selected):
        graph = ex['graph']; path = ex['trajectory']; prepared = reference.prepare_graph(graph)
        context = SegmentFeatures(graph,path,prepared); keys = context.candidates('both')
        x, h = context.transform(keys,'both'); fast, fh = transform(context,keys,cuda=optimized.cuda,device_output=optimized.cuda is not None)
        if hasattr(fast,'cpu'):fast=fast.cpu().numpy()
        difference = np.max(abs(x[:,16:104]-fast),axis=0) if len(keys) else np.zeros(88)
        maximum = np.maximum(maximum,difference)
        np.testing.assert_allclose(x[:,16:104],fast,atol=2e-6,rtol=0)
        np.testing.assert_array_equal(h,fh)
        np.testing.assert_array_equal(legal_lines_reference(keys,context.path,context.resolution,context.lo,context.states),
                                      legal_lines(keys,context.path,context.resolution,context.lo,context.states))
        a = reference.predict(partial_graph=graph,trajectory=path); b = optimized.predict(partial_graph=graph,trajectory=path)
        for field in ('candidate_cells','states','unknown_cells'):
            np.testing.assert_array_equal(a[field],b[field])
        for field in ('probabilities','heights'):
            np.testing.assert_allclose(a[field],b[field],atol=2e-6,rtol=0)
        af = a['future_partial_graph']; native=b['future_partial_graph'];bf=native.to_snapshot() if hasattr(native,'to_snapshot') else native
        assert set(af)==set(bf)
        for key in af:
            if key!='cells':assert af[key]==bf[key]
        assert len(af['cells'])==len(bf['cells'])
        for ac,bc in zip(af['cells'],bf['cells']):
            assert {k:v for k,v in ac.items() if k!='height'}=={k:v for k,v in bc.items() if k!='height'}
            assert abs(ac['height']-bc['height'])<=2e-6
        rows.append(dict(root_id=ex['root_id'],start_node=ex['start_node'],end_node=ex['end_node'],candidates=len(keys),
            max_probability_error=float(np.max(abs(a['probabilities']-b['probabilities']))) if len(keys) else 0,
            max_height_error=float(np.max(abs(a['heights']-b['heights']))) if len(keys) else 0,hard_states_exact=True,terminal_graph_tolerance_pass=True))
        if index%16==0:print(json.dumps(dict(segments=index+1,total=len(selected),seconds=time.perf_counter()-tick,max_feature_error=float(maximum.max()))),flush=True)
    report=dict(checkpoint_sha256=sha(checkpoint),source=source,segments=len(rows),candidates=sum(r['candidates'] for r in rows),
        tolerance_abs=2e-6,rtol=0,max_feature_error_per_active_column=maximum.tolist(),
        all_hard_states_exact=True,all_graph_topology_and_nonheight_metadata_exact=True,future_format=future_format,device=device,rows=rows,seconds=time.perf_counter()-tick)
    dump(OUT/(name+'.json'),report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--name',default='runtime_equivalence_pilot');p.add_argument('--limit',type=int)
    p.add_argument('--future-format',choices=['snapshot','native'],default='snapshot')
    p.add_argument('--device',default='cpu')
    p.add_argument('--feature-threads',type=int,default=1)
    verify(**vars(p.parse_args()))
