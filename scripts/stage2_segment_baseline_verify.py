"""Verify the matched baseline adapter against untouched dfde4322 execution."""
import time
import numpy as np
from eventwm_graph.stage2_segment.benchmark import BaselineComplete,inputs,reference_identity
from eventwm_graph.stage2_segment.api import SegmentPredictor
from eventwm_graph.stage2_segment.common import OUT,CHECKPOINT,dump,snapshot


def main():
    if (OUT/'baseline_untouched_source_equivalence.json').exists():raise FileExistsError('preserve completed baseline verification')
    source=reference_identity();original=BaselineComplete(CHECKPOINT)
    adapted=SegmentPredictor(CHECKPOINT,execution='reference',future_format='snapshot')
    examples={}
    for group in inputs('single').values():
        for ex in group:examples.setdefault((ex['root_id'],ex['start_node'],ex['end_node']),ex)
    rows=[];tick=time.perf_counter()
    for ex in examples.values():
        a=original.predict(partial_graph=ex['graph'],trajectory=ex['trajectory'])
        b=adapted.predict(partial_graph=ex['graph'],trajectory=ex['trajectory'])
        for name in ('candidate_cells','probabilities','states','heights','unknown_cells'):np.testing.assert_array_equal(a[name],b[name])
        assert a['static_updates']==b['static_updates'] and a['future_partial_graph']==b['future_partial_graph']
        rows.append(dict(root_id=ex['root_id'],start_node=ex['start_node'],end_node=ex['end_node'],candidates=len(a['states']),exact=True))
    dump(OUT/'baseline_untouched_source_equivalence.json',dict(passed=True,source=snapshot(),untouched_original=source,segments=len(rows),
        candidates=sum(r['candidates'] for r in rows),arrays_and_complete_future_graphs_exact=True,rows=rows,seconds=time.perf_counter()-tick))


if __name__=='__main__':main()
