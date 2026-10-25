"""Recorded-step audit of static feature geometry versus changing metadata."""
from collections import Counter,defaultdict
import json
import time
import numpy as np
from eventwm_graph.stage2.data import source_rows
from eventwm_graph.stage2_quality.update import FastPartialGraph
from eventwm_graph.stage2_segment.data import load
from eventwm_graph.stage2_segment.common import OUT,dump,snapshot


def same(a,b):
    if a is None or b is None:return a is None and b is None
    return a['state']==b['state'] and np.float32(a['height']).view(np.uint32)==np.float32(b['height']).view(np.uint32)


def main():
    dest=OUT/'geometry_reuse_audit';dest.mkdir(exist_ok=False)
    protocol=json.loads((OUT/'protocol.json').read_text());wanted=set(protocol['pilot_fit'][:64])
    rows,_=source_rows('train');selected=[r for r in rows if r['root_id'] in wanted];assert len(selected)==64
    records=[];tick=time.perf_counter();totals=Counter();actions=defaultdict(Counter)
    for index,row in enumerate(selected):
        legal,queries,outcomes,nodes,children=load(row);graph=FastPartialGraph.restore(legal['partial_graph']);counts=Counter()
        def visit(identity):
            node=nodes[identity];ob=node['observation'];keys={tuple(u['cell']) for u in ob['static_updates']};oldcells={k:graph.cells.get(k) for k in keys}
            old=(graph.timestamp,graph.revision,graph.ego_support,len(graph.visited),graph.global_nodes,graph.global_edges)
            delta=graph.update(ob);assert delta==node['graph_delta']
            reusable=all(same(value,graph.cells.get(k)) for k,value in oldcells.items())
            f64_same=all((a is None and graph.cells.get(k) is None) or (a is not None and graph.cells.get(k) is not None and a['state']==graph.cells[k]['state'] and np.float64(a['height']).view(np.uint64)==np.float64(graph.cells[k]['height']).view(np.uint64)) for k,a in oldcells.items())
            counts['saved_steps']+=1;counts['feature_geometry_unchanged']+=int(reusable);counts['f64_geometry_unchanged']+=int(f64_same)
            actions[str(node['action'])]['saved_steps']+=1;actions[str(node['action'])]['feature_geometry_unchanged']+=int(reusable)
            for child in children[identity]:visit(child)
            for key,value in oldcells.items():
                if value is None:graph.cells.pop(key,None)
                else:graph.cells[key]=value
            graph.timestamp,graph.revision,graph.ego_support,length,graph.global_nodes,graph.global_edges=old;del graph.visited[length:];graph._fast_grid=None
        for child in children[None]:visit(child)
        records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],counts=dict(counts)));totals.update(counts)
        if index%16==0:print(json.dumps(dict(roots=index+1,steps=totals['saved_steps'],seconds=time.perf_counter()-tick)),flush=True)
    dump(dest/'results.json',dict(roots=64,counts=dict(totals),actions={k:dict(v) for k,v in actions.items()},rows=records,source=snapshot(),
        reusable_step_fraction=totals['feature_geometry_unchanged']/totals['saved_steps'],seconds=time.perf_counter()-tick,
        scope='Engineering opportunity audit on actual saved legal parent/child states from fitting-only pilotRoots; no new simulator, synthetic labels, model fit or accuracy-protocol change',
        feature_dependence='Known coordinates/states/F32heightbits +fixedres/defaultheight/clearance; alltrajectoryfeatures remain segment-specific; currentF64metadata/identity/Base must still rebuild'))


if __name__=='__main__':main()
