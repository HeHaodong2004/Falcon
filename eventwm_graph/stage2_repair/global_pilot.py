"""Matched accuracy check for shared full-current-map distance grids."""
import argparse
import time
import numpy as np
from .diagnose import OUT,HF,read,save,sha,source_rows
from .global_features import GlobalTimePredictor
from eventwm_graph.stage2_segment.data import replay
from eventwm_graph.stage2_hf.light_eval import aggregate
from eventwm_graph.stage2_hf.spatial_eval import score
from .full_evaluate import labels
from scripts.stage2_diagnostic.common import extended_rates


def evaluate(checkpoint,device='cuda:1'):
    model=GlobalTimePredictor(checkpoint,device=device,increment_format='columns');old={r['root_id']:r for r in read(HF/'light_eval_cache/manifest.json')['rows']};rows,_=source_rows('dev');rows=[r for r in rows if r['root_id']in old];records=[];pooled=np.zeros((14,3,3),np.int64);windows=0;tick=time.perf_counter()
    for row in rows:
        previous=old[row['root_id']];lookup={(e['start_node'],e['end_node']):e for e in previous['segments']}
        with np.load(previous['path'],allow_pickle=False)as z:oldx=z['x'];oldy=z['y']
        cms=[];scores=[];empty=np.zeros(14,int);zero=0
        for ex in replay(row):
            if'audit'in ex:continue
            delta=model.predict(partial_graph=ex['graph'],trajectory=ex['trajectory']);y,_,missing=labels(delta['candidate_cells'],ex['target']);assert not missing
            e=lookup[(ex['start_node'],ex['end_node'])];sl=slice(e['start'],e['end']);np.testing.assert_array_equal(y,oldy[sl])
            p=ex['trajectory'];expected=np.floor((oldx[sl,20:22].astype(np.float64)*10+p[0,[1,3]])/ex['graph']['resolution']).astype(np.int32);np.testing.assert_array_equal(delta['candidate_cells'],expected)
            cm,sc,ee=score(delta['probabilities'],y);cms.append(cm);scores.append(sc);pooled+=cm;windows+=1
            if not len(ex['target']):zero+=1;empty+=ee
        records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],cm=np.mean(cms,axis=0),scores=np.mean(scores,axis=0),empty=empty,zero_cases=zero))
    r=aggregate(records);r.update(windows=windows,checkpoint_sha256=sha(checkpoint),feature_scope='All supplied known cells for shared EDT/height; original candidates and labels exact; no hidden map.',pooled={k:extended_rates(cm)for k,cm in zip(r['rules'],pooled)},pooled_confusion=pooled.tolist(),seconds=time.perf_counter()-tick);save(OUT/'global_pilot.json',r)
    print('global-grid pilot',r['rules']['threshold_0.20']['label_aware_delta_iou'],r['rules']['threshold_0.20']['revelation_iou'],'seconds',r['seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:1');evaluate(**vars(p.parse_args()))
