"""Bounded exploratory component profile, not an acceptance measurement."""
import json
import argparse
import time
from pathlib import Path
import numpy as np
from eventwm_graph.stage2_segment.benchmark import inputs, models
from eventwm_graph.stage2_segment.common import OUT, dump
from eventwm_graph.stage2_segment.api import SegmentPredictor

parser=argparse.ArgumentParser();parser.add_argument('--execution',choices=['reference','optimized'],default='reference');parser.add_argument('--name',default='exploratory_profile_new')
parser.add_argument('--future-format',choices=['snapshot','native'],default='snapshot')
parser.add_argument('--device',default='cpu')
parser.add_argument('--feature-threads',type=int,default=1)
args=parser.parse_args()
if (OUT/(args.name+'.json')).exists():raise FileExistsError('preserve previous profile')
modeldict = models(OUT/'training/pilot_ray/last.pt')
modeldict['candidate']=SegmentPredictor(OUT/'training/pilot_ray/last.pt',execution=args.execution,future_format=args.future_format,device=args.device,feature_threads=args.feature_threads)
examples = inputs('single')['representative'][:12]
rows = []
for i, ex in enumerate(examples):
    for model in modeldict.values():
        model.predict(partial_graph=ex['graph'], trajectory=ex['trajectory'])
    for repeat in range(5):
        for name in (('candidate','baseline') if (i+repeat)%2 else ('baseline','candidate')):
            model = modeldict[name]; model.clear_cache(); tick = time.perf_counter()
            delta = model.predict(partial_graph=ex['graph'], trajectory=ex['trajectory'])
            rows.append(dict(model=name, root_id=ex['root_id'], duration=ex['duration'],
                seconds=time.perf_counter()-tick, profile=model.last_profile, future_cells=len(delta['future_partial_graph']['cells'])))
summary = {}
for name in modeldict:
    group = [r for r in rows if r['model']==name]
    summary[name] = dict(p50_ms=float(np.median([r['seconds'] for r in group]))*1000,
        p95_ms=float(np.quantile([r['seconds'] for r in group], .95))*1000,
        components_ms={k:float(np.mean([r['profile'][k] for r in group if r['profile'].get(k) is not None]))*1000
                       for k in set().union(*(r['profile'] for r in group)) if k.endswith('_seconds') and any(r['profile'].get(k) is not None for r in group)})
dump(OUT/(args.name+'.json'), dict(summary=summary,calls=rows,not_acceptance=True,execution=args.execution,
    contention='full fitting extraction14workers; exploratory bottleneck profile only'))
print(json.dumps(summary),flush=True)
