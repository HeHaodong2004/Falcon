"""Forecast error by legal observation age, with fixed3s identity support."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model
from eventforecastwm.train import load_roots,prepare_roots,prepared_batch,encode
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.guard import Guard


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--prepared-cache',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);guard=Guard(out.parent);model=load_model(a.checkpoint).eval()
    roots=load_roots(a.manifest,read(a.protocol)['development'],guard)
    prepare(model,roots,a.manifest,a.prepared_cache,prepare_roots,guard);rows=[]
    for root in roots:
        guard.check();batch=prepared_batch([root['windows'][0]],'cpu');belief=encode(model,batch)
        f=model.imagine(belief,horizon_s=3.);mask=batch['targets']['track_valid'][0,:12].all(0)
        mean=(f.positions*f.weights[:,:,None,None,None]).sum(1)[0]
        error=(mean-batch['targets']['track_positions'][0,:12]).norm(dim=-1)
        age=belief.core.age[0]
        for name,lo,hi in [('fresh',0,.25),('lost_025_1',.25,1.),('lost_1_3',1.,3.),('lost_ge3',3.,float('inf'))]:
            support=mask&(age>=lo)&(age<hi)
            if not support.any():continue
            rows.append(dict(root_id=root['root_id'],scene=root['scene'],stratum=name,agents=int(support.sum()),
                mean_age_s=float(age[support].mean()),position_error_by_tick=error[:,support].mean(-1).tolist(),
                velocity_valid_fraction=float(belief.core.velocity_valid[0,support].float().mean())))
    strata={}
    for name in sorted({r['stratum'] for r in rows}):
        selected=[r for r in rows if r['stratum']==name]
        strata[name]=dict(roots=len(selected),agents=sum(r['agents'] for r in selected),
            mean_age_s=float(np.mean([r['mean_age_s'] for r in selected])),
            position_error_by_tick=np.mean([r['position_error_by_tick'] for r in selected],0).tolist())
    save_json(out,dict(checkpoint_sha256=sha256(a.checkpoint),manifest_sha256=sha256(a.manifest),
        strata=strata,records=rows,source_sha256=sha256(__file__),
        scope='Root-balanced within age stratum; exact same complete3s identity population at every .25s tick. No current-state truth exists in this cache; first error is at+.25s.'))
    print(json.dumps(strata))


if __name__=='__main__':main()
