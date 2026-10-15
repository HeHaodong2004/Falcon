"""Measure motion support from exposed timestamped physical positions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.train import load_roots
from eventforecastwm.guard import Guard


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--protocol',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);guard=Guard(out.parent)
    roots=load_roots(a.manifest,read(a.protocol)['development'],guard);rows=[]
    for root in roots:
        guard.check();w=root['windows'][0];t=w['targets'];mask=t['track_valid'][:12].all(0)
        if not mask.any():continue
        p=t['track_positions'][:12,mask];velocity=(p[1:]-p[:-1])/.25;speed=velocity.norm(dim=-1)
        acceleration=((velocity[1:]-velocity[:-1])/.25).norm(dim=-1)
        rows.append(dict(root_id=root['root_id'],scene=root['scene'],agents=int(mask.sum()),
            speed_p95=float(torch.quantile(speed.flatten(),.95)),speed_max=float(speed.max()),
            speed_gt3_fraction=float((speed>3.).float().mean()),acceleration_p95=float(torch.quantile(acceleration.flatten(),.95)),
            acceleration_gt2_fraction=float((acceleration>2.).float().mean())))
    metrics={k:float(np.mean([r[k] for r in rows])) for k in ('speed_p95','speed_max','speed_gt3_fraction','acceleration_p95','acceleration_gt2_fraction')}
    save_json(out,dict(roots=len(rows),metrics=metrics,records=rows,manifest_sha256=sha256(a.manifest),
        definition='Root-mean statistics on the exact147-root complete3s identity population, canonical.25s position differences; no rigid-body velocity fields'))
    print(json.dumps(metrics))


if __name__=='__main__':main()
