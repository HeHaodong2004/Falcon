"""Replay only the deterministic root/window sampler of a completed run."""
import argparse
from pathlib import Path
import numpy as np
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json


def replay(state,config,ids,counts,multiplicities):
    generator=np.random.default_rng(config['seed']);order=generator.permutation(len(ids)).tolist();cursor=0
    visits=[np.zeros(counts[rid],dtype=np.int64) for rid in ids]
    slots=[np.repeat(np.arange(counts[rid]),multiplicities.get(rid,[1]*counts[rid])) for rid in ids]
    for _ in range(state['step']):
        if cursor>=len(order):order=generator.permutation(len(ids)).tolist();cursor=0
        indices=order[cursor:cursor+config.get('batch_size',32)];cursor+=len(indices)
        for index in indices:
            window=slots[index][int(generator.integers(len(slots[index])))];visits[index][window]+=1
    presentations=int(sum(v.sum() for v in visits))
    checks=dict(presentations=presentations==state['presentations'],cursor=cursor==state['cursor'],
        order=order==state['order'],sampler_rng=generator.bit_generator.state==state['sampler_rng'],
        seen=set(state['seen'])=={rid for rid,v in zip(ids,visits) if v.any()})
    if not all(checks.values()):raise AssertionError(checks)
    flattened=np.concatenate(visits)
    return dict(step=state['step'],presentations=presentations,roots_seen=sum(bool(v.any()) for v in visits),
        fitting_roots=len(ids),fitting_windows=int(sum(counts[rid] for rid in ids)),
        windows_presented=int((flattened>0).sum()),minimum_window_presentations=int(flattened.min()),
        maximum_window_presentations=int(flattened.max()),mean_window_presentations=float(flattened.mean()),
        all_windows_presented=bool((flattened>0).all()),exact_replay_checks=checks,
        root_window_presentations={rid:v.tolist() for rid,v in zip(ids,visits)})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('Completed sampler audit is immutable')
    config=read(args.config);run=Path(config['run']);receipt=read(run/'result.json')
    if receipt['status']!='training_complete':raise ValueError('A completed training receipt is required')
    if any(sha256(run/name)!=receipt[key] for name,key in [('best.pt','best_sha256'),('last.pt','last_sha256')]):
        raise ValueError('Training checkpoint bytes changed')
    protocol=read(config['protocol']);ids=protocol['subsets'][str(config['scale'])]
    counts={r['root_id']:r['windows'] for r in read(config['manifest'])['roots']}
    path=run/'sampling.json';multiplicities={r['root_id']:r['multiplicities'] for r in read(path)['roots']} if path.exists() else {}
    result={name:replay(torch.load(run/(name+'.pt'),map_location='cpu'),config,ids,counts,multiplicities) for name in ('last','best')}
    out.parent.mkdir(parents=True,exist_ok=True)
    save_json(out,dict(status='exposure_audit_complete',config_sha256=sha256(args.config),
        training_receipt_sha256=sha256(run/'result.json'),source_sha256=sha256(__file__),checkpoints=result,
        independent_unit='root; windows and repeated robot queries are dependent'))


if __name__=='__main__':main()
