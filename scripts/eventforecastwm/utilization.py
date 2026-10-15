"""Replay only the frozen root/window sampler to audit actual data utilization.

Checks exact final RNG state, order and cursor against a checkpoint. It does
not rerun training or read raw trajectories. Window counts come from the
immutable compact manifest; windows within a root remain correlated.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json


def audit(checkpoint):
    digest=sha256(checkpoint);state=torch.load(checkpoint,map_location='cpu');cfg=state['identity']['config']
    if sha256(cfg['manifest'])!=state['identity']['manifest_sha256'] or sha256(cfg['protocol'])!=state['identity']['protocol_sha256']:
        raise ValueError('Frozen sampler inputs changed')
    ids=read(cfg['protocol'])['subsets'][str(cfg['scale'])]
    entries={r['root_id']:r for r in read(cfg['manifest'])['roots']}
    visits=[np.zeros(entries[r]['windows'],dtype=np.int64) for r in ids]
    rng=np.random.default_rng(cfg['seed']);order=rng.permutation(len(ids)).tolist();cursor=0;presentations=0
    for _ in range(state['step']):
        if cursor>=len(order):order=rng.permutation(len(ids)).tolist();cursor=0
        indices=order[cursor:cursor+cfg.get('batch_size',32)];cursor+=len(indices);presentations+=len(indices)
        for i in indices:visits[i][int(rng.integers(len(visits[i])))]+=1
    if (rng.bit_generator.state!=state['sampler_rng'] or order!=state['order'] or cursor!=state['cursor'] or
        presentations!=state['presentations']):raise ValueError('Sampler replay does not match checkpoint')
    seen=sorted(ids[i] for i,v in enumerate(visits) if v.sum())
    if seen!=state['seen']:raise ValueError('Root visitation differs')
    available=sum(len(v) for v in visits);visited=sum(int((v>0).sum()) for v in visits)
    if sha256(checkpoint)!=digest:raise ValueError('Checkpoint changed during audit; use a frozen or completed checkpoint')
    return dict(checkpoint_sha256=digest,source_sha256=sha256(__file__),numpy_version=np.__version__,optimizer_steps=state['step'],root_presentations=presentations,
        fitting_roots=len(ids),visited_roots=len(seen),available_windows=available,visited_windows=visited,
        window_fraction=visited/available,effective_root_epochs=presentations/len(ids),sampler_replay_exact=True,
        root_records=[dict(root_id=r,scene=entries[r]['scene'],presentations=int(v.sum()),window_visits=v.tolist()) for r,v in zip(ids,visits)],
        meaning='Exact sampled windows, including windows with no valid tracked-motion target. These are not independent physical trajectories.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    result=audit(a.checkpoint);save_json(out,result)
    print({k:v for k,v in result.items() if k!='root_records'})


if __name__=='__main__':main()
