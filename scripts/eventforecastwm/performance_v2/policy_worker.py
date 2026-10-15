"""Explicit runtime-only guard replacement for authorized v2 research jobs."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time

from eventbeliefwm.data import sha256
from eventbeliefwm.experiment import save_json
from . import resource_policy


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--module',required=True);parser.add_argument('--run',required=True)
    args,remaining=parser.parse_known_args()
    if remaining[:1]==['--']:remaining=remaining[1:]
    out=Path(args.run)
    from eventforecastwm import guard
    guard.Guard=resource_policy.AdaptiveGuard
    # No physical/training source or checkpoint identity is rewritten.
    runtime=dict(timestamp=time.time(),policy=resource_policy.VERSION,module=args.module,
        policy_sha256=sha256(resource_policy.__file__),worker_sha256=sha256(__file__),
        frozen_legacy_guard_sha256=sha256(guard.__file__),optimizer_semantics_changed=False)
    save_json(out/'runtime_policy.json',runtime)
    with (out/'runtime_policy_history.jsonl').open('a') as log:log.write(json.dumps(runtime)+'\n')
    if args.module=='eventforecastwm.performance_v2.confirmation_population':
        from eventforecastwm import confirmation
        confirmation.recovery=resource_policy.extraction_recovery
    if args.module.startswith('eventforecastwm.performance_v2.train'):
        import torch
        original=torch.load;last=(out/'last.pt').resolve()
        def load(path,*positional,**keywords):
            value=original(path,*positional,**keywords)
            if isinstance(path,(str,Path)) and Path(path).resolve()==last:
                save_json(out/'resume_load_receipt.json',dict(runtime=runtime,checkpoint_sha256=sha256(last),
                    step=value['step'],presentations=value['presentations'],identity=value['identity'],
                    checkpoint_fields_loaded=['model','optimizer','sampler_rng','torch_rng','cuda_rng','order','cursor',
                                     'step','presentations','seen','best','best_step','elapsed'],
                    scheduler='Unchanged deterministic function of saved optimizer step and frozen config'))
            return value
        torch.load=load
    sys.argv=[args.module,*remaining];runpy.run_module(args.module,run_name='__main__')


if __name__=='__main__':main()
