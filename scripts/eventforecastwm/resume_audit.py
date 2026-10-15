"""Bounded end-to-end recovery check through the actual GPU training loop."""
import argparse
import json
from pathlib import Path
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm import train
from eventforecastwm.prepared_cache import prepare


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();out=Path(a.output)
    if (out/'result.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    parent=read(a.protocol)
    diagnostic=dict(parent,development=parent['development'][:8],subsets={'32':parent['subsets']['128'][:32]},
        purpose='recovery engineering test only; no model selection and no replacement evaluation membership')
    save_json(out/'protocol.json',diagnostic)
    base=dict(protocol=str(out/'protocol.json'),manifest=a.manifest,scale='32',device=a.device,seed=9917,
        model=dict(kind='stochastic',width=128,modes=6,noise='integrated'),steps=32,batch_size=8,
        lr=1e-4,weight_decay=.01,schedule='constant',eval_every=32,objective='distribution',nll_weight=.05)
    old_prepare=train.prepare_roots;old_loss=train.loss
    train.prepare_roots=lambda model,roots,guard=None:prepare(model,roots,a.manifest,out/'prepared',old_prepare,guard)
    configs={}
    for name in ('continuous','resumed'):
        config=dict(base,run=str(out/name));path=out/(name+'.json');save_json(path,config);configs[name]=path
    train.train(configs['continuous'])
    count=0
    def stopping_loss(*args,**kwargs):
        nonlocal count
        result=old_loss(*args,**kwargs);count+=1
        if count==8:(out/'resumed/STOP_RESEARCH').write_text('Intentional recovery test after eighth optimizer step\n')
        return result
    train.loss=stopping_loss;train.train(configs['resumed']);train.loss=old_loss
    interrupted=torch.load(out/'resumed/last.pt',map_location='cpu')
    if interrupted['step']!=8:raise AssertionError('Recovery test did not stop at requested boundary')
    (out/'resumed/STOP_RESEARCH').unlink();train.train(configs['resumed'])
    left=torch.load(out/'continuous/last.pt',map_location='cpu');right=torch.load(out/'resumed/last.pt',map_location='cpu')
    fields=('model','optimizer','sampler_rng','torch_rng','cuda_rng','order','cursor','step','presentations','seen','best','best_step')
    matches={k:equal(left[k],right[k]) for k in fields}
    save_json(out/'result.json',dict(bitwise_equal=all(matches.values()),fields=matches,interrupted_step=8,final_step=32,
        device=a.device,training_roots=32,diagnostic_development_roots=8,parent_protocol_sha256=sha256(a.protocol),
        source_sha256=sha256(__file__),continuous_sha256=sha256(out/'continuous/last.pt'),
        resumed_sha256=sha256(out/'resumed/last.pt'),
        scope='Same-machine bounded actual trainer test; elapsed time/run identities differ; no cross-hardware determinism claim'))
    if not all(matches.values()):raise AssertionError(matches)
    print(json.dumps(matches))


if __name__=='__main__':main()
