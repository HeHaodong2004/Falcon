"""Fixed original pilot evaluation of the matched auxiliary-training pair."""
import argparse
import torch
from .diagnose import OUT,read,save,sha
from .range_spatial_eval import evaluate
from .range_spatial_train import load_groups


def run(name='support256',device='cuda:1',epochs=4):
    prepared=load_groups('dev');records=[]
    for variant in ['control','teacher']:
        candidates=[]
        for epoch in range(1,epochs+1):
            cp=OUT/'training'/(name+'_'+variant)/('epoch%02d.pt'%epoch)
            evaluate(cp,device=device,_prepared=prepared)
            result=read(OUT/'pilot'/('%s_%s_dev_epoch%02d.json'%(name,variant,epoch)))
            rule=result['best_rule'];candidates.append(dict(epoch=epoch,rule=rule,metrics=result['rules'][rule],checkpoint=str(cp),checkpoint_sha256=sha(cp)))
        best=max(candidates,key=lambda r:r['metrics']['label_aware_delta_iou'])
        saved=torch.load(best['checkpoint'],map_location='cpu');saved['decision_rule']='argmax'if best['rule']=='argmax'else'threshold'
        if saved['decision_rule']=='threshold':saved['threshold']=float(best['rule'].split('_')[1])
        torch.save(saved,OUT/'training'/(name+'_'+variant)/'selected.pt')
        records.append(dict(variant=variant,best=best,candidates=candidates))
        print('BEST',variant,best,flush=True)
    save(OUT/(name+'_selection.json'),dict(role='Matched original fixed-pilot diagnostic; not full-development acceptance',variants=records))
    del prepared
    for variant in records:evaluate(variant['best']['checkpoint'],device=device,full_training=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='support256');p.add_argument('--device',default='cuda:1');p.add_argument('--epochs',type=int,default=4);run(**vars(p.parse_args()))
