import argparse
import torch
from .diagnose import OUT,read,save,sha
from .range_ray_eval import evaluate,prepare_groups


def run(name='range_ray256',device='cuda:0'):
    directory=OUT/'training'/name;cfg=read(directory/'config.json');records=[];prepared=prepare_groups()
    for epoch in range(4,cfg['epochs']+1,4):
        checkpoint=directory/('epoch%02d.pt'%epoch);evaluate(checkpoint,device=device,_prepared=prepared)
        result=read(OUT/'pilot'/('%s_dev_epoch%02d.json'%(name,epoch)));rule=result['best_rule']
        records.append(dict(epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),rule=rule,metrics=result['rules'][rule]))
    best=max(records,key=lambda r:r['metrics']['label_aware_delta_iou']);saved=torch.load(best['checkpoint'],map_location='cpu');saved['decision_rule']='argmax'if best['rule']=='argmax'else'threshold'
    if saved['decision_rule']=='threshold':saved['threshold']=float(best['rule'].split('_')[1])
    torch.save(saved,directory/'selected.pt');save(OUT/(name+'_selection.json'),dict(role='Original fixed pilot selection only, no full-development or confirmation selection',best=best,candidates=records));print('BEST',best,flush=True)
    del prepared;evaluate(best['checkpoint'],device=device,training=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='range_ray256');p.add_argument('--device',default='cuda:0');run(**vars(p.parse_args()))
