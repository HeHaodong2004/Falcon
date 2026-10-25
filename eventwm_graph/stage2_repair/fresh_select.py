import argparse
from pathlib import Path
import torch
from .diagnose import OUT,read,save,sha
from .range_spatial_eval import evaluate


def run(name='range_fresh64',device='cuda:1'):
    directory=OUT/'training'/name;config=read(directory/'config.json');records=[]
    for epoch in range(4,config['epochs']+1,4):
        checkpoint=directory/('epoch%02d.pt'%epoch);path=OUT/'pilot'/('%s_dev_epoch%02d.json'%(name,epoch))
        if not path.exists():evaluate(checkpoint,device=device)
        result=read(path);rule=result['best_rule'];records.append(dict(epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),rule=rule,metrics=result['rules'][rule]))
    best=max(records,key=lambda r:r['metrics']['label_aware_delta_iou']);save(OUT/(name+'_selection.json'),dict(role='Original fixed pilot selection, not full-development acceptance or confirmation.',best=best,candidates=records));saved=torch.load(best['checkpoint'],map_location='cpu');saved['decision_rule']='argmax'if best['rule']=='argmax'else'threshold'
    if saved['decision_rule']=='threshold':saved['threshold']=float(best['rule'].split('_')[1])
    torch.save(saved,directory/'selected.pt');print('SELECTION',best,flush=True);evaluate(best['checkpoint'],device=device,full_training=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='range_fresh64');p.add_argument('--device',default='cuda:1');run(**vars(p.parse_args()))
