"""Evaluate each completed online epoch on the fixed original666-window pilot.

Each chosen operating point gets an immutable deployment copy. This is pilot
selection, never evidence that original full-development acceptance is met.
"""
import argparse
from pathlib import Path
import time
import torch
from .diagnose import OUT,read,save,sha
from .range_spatial_eval import evaluate
from .range_spatial_train import load_groups


def run(names,epochs=4,device='cuda:1',watch=False,output_name='online_selections'):
    if not output_name or Path(output_name).name!=output_name:raise ValueError('selection output must be a simple filename stem')
    prepared=load_groups('dev',OUT/'range_evidence_dev_v2/manifest.json');results={n:[]for n in names};done=set();tick=time.perf_counter();destination=OUT/(output_name+'.json')
    while len(done)<len(names)*epochs:
        progress=False
        for name in names:
            for epoch in range(1,epochs+1):
                if(name,epoch)in done:continue
                cp=OUT/'training'/name/('epoch%02d.pt'%epoch)
                if not cp.exists()or time.time()-cp.stat().st_mtime<10:continue
                # Epoch files are written once. Wait for a complete archive;
                # never infer a completed checkpoint from the learning log.
                try:saved=torch.load(cp,map_location='cpu')
                except (EOFError,RuntimeError):continue
                expected=str((OUT/'range_evidence_dev_v2/manifest.json').resolve())
                if str(Path(saved['config']['dev_feature_manifest']).resolve())!=expected:raise ValueError('unexpected dev feature schema')
                evaluate(cp,device=device,_prepared=prepared)
                score=read(OUT/'pilot'/('%s_dev_epoch%02d.json'%(name,epoch)));rule=score['best_rule'];record=dict(epoch=epoch,rule=rule,metrics=score['rules'][rule],checkpoint=str(cp.resolve()),checkpoint_sha256=sha(cp));results[name].append(record);done.add((name,epoch));progress=True
                best=max(results[name],key=lambda r:r['metrics']['label_aware_delta_iou'])
                if best is record:
                    saved['decision_rule']='argmax'if rule=='argmax'else'threshold'
                    if saved['decision_rule']=='threshold':saved['threshold']=float(rule.split('_')[1])
                    archive=cp.parent/('selected_epoch%02d_%s.pt'%(epoch,rule));torch.save(saved,archive);record['deployment_checkpoint']=str(archive.resolve());record['deployment_sha256']=sha(archive);torch.save(saved,cp.parent/'selected.pt')
                save(destination,dict(role='Fixed666-window pilot selection only; operating point frozen per immutable deployment archive. No confirmation use, no full-development acceptance claim.',names=results,best={n:max(r,key=lambda q:q['metrics']['label_aware_delta_iou'])for n,r in results.items()if r},seconds=time.perf_counter()-tick));print('SELECTED',name,best,flush=True)
        if not progress:
            if not watch:break
            time.sleep(30)
    print('SELECTION_COMPLETE',len(done),'/',len(names)*epochs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',required=True);p.add_argument('--epochs',type=int,default=4);p.add_argument('--device',default='cuda:1');p.add_argument('--watch',action='store_true');p.add_argument('--output-name',default='online_selections');run(**vars(p.parse_args()))
