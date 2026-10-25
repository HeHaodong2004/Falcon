"""Full fixed-pilot reconstruction for the shared current-map model."""
import argparse
import time
import numpy as np
import torch
from .diagnose import OUT,HF,read,save,sha
from .shared_map import SharedMapNet,map_canvas
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment
from eventwm_graph.stage2_hf.spatial_eval import score
from eventwm_graph.stage2_hf.light_eval import aggregate
from scripts.stage2_diagnostic.common import extended_rates


def evaluate(checkpoint,device='cuda:0',name=None):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;tick=time.perf_counter()
    saved=torch.load(checkpoint,map_location='cpu');config=saved.get('model_config',{});fine=config.get('fine',False);model=SharedMapNet(**config).to(device).eval();model.load_state_dict(saved['model']);m=read(HF/'packed/dev_pilot/manifest.json');oldrows={r['root_id']:r for r in read(HF/'light_eval_cache/manifest.json')['rows']}
    records=[];pooled=np.zeros((14,3,3),np.int64);windows=0
    for row in m['rows']:
        a=load_arrays(row);tr=read(HF/'temporal/dev_pilot'/(row['root_id']+'.json'));assert tr['packed_sha256']==row['sha256']
        with np.load(tr['path'],allow_pickle=False)as z:last=z['last']
        old=oldrows[row['root_id']];lookup={(e['start_node'],e['end_node']):e for e in old['segments']}
        with np.load(old['path'],allow_pickle=False)as z:x=z['x'];y=z['y']
        cache={};cms=[];scores=[];empty=np.zeros(14,int);zero=0
        for i,meta in enumerate(row['segments']):
            v=get_segment(a,i);gi=int(a['graph_index'][i]);p=v[3];e=lookup[(meta['start_node'],meta['end_node'])];sl=slice(e['start'],e['end'])
            if gi not in cache:
                gs=slice(*a['goffsets'][gi:gi+2]);d=map_canvas(v[0],v[1],v[2],last[gs],p[0,0],v[-1],p[0,1:4],anchor=fine,halo=128 if fine else 96)
                with torch.inference_mode():enc=model.encode(torch.from_numpy(d['maps'][None]).to(device),torch.tensor([d['maps'].shape[1:]],device=device),torch.from_numpy(d['fine'][None]).to(device)if fine else None)
                cache[gi]=(enc,d['lo'])
            enc,lo=cache[gi];keys=np.floor((x[sl,20:22].astype(np.float64)*10+p[0,[1,3]])/v[-1][0]).astype(np.int32)
            from eventwm_graph.stage2_hf.spatial import build
            expected=build(*v[:4],v[-1],make_query=False)['keys'];np.testing.assert_array_equal(keys,expected)
            gt=y[sl];assert np.count_nonzero(gt)==len(v[4]);pp=[]
            with torch.inference_mode():
                for start in range(sl.start,sl.stop,65536):
                    end=min(start+65536,sl.stop);xt=torch.from_numpy(x[start:end]).to(device);ii=torch.from_numpy((keys[start-sl.start:end-sl.start]-lo).astype(np.int64)).to(device)
                    prob,_=model.probabilities(xt,ii,torch.zeros(len(xt),device=device,dtype=torch.long),enc);pp.append(prob.cpu().numpy())
            prob=np.concatenate(pp)
            if len(p)==1:prob.fill(0);prob[:,0]=1
            cm,sc,ee=score(prob,gt);cms.append(cm);scores.append(sc);pooled+=cm;windows+=1
            if not np.any(gt):zero+=1;empty+=ee
        records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],cm=np.mean(cms,axis=0),scores=np.mean(scores,axis=0),empty=empty,zero_cases=zero))
    r=aggregate(records);best=max(r['rules'],key=lambda k:r['rules'][k]['label_aware_delta_iou']);r.update(best_rule=best,windows=windows,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),pooled={k:extended_rates(cm)for k,cm in zip(r['rules'],pooled)},pooled_confusion=pooled.tolist(),seconds=time.perf_counter()-tick)
    save(OUT/'pilot'/((name or saved['config']['name']+'_epoch%02d'%saved['epoch'])+'.json'),r)
    print(name,best,r['rules'][best]['label_aware_delta_iou'],r['rules'][best]['revelation_iou'],'seconds',r['seconds'],flush=True)
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--name');evaluate(**vars(p.parse_args()))
