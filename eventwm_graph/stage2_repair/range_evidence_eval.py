"""Fixed pilot and same-training-example reconstruction for range fusion."""
import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from .diagnose import OUT,HF,read,save,sha,source_rows,member
from .range_history import ancestor_frames
from .range_gpu import GpuObservedRangeMesh
from .range_evidence import features,EvidenceNet
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment
from eventwm_graph.stage2_hf.spatial import build
from eventwm_graph.stage2_hf.spatial_eval import score
from eventwm_graph.stage2_hf.light_eval import aggregate
from scripts.stage2_diagnostic.common import extended_rates


def prepare(device='cuda:0',training=False):
    torch.set_num_threads(1);tick=time.perf_counter();raw_rows,source_path=source_rows('train'if training else'dev');raw={r['root_id']:r for r in raw_rows}
    packed=read(HF/'packed'/('pilot_fit'if training else'dev_pilot')/'manifest.json');oldrows={}if training else{r['root_id']:r for r in read(HF/'light_eval_cache/manifest.json')['rows']}
    selected=packed['rows']
    if training:
        trained=read(OUT/'range_evidence_train/manifest.json');ids={r['root_id']for r in trained['records'][:4]};selected=[r for r in selected if r['root_id']in ids]
    dest=OUT/('range_evidence_train_fullcells'if training else'range_evidence_dev');dest.mkdir(exist_ok=True);records=[]
    names=['range_evidence.py','range_gpu.py','range_history.py','observed_range_mesh.py','geometry_cuda.py','legal_camera_features.py','range_evidence_eval.py']
    for row in selected:
        rid=row['root_id'];path=dest/(rid+'.npz');receipt=path.with_suffix('.json');sensor_path=OUT/('train_range_branches'if training else'range_branches')/(rid+'.npz')
        identity=dict(schema='legal_range_evidence16_1',packed_sha256=row['sha256'],sensor_sha256=sha(sensor_path),sources={n:sha(Path(__file__).with_name(n))for n in names})
        if receipt.exists():
            prior=read(receipt)
            if prior['identity']!=identity or prior['sha256']!=sha(path):raise ValueError('range evidence evaluation cache invalidated')
            records.append(prior);continue
        legal=member(raw[rid],'legal_root.json.gz');nodes=member(raw[rid],'future_observations.json.gz');all_tracks=[o['tracks']for o in legal['history']]+[n['observation']['tracks']for n in nodes]
        with np.load(sensor_path)as z:sensors={k:z[k]for k in z.files}
        a=load_arrays(row);old=oldrows.get(rid);lookup={(e['start_node'],e['end_node']):e for e in old['segments']}if old else{}
        if old:
            with np.load(old['path'])as z:oldx=z['x'];oldy=z['y']
        meshes={};arrays=defaultdict(list);segments=[];cursor=0
        if training:
            from .global_features import GlobalTimePredictor
            from eventwm_graph.stage2_segment.fused import transform
            from eventwm_graph.stage2_segment.data import replay
            predictor=GlobalTimePredictor(OUT/'training/point_residual_full/epoch06.pt',device=device,increment_format='columns')
            examples={(e['start_node'],e['end_node']):e for e in replay(raw[rid])if'audit'not in e}
        for i,meta in enumerate(row['segments']):
            v=get_segment(a,i);p=v[3];res=float(v[-1][0]);sid=meta['start_node'];expected=build(*v[:4],v[-1],make_query=False)['keys']
            if training:
                ex=examples[sid,meta['end_node']];ctx=predictor.feature_class(ex['graph'],p,predictor.prepare_graph(ex['graph']));x,_=transform(ctx,expected,cuda=predictor.cuda,device_output=True);x=x.cpu().numpy();keys=expected;target={tuple(k):int(y)for k,y in zip(v[4],v[5])};y=np.array([target.get(tuple(k),0)for k in keys],np.uint8)
            else:
                e=lookup[sid,meta['end_node']];sl=slice(e['start'],e['end']);x=oldx[sl];y=oldy[sl];keys=np.floor((x[:,20:22].astype(np.float64)*10+p[0,[1,3]])/res).astype(np.int32);np.testing.assert_array_equal(keys,expected)
            assert np.count_nonzero(y)==len(v[4])
            if sid not in meshes:
                take=ancestor_frames(sensors['parents'],sensors['history_count'],sid);meshes[sid]=GpuObservedRangeMesh(sensors['depth'][take],sensors['origins'][take],sensors['rotations'][take],sensors['timestamps'][take],p[0,0],res,device,tracks=[all_tracks[j]for j in take])
            current=legal['history'][-1]if sid is None else nodes[sid]['observation'];extra=features(meshes[sid],keys,x,p,v[0],current['camera_origin'],current['camera_rotation']);arrays['extra'].append(extra);arrays['y'].append(y)
            if training:arrays['x'].append(x)
            segments.append(dict(start_node=sid,end_node=meta['end_node'],start=cursor,end=cursor+len(y),poses=len(p),positive=int(np.count_nonzero(y))));cursor+=len(y)
        temp=path.with_suffix('.tmp.npz');np.savez(temp,**{k:np.concatenate(v)for k,v in arrays.items()});temp.replace(path);record=dict(root_id=rid,scene_id=row['scene_id'],path=str(path),sha256=sha(path),identity=identity,segments=segments,original_feature_path=old['path']if old else None);save(receipt,record);records.append(record);print('PREPARED',len(records),rid,'windows',len(segments),'seconds',time.perf_counter()-tick,flush=True)
    save(dest/'manifest.json',dict(role='SAME TRAINING EXAMPLES, not generalization'if training else'Original fixed 31-Root/16-scene/666-window development pilot; not full development',source_manifest_sha256=sha(source_path),records=records,windows=sum(len(r['segments'])for r in records),seconds=time.perf_counter()-tick))


def evaluate(checkpoint,device='cuda:0',training=False):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;tick=time.perf_counter();saved=torch.load(checkpoint,map_location='cpu');model=EvidenceNet().to(device).eval();model.load_state_dict(saved['model']);dest=OUT/('range_evidence_train_fullcells'if training else'range_evidence_dev');m=read(dest/'manifest.json');records=[];pooled=np.zeros((14,3,3),np.int64)
    for row in m['records']:
        with np.load(row['path'])as z:extra=z['extra'];y=z['y'];x=z['x']if training else None
        if not training:
            with np.load(row['original_feature_path'])as z:x=z['x']
        cms=[];scores=[];empty=np.zeros(14,int);zero=0
        for e in row['segments']:
            sl=slice(e['start'],e['end']);gt=y[sl];prob=[]
            with torch.inference_mode():
                for begin in range(sl.start,sl.stop,65536):
                    end=min(begin+65536,sl.stop);p,_=model.probabilities(torch.from_numpy(x[begin:end]).to(device),torch.from_numpy(extra[begin:end]).to(device));prob.append(p.cpu().numpy())
            p=np.concatenate(prob)
            if e['poses']==1:p.fill(0);p[:,0]=1
            cm,sc,ee=score(p,gt);cms.append(cm);scores.append(sc);pooled+=cm
            if not np.any(gt):zero+=1;empty+=ee
        records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],cm=np.mean(cms,axis=0),scores=np.mean(scores,axis=0),empty=empty,zero_cases=zero))
    result=aggregate(records);best=max(result['rules'],key=lambda k:result['rules'][k]['label_aware_delta_iou']);result.update(best_rule=best,windows=m['windows'],role=m['role'],checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),pooled={k:extended_rates(c)for k,c in zip(result['rules'],pooled)},pooled_confusion=pooled.tolist(),seconds=time.perf_counter()-tick);path=OUT/'pilot'/('range_evidence_%s_epoch%02d.json'%('training'if training else'dev',saved['epoch']));save(path,result);print('RESULT',best,result['rules'][best]['label_aware_delta_iou'],result['rules'][best]['revelation_iou'],result['role'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--training',action='store_true');p.add_argument('--checkpoint');args=vars(p.parse_args());cp=args.pop('checkpoint');evaluate(cp,**args)if cp else prepare(**args)
