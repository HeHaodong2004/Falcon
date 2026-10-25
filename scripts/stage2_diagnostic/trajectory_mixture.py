"""Cached action-identity removal: equal mixture of two valid path forecasts.

The mixture is NOT an independently trained graph-only model. It uses both
legal candidate paths and removes knowledge of which one is chosen. No future
observation enters its predictions. Own-minus-mixture proper scores test value
of conditioning beyond the simpler own-versus-wrong sensitivity diagnostic.
"""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from .common import OUT,save,extended_rates,digest
from .trajectory_metrics import loss_arrays,cm

def evaluate(path):
    r=json.loads(path.read_text());result={}
    with np.load(path.with_suffix('.npz')) as z:
        intersect=z['support_a']&z['support_b']
        ya=z['y_a'];yb=z['y_b']
        for mode,left,right in (('production_U','production_a','production_b'),('direct_I','direct_a','direct_b')):
            mask=np.ones(len(ya),bool) if mode.endswith('_U') else intersect
            pa=z[left][mask];pb=z[right][mask];a=ya[mask];b=yb[mask]
            mixture=.5*(pa.astype(np.float64)+pb.astype(np.float64))
            own=.5*(loss_arrays(a,pa).mean(0)+loss_arrays(b,pb).mean(0))
            marginal=.5*(loss_arrays(a,mixture).mean(0)+loss_arrays(b,mixture).mean(0))
            owncm=.5*(cm(a,pa.argmax(1))+cm(b,pb.argmax(1)))
            marginalcm=.5*(cm(a,mixture.argmax(1))+cm(b,mixture.argmax(1)))
            result[mode]=dict(own=own.tolist(),marginal=marginal.tolist(),gain=(marginal-own).tolist(),
                own_confusion=owncm.tolist(),marginal_confusion=marginalcm.tolist())
    return dict(root_id=r['root_id'],scene_id=r['scene_id'],modes=result)

def main():
    dest=OUT/'trajectory/mixture_v1';dest.mkdir(parents=True,exist_ok=True)
    protocol=dest/'protocol.json'
    if not protocol.exists():
        save(protocol,dict(stage='before mixture-control scoring; supplementary after primary own-v-cross audit',
            formula='m=0.5*(pA+pB), identical m for both recorded outcomes; compare0.5[L(m,yA)+L(m,yB)] vs0.5[L(pA,yA)+L(pB,yB)]',
            scope='same1705pose-selectedRootpairs, samefixedU ornaturalintersectionI, no newinference/no fitting/no GTinputtomix',
            distinction='trajectory-identity marginalized mixture of trained forecasts; NOT an independentlytrained graphonlybaseline',
            weights='onepairperRoot; RootmeanwithinScene, thenScenemean; paired2000sceneCIseed2525',source_sha256=digest(__file__)))
    elif json.loads(protocol.read_text())['source_sha256']!=digest(__file__):raise ValueError('mixture protocolsource changed')
    paths=sorted((OUT/'trajectory/primary_v1/roots').glob('*.json'))
    if len(paths)!=1705:raise ValueError('complete pairedcache required')
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(evaluate,paths,chunksize=4))
    cfg=json.loads((OUT/'protocol.json').read_text());validation=set(cfg['diagnostic_validation_scenes']);output={}
    for subset,rs in [('all',rows),('diagnostic_validation38',[r for r in rows if r['scene_id'] in validation])]:
        by=defaultdict(list)
        for r in rs:by[r['scene_id']].append(r)
        result={}
        for mode in rs[0]['modes']:
            own=np.stack([np.mean([r['modes'][mode]['own'] for r in root],axis=0) for _,root in sorted(by.items())])
            mix=np.stack([np.mean([r['modes'][mode]['marginal'] for r in root],axis=0) for _,root in sorted(by.items())])
            oc=np.stack([np.mean([r['modes'][mode]['own_confusion'] for r in root],axis=0) for _,root in sorted(by.items())])
            mc=np.stack([np.mean([r['modes'][mode]['marginal_confusion'] for r in root],axis=0) for _,root in sorted(by.items())])
            rng=np.random.default_rng(2525);gain=[];iou=[]
            for _ in range(2000):
                index=rng.integers(0,len(own),len(own));gain.append((mix-own)[index].mean(0))
                iou.append(extended_rates(oc[index].mean(0))['label_aware_delta_iou']-extended_rates(mc[index].mean(0))['label_aware_delta_iou'])
            result[mode]=dict(roots=len(rs),scenes=len(by),own_logloss=float(own.mean(0)[0]),mixture_logloss=float(mix.mean(0)[0]),
                own_brier=float(own.mean(0)[1]),mixture_brier=float(mix.mean(0)[1]),mixture_minus_own_logloss=float((mix-own).mean(0)[0]),
                mixture_minus_own_brier=float((mix-own).mean(0)[1]),ci95=np.quantile(gain,[.025,.975],axis=0).T.tolist(),
                own_iou=extended_rates(oc.mean(0))['label_aware_delta_iou'],mixture_iou=extended_rates(mc.mean(0))['label_aware_delta_iou'],
                own_minus_mixture_iou_ci95=np.quantile(iou,[.025,.975]).tolist())
        output[subset]=result
    save(dest/'summary.json',output);print(json.dumps(output),flush=True)

if __name__=='__main__':main()
