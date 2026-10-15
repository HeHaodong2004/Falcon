"""Matched masked metrics and movement diagnostics; no planning claims."""
from collections import defaultdict
import numpy as np
from ..sr_metrics import metrics, ranking_metrics


def root_interval(values, seed=20261009, replicates=200):
    values = np.asarray(values,dtype=float)
    if not len(values):return dict(roots=0,mean=None,ci95=None)
    rng = np.random.RandomState(seed)
    means = [values[rng.randint(0,len(values),len(values))].mean() for _ in range(replicates)]
    return dict(roots=len(values),mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
                bootstrap_replicates=replicates,seed=seed)


def outcome_metrics(y, p, metadata, train_prior=None):
    y,p = np.asarray(y,dtype=int),np.asarray(p,dtype=float)
    if p.shape != (len(y),3) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1),1,atol=1e-6):
        raise ValueError('Expected normalized three-class probabilities')
    if not len(y) or np.any((y < 0)|(y > 2)):raise ValueError('Need valid categorical outcomes')
    nll = -np.log(np.clip(p[np.arange(len(y)),y],1e-7,1)).mean()
    confusion = np.zeros((3,3),dtype=int)
    np.add.at(confusion,(y,p.argmax(1)),1)
    classwise,f1 = [],[]
    for c in range(3):
        target = (y == c).astype(int)
        ap,roc = ranking_metrics(target,p[:,c])
        bins = np.minimum((p[:,c]*15).astype(int),14)
        reliability = [dict(bin=k,count=int((bins == k).sum()),
            mean_probability=float(p[bins == k,c].mean()),frequency=float(target[bins == k].mean()))
            for k in range(15) if (bins == k).any()]
        ece = sum(v['count']/len(y)*abs(v['mean_probability']-v['frequency']) for v in reliability)
        denom = confusion[c].sum()+confusion[:,c].sum()
        f1.append(2*confusion[c,c]/denom if denom else 0.)
        classwise.append(dict(class_id=c,prevalence=float(target.mean()),mean_probability=float(p[:,c].mean()),
            ap=ap,roc_auc=roc,brier=float(np.mean((p[:,c]-target)**2)),ece=float(ece),reliability=reliability))
    out = dict(samples=len(y),nll=float(nll),classwise=classwise,confusion_matrix=confusion.tolist(),
               confusion_axes='rows true, columns predicted; classes 0,1,2',macro_f1=float(np.mean(f1)),
               multiclass_brier=float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))),
               physical_collision=metrics((y != 2).astype(int),1-p[:,2]))
    if train_prior is not None:
        out['constant_train_prior_nll'] = float(-np.log(np.clip(np.asarray(train_prior)[y],1e-7,1)).mean())
    losses = -np.log(np.clip(p[np.arange(len(y)),y],1e-7,1))
    roots = defaultdict(list)
    for loss,m in zip(losses,metadata):roots[m['root_id']].append(float(loss))
    out['root_mean_nll'] = root_interval([np.mean(v) for v in roots.values()])
    return out


def outcome_ranking(metadata, y, p, duration, movement):
    """Full banks and same-path timing banks, with root-cluster uncertainty."""
    y,p,duration,movement = map(np.asarray,(y,p,duration,movement))
    groups = defaultdict(list)
    for i,m in enumerate(metadata):groups[m.get('group_id',m['root_id'])].append(i)
    eligible = {}
    for key,ix in groups.items():
        max_move = max(movement[ix])
        ix = [i for i in ix if movement[i] > .1 and movement[i] >= .95*max_move]
        if len(ix) >= 2:eligible[key] = ix
    report = dict(groups=len(eligible),roots=len({metadata[ix[0]]['root_id'] for ix in eligible.values()}))
    for name,good,bad,score in [('physical_collision',y != 2,y == 2,1-p[:,2]),
        ('social_success',y == 0,y == 1,p[:,0]/np.clip(p[:,0]+p[:,1],1e-7,None)),
        ('safe_success',y == 0,y != 0,p[:,0])]:
        byroot = defaultdict(list);pairs=0;informative=0;total=0.
        for ix in eligible.values():
            results = [float(score[a] > score[b])+.5*float(score[a] == score[b])
                       for a in ix for b in ix if good[a] and bad[b]]
            if results:
                informative += 1;pairs += len(results);total += sum(results)
                byroot[metadata[ix[0]]['root_id']].extend(results)
        report[name] = dict(informative_groups=informative,pairs=pairs,
            pair_accuracy=total/pairs if pairs else None,
            root_mean_accuracy=root_interval([np.mean(v) for v in byroot.values()]))
    selectors_report = {}
    for matched in (False,True):
        banks = eligible
        if not matched:
            banks = defaultdict(list)
            for i,m in enumerate(metadata):banks[m['root_id']].append(i)
        methods = {}
        for name,score in [('minimum_physical_risk',p[:,2]),('maximum_safe_success',-p[:,0]),('fastest',duration)]:
            selected = [min(ix,key=lambda i:(score[i],duration[i],metadata[i]['query_id'])) for ix in banks.values()]
            root_values = defaultdict(list)
            for i in selected:root_values[metadata[i]['root_id']].append(float(y[i] == 2))
            methods[name] = dict(groups=len(selected),class_fractions=[float(np.mean(y[selected] == c)) for c in range(3)] if selected else None,
                mean_duration_s=float(duration[selected].mean()) if selected else None,
                mean_movement_m=float(movement[selected].mean()) if selected else None,
                root_collision_rate=root_interval([np.mean(v) for v in root_values.values()]),
                selected_query_ids=[metadata[i]['query_id'] for i in selected])
        selectors_report['movement_matched' if matched else 'same_root_bank'] = methods
    report['selectors'] = selectors_report
    report['interpretation'] = 'Saved fixed-motion query outcomes; no closed-loop completion objective or collision constraint guarantee.'
    return report


def sr_metrics(y, p, metadata, train_prior):
    y, p = np.asarray(y), np.asarray(p)
    result = metrics(y, p)
    result['constant_train_prior_bce'] = metrics(y, np.full(len(y), train_prior))['bce']
    result['collision_recall_at_threshold'] = {str(t):float(np.mean((1-p[y == 0]) >= t)) if (y == 0).any() else None for t in (.05,.1,.2,.5)}
    groups = defaultdict(list)
    for i, row in enumerate(metadata):
        groups[row.get('group_id', row['root_id'])].append(i)
    wins, pairs = 0., 0
    for indices in groups.values():
        for a in indices:
            for b in indices:
                if y[a] == 1 and y[b] == 0:
                    wins += float(p[a] > p[b])+.5*float(p[a] == p[b])
                    pairs += 1
    result['same_root_same_path_ranking'] = dict(pairs=pairs, accuracy=wins/pairs if pairs else None)
    return result


def occupancy_metrics(target, probability, mask, horizons=(.25,.5,1,2,3)):
    result = []
    for h, time in enumerate(horizons):
        # Negating uint8 targets wraps 1 to 255 and silently corrupts BCE.
        y = target[mask[:, h], h].reshape(-1).astype(np.float64)
        p = probability[mask[:, h], h].reshape(-1).astype(np.float64)
        if not len(y):
            result.append(dict(horizon_s=time, available=0))
            continue
        ap, _ = ranking_metrics(y.astype(np.int64), p)
        clipped = np.clip(p, 1e-7, 1-1e-7)
        bins = np.minimum((p*15).astype(int), 14)
        ece = sum(np.mean(bins == k)*abs(p[bins == k].mean()-y[bins == k].mean()) for k in range(15) if (bins == k).any())
        result.append(dict(horizon_s=time, available=int(mask[:, h].sum()), ap=ap, prevalence=float(y.mean()),
            mean_probability=float(p.mean()), bce=float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped))),
            brier=float(np.mean((p-y)**2)), ece=float(ece)))
    return result


def selectors(metadata, y, p_safe, exposure, duration, movement, failure_horizon=None):
    """Same observed candidate bank. Heuristic exposure never becomes probability.

    Movement matched: choose within each root/path, excluding stationary
    candidates and requiring >=95% maximum candidate arclength. This diagnostic
    changes timing, not path length. The episode goal is unavailable in caches.
    """
    y, p_safe = np.asarray(y), np.asarray(p_safe)
    duration, movement, exposure = map(np.asarray, (duration, movement, exposure))
    report = {}
    for movement_matched in (False, True):
        banks = defaultdict(list)
        for i, row in enumerate(metadata):
            banks[row.get('group_id',row['root_id']) if movement_matched else row['root_id']].append(i)
        eligible = {}
        for key, ix in banks.items():
            if movement_matched:
                maximum = max(movement[ix])
                ix = [j for j in ix if movement[j] > .1 and movement[j] >= .95*maximum]
            if len(ix) >= 2:
                eligible[key] = ix
        methods = {}
        for name, score in [('sr_min_risk', 1-p_safe), ('dynamics_exposure', exposure), ('fastest', duration)]:
            selected = [min(ix, key=lambda j:(score[j], duration[j], metadata[j]['query_id'])) for ix in eligible.values()]
            methods[name] = dict(groups=len(selected), collision_rate=float(np.mean(y[selected] == 0)) if selected else None,
                mean_duration_s=float(np.mean(duration[selected])) if selected else None,
                mean_movement_m=float(np.mean(movement[selected])) if selected else None,
                selected_query_ids=[metadata[j]['query_id'] for j in selected],
                failure_penalized_query_time_s=float(np.mean(np.where(y[selected] == 1, duration[selected], failure_horizon))) if selected and failure_horizon is not None else None)
        report['movement_matched' if movement_matched else 'same_root_bank'] = methods
    report['interpretation'] = 'Saved branch diagnostic; query movement and duration are proxies. No task goal, closed loop, or risk guarantee.'
    report['failure_horizon_s'] = failure_horizon
    return report
