"""Matched segment evaluation; scene-cluster uncertainty, no ranking metrics."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import time
import numpy as np
from eventwm_graph.stage2.data import source_rows
from .api import SegmentPredictor
from .data import replay
from .common import OUT, CHECKPOINT, CHECKPOINT_SHA, dump, sha, snapshot


def rates(cm):
    cm = np.asarray(cm, np.float64)
    tp = np.diag(cm); actual = cm.sum(1); predicted = cm.sum(0)
    result = {}
    for i, name in enumerate(('unknown', 'free', 'blocked')):
        result[name] = dict(precision=float(tp[i]/max(predicted[i], 1e-20)),
            recall=float(tp[i]/max(actual[i], 1e-20)), f1=float(2*tp[i]/max(actual[i]+predicted[i], 1e-20)))
    result['label_aware_delta_iou'] = float((tp[1]+tp[2])/max(actual[1:].sum()+predicted[1:].sum()-tp[1]-tp[2], 1e-20))
    result['revelation_iou'] = float(cm[1:, 1:].sum()/max(cm[1:].sum()+cm[:, 1:].sum()-cm[1:, 1:].sum(), 1e-20))
    result['missed_revelations'] = float(cm[1:, 0].sum())
    result['incorrect_free_blocked'] = float(cm[1, 2]+cm[2, 1])
    result['false_revelations'] = float(cm[0, 1:].sum())
    result['terminal_label_agreement_on_true_revelations'] = float(tp[1:].sum()/max(actual[1:].sum(), 1e-20))
    return result


def groups(ex):
    p = ex['trajectory']; duration = p[-1, 0]-p[0, 0]
    distance = float(np.linalg.norm(np.diff(p[:, [1, 3]], axis=0), axis=1).sum())
    dy = np.diff(p[:, 4]); rotation = float(np.abs(np.arctan2(np.sin(dy), np.cos(dy))).sum())
    duration_label = next((label for cutoff, label in ((.5000001, '<=0.5'), (1.0000001, '0.5-1'), (2.0000001, '1-2')) if duration <= cutoff), '>2')
    motion = 'mixed' if distance > .01 and rotation > .01 else 'translation' if distance > .01 else 'rotation' if rotation > .01 else 'stationary'
    return ['duration:'+duration_label, 'motion:'+motion,
            'new_cells:'+('0' if not ex['target'] else '1-99' if len(ex['target']) < 100 else '>=100')]


def score(ex, delta):
    keys = delta['candidate_cells']; p = delta['probabilities']; pred = delta['states']; n = len(keys)
    target = ex['target']; y = np.array([target.get(tuple(k), (0, 0))[0] for k in keys], np.int64)
    present = set(map(tuple, keys)); missing = [v for k, v in target.items() if k not in present]
    cm = np.bincount(3*y+pred, minlength=9).reshape(3, 3)
    for label, _ in missing:
        cm[label, 0] += 1
    denom = n+len(missing)
    ce = -np.log(np.clip(p[np.arange(n), y], 1e-9, 1))
    br = np.square(p).sum(-1)-2*p[np.arange(n), y]+1
    scores = [float((ce.sum()-len(missing)*np.log(1e-9))/max(denom, 1)),
              float((br.sum()+2*len(missing))/max(denom, 1))]
    observed = y > 0
    observed_scores = [float(ce[observed].sum()-len(missing)*np.log(1e-9)),
                       float(br[observed].sum()+2*len(missing)), int(observed.sum())+len(missing)]
    reliability = np.zeros((3, 15, 3))
    for cls in range(3):
        bins = np.minimum((p[:, cls]*15).astype(int), 14)
        reliability[cls, :, 0] = np.bincount(bins, minlength=15)
        reliability[cls, :, 1] = np.bincount(bins, weights=p[:, cls], minlength=15)
        reliability[cls, :, 2] = np.bincount(bins, weights=y == cls, minlength=15)
        if missing:
            index = 14 if cls == 0 else 0
            reliability[cls, index] += [len(missing), len(missing)*(cls == 0), sum(label == cls for label, _ in missing)]
    reliability /= max(denom, 1)
    # Check the ACTUAL returned graph, including preservation of original cells.
    original = {tuple(c['cell']): c for c in ex['graph']['cells']}
    future = {tuple(c['cell']): c for c in delta['future_partial_graph']['cells']}
    if any(future.get(k) != c for k, c in original.items()):
        raise ValueError('known map cell changed in hypothetical graph')
    if len(future) != len(original)+len(delta['static_updates']):
        raise ValueError('hypothetical graph cardinality mismatch')
    terminal_cm = np.zeros((3, 3), np.int64)
    height_error = [0., 0]
    for key, truth in zip(keys, y):
        row = future.get(tuple(key)); label = row['state'] if row else 0
        terminal_cm[truth, label] += 1
        if truth and label == truth:
            height_error[0] += abs(row['height']-target[tuple(key)][1]); height_error[1] += 1
    for label, _ in missing:
        terminal_cm[label, 0] += 1
    np.testing.assert_array_equal(terminal_cm, cm)
    pr = []
    reveal = p[:, 1]+p[:, 2]
    for threshold in (.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9):
        chosen = reveal >= threshold
        pr.append([int(np.count_nonzero(chosen & observed)), int(np.count_nonzero(chosen & ~observed)),
                   int(np.count_nonzero(~chosen & observed))+len(missing)])
    return dict(confusion=cm.tolist(), scores=scores, observed_scores=observed_scores, reliability=reliability.tolist(),
                height_error=height_error, pr=pr, out_of_domain_positive=len(missing), candidates=n,
                future_graph_verified=True, terminal_confusion=terminal_cm.tolist())


MODELS = None; DEST = None

def initialize(checkpoint, dest):
    global MODELS, DEST
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    MODELS = dict(baseline=SegmentPredictor(CHECKPOINT))
    if checkpoint:
        MODELS['candidate'] = SegmentPredictor(checkpoint,execution='optimized',future_format='native',device='cuda:1')
        MODELS['reference'] = SegmentPredictor(checkpoint,execution='reference',future_format='snapshot')
    DEST = dest


def root_worker(row):
    dest = DEST/(row['root_id']+'.json.gz')
    if dest.exists():
        with gzip.open(dest, 'rt') as f:
            return json.load(f)
    records = []; tick = time.perf_counter(); audit = None
    try:
        for ex in replay(row, 'mixed'):
            if 'audit' in ex:
                audit = ex['audit']; continue
            record = {k: ex[k] for k in ('query_id', 'query_aliases', 'start_node', 'end_node', 'known_terminal_differences')}
            record.update(groups=groups(ex), duration=float(ex['trajectory'][-1, 0]-ex['trajectory'][0, 0]),
                          poses=len(ex['trajectory']), new_free=sum(v[0] == 1 for v in ex['target'].values()),
                          new_blocked=sum(v[0] == 2 for v in ex['target'].values()))
            reference_keys = None
            candidate_delta=None
            for name, model in MODELS.items():
                delta = model.predict(partial_graph=ex['graph'], trajectory=ex['trajectory'])
                if reference_keys is None:
                    reference_keys = delta['candidate_cells']
                else:
                    np.testing.assert_array_equal(reference_keys, delta['candidate_cells'])
                record[name] = score(ex, delta)
                if name=='candidate':candidate_delta=delta
                if name=='reference':
                    for field in ('candidate_cells','states','unknown_cells'):np.testing.assert_array_equal(candidate_delta[field],delta[field])
                    for field in ('probabilities','heights'):np.testing.assert_allclose(candidate_delta[field],delta[field],atol=2e-6,rtol=0)
                    record['optimized_equivalence']=dict(max_probability_error=float(np.max(abs(candidate_delta['probabilities']-delta['probabilities']))) if len(delta['states']) else 0.,
                        max_height_error=float(np.max(abs(candidate_delta['heights']-delta['heights']))) if len(delta['states']) else 0.,hard_states_exact=True)
            records.append(record)
        if audit is None:
            raise ValueError('incomplete evaluation replay')
        value = dict(root_id=row['root_id'], scene_id=row['scene_id'], records=records, audit=audit, seconds=time.perf_counter()-tick)
    except Exception as error:
        # Correctness errors abort evaluation; data failures are accounted by the
        # extraction/audit stage and must never silently drop hard examples here.
        raise RuntimeError(row['root_id']+': '+repr(error)) from error
    with gzip.open(dest, 'wt') as f:
        json.dump(value, f)
    return value


def aggregate(results, model, group=None):
    scenes = defaultdict(list); pooled = np.zeros((3, 3)); nsegments = 0
    for root in results:
        records = [r[model] for r in root['records'] if model in r and (group is None or group in r['groups'])]
        if not records:
            continue
        nsegments += len(records)
        pooled += np.sum([r['confusion'] for r in records], axis=0)
        scenes[root['scene_id']].append(dict(cm=np.mean([r['confusion'] for r in records], axis=0),
            scores=np.mean([r['scores'] for r in records], axis=0),
            reliability=np.mean([r['reliability'] for r in records], axis=0),
            observed=np.mean([r['observed_scores'] for r in records], axis=0),
            pr=np.mean([r['pr'] for r in records], axis=0), height=np.sum([r['height_error'] for r in records], axis=0)))
    scene_names = sorted(scenes)
    arrays = {key: np.array([np.mean([r[key] for r in scenes[s]], axis=0) for s in scene_names])
              for key in ('cm', 'scores', 'reliability', 'observed', 'pr')}
    cm = arrays['cm'].mean(0); scores = arrays['scores'].mean(0); observed = arrays['observed'].mean(0)
    calibration = {}
    for i, name in enumerate(('unknown', 'free', 'blocked')):
        bins = arrays['reliability'].mean(0)[i]
        calibration[name] = dict(ece=float(np.abs(bins[:, 1]-bins[:, 2]).sum()/max(bins[:, 0].sum(), 1e-20)),
            bins=[dict(lower=j/15, upper=(j+1)/15, mass=float(row[0]), probability=float(row[1]/max(row[0], 1e-20)),
                       frequency=float(row[2]/max(row[0], 1e-20))) for j, row in enumerate(bins)])
    heights = np.sum([r['height'] for rs in scenes.values() for r in rs], axis=0)
    metrics = dict(roots=sum(len(rs) for rs in scenes.values()), scenes=len(scenes), segments=nsegments,
        scene_balanced=rates(cm), pooled=rates(pooled), pooled_confusion=pooled.tolist(),
        logloss=float(scores[0]), brier=float(scores[1]), calibration=calibration,
        observed_cell_logloss=float(observed[0]/max(observed[2], 1e-20)), observed_cell_brier=float(observed[1]/max(observed[2], 1e-20)),
        correct_revelation_height_mae=float(heights[0]/max(heights[1], 1)), correct_revelation_height_support=int(heights[1]),
        revelation_precision_recall=[dict(threshold=t, precision=float(a/max(a+b, 1e-20)), recall=float(a/max(a+c, 1e-20)))
            for t, (a, b, c) in zip((.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9), arrays['pr'].mean(0))])
    return metrics, arrays, scene_names


def uncertainty(arrays):
    rng = np.random.default_rng(2425); draws = defaultdict(list)
    n = len(arrays['cm'])
    for _ in range(2000):
        indices = rng.integers(0, n, n)
        r = rates(arrays['cm'][indices].mean(0)); scores = arrays['scores'][indices].mean(0)
        for cls in ('unknown', 'free', 'blocked'):
            for metric in ('precision', 'recall', 'f1'):
                draws[cls+'_'+metric].append(r[cls][metric])
        draws['label_aware_delta_iou'].append(r['label_aware_delta_iou'])
        draws['logloss'].append(float(scores[0])); draws['brier'].append(float(scores[1]))
    return {key: np.array(values) for key, values in draws.items()}


def summarize(results, dest):
    models = [name for name in ('baseline','candidate','reference') if name in results[0]['records'][0]]
    report = {}; draws = {}; names = None
    for name in models:
        m, arrays, scenes = aggregate(results, name)
        if names is not None:
            assert names == scenes
        names = scenes; draws[name] = uncertainty(arrays)
        m['ci95'] = {k: np.quantile(v, [.025, .975]).tolist() for k, v in draws[name].items()}
        report[name] = m
    if 'candidate' in models:
        report['paired_candidate_minus_baseline_ci95'] = {k: np.quantile(draws['candidate'][k]-draws['baseline'][k], [.025, .975]).tolist() for k in draws['baseline']}
    report['cohorts'] = {}
    groups = sorted({g for root in results for r in root['records'] for g in r['groups']})
    for group in groups:
        report['cohorts'][group] = {}
        for name in models:
            m, a, _ = aggregate(results, name, group)
            m['ci95'] = {k: np.quantile(v, [.025, .975]).tolist() for k, v in uncertainty(a).items()}
            report['cohorts'][group][name] = m
    report['audit'] = {key: sum(r['audit'].get(key, 0) for r in results) for key in set().union(*(r['audit'] for r in results))}
    report['terminal_graph_verification'] = all(s['future_graph_verified'] for root in results for rec in root['records'] for name in models for s in [rec[name]])
    if 'reference' in models:
        eq=[rec['optimized_equivalence'] for root in results for rec in root['records']]
        report['optimized_equivalence']=dict(segments=len(eq),all_hard_states_exact=all(r['hard_states_exact'] for r in eq),
            max_probability_error=max(r['max_probability_error'] for r in eq),max_height_error=max(r['max_height_error'] for r in eq),atol=2e-6,rtol=0)
    zeros=[(root,rec) for root in results for rec in root['records'] if rec['new_free']+rec['new_blocked']==0]
    report['zero_new_observation_cases']=dict(segments=len(zeros),models={name:dict(exact_empty_predictions=int(sum(np.asarray(rec[name]['confusion'])[:,1:].sum()==0 for _,rec in zeros)),
        false_revelations=sum(int(np.asarray(rec[name]['confusion'])[:,1:].sum()) for _,rec in zeros)) for name in models})
    dump(dest/'metrics.json', report)
    return report


def evaluate(name, checkpoint=None, workers=8):
    dest = OUT/'evaluation'/name; dest.mkdir(parents=True, exist_ok=False)
    rows, source = source_rows('dev')
    dump(dest/'protocol.json', dict(protocol_sha256=sha(OUT/'protocol.json'), source=snapshot(),
        data_manifest_sha256=sha(source), baseline_checkpoint_sha256=CHECKPOINT_SHA,
        checkpoint=checkpoint, checkpoint_sha256=sha(checkpoint) if checkpoint else None,
        strategy='fixed_mixed', protected_payloads_opened=False))
    tick = time.perf_counter(); results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(checkpoint, dest)) as pool:
        for i, result in enumerate(pool.map(root_worker, rows)):
            results.append(result)
            if i % 32 == 0:
                print(json.dumps(dict(roots=i+1, total=len(rows), seconds=time.perf_counter()-tick)), flush=True)
    summarize(results, dest)
    dump(dest/'complete.json', dict(roots=len(results), seconds=time.perf_counter()-tick))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--name', required=True); p.add_argument('--checkpoint'); p.add_argument('--workers', type=int, default=8)
    evaluate(**vars(p.parse_args()))
