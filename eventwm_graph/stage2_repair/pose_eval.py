"""Score signed supplied-pose features on the unchanged full pilot domain."""
import argparse
import time
import numpy as np
import torch
from .diagnose import OUT, HF, read, save, sha
from .pose_features import signed_pose_features
from .pose_model import PoseNet
from eventwm_graph.stage2_hf.packed import load_arrays, get_segment
from eventwm_graph.stage2_hf.spatial import build
from eventwm_graph.stage2_hf.spatial_eval import score
from eventwm_graph.stage2_hf.light_eval import aggregate
from scripts.stage2_diagnostic.common import extended_rates


def evaluate(checkpoint, device='cuda:0', name=None):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    tick = time.perf_counter()
    saved = torch.load(checkpoint, map_location='cpu')
    model = PoseNet().to(device).eval()
    model.load_state_dict(saved['model'])
    manifest = read(HF/'packed/dev_pilot/manifest.json')
    oldrows = {r['root_id']: r for r in read(HF/'light_eval_cache/manifest.json')['rows']}
    records = []
    pooled = np.zeros((14, 3, 3), np.int64)
    windows = 0
    for row in manifest['rows']:
        a = load_arrays(row)
        old = oldrows[row['root_id']]
        lookup = {(e['start_node'], e['end_node']): e for e in old['segments']}
        with np.load(old['path'], allow_pickle=False) as z:
            x, y = z['x'], z['y']
        # Evaluation stores the active 88 columns; sampled training caches
        # retain the legacy 104 columns. Do not apply the training offset here.
        assert x.ndim == 2 and x.shape[1] == 88
        cms, scores = [], []
        empty, zero = np.zeros(14, int), 0
        for i, meta in enumerate(row['segments']):
            v = get_segment(a, i)
            path = v[3]
            e = lookup[(meta['start_node'], meta['end_node'])]
            sl = slice(e['start'], e['end'])
            x88 = x[sl]
            keys = np.floor((x88[:, 20:22].astype(np.float64)*10 + path[0, [1, 3]])/v[-1][0]).astype(np.int32)
            expected = build(*v[:4], v[-1], make_query=False)['keys']
            np.testing.assert_array_equal(keys, expected)
            gt = y[sl]
            assert np.count_nonzero(gt) == len(v[4])
            extra = signed_pose_features(keys, path, float(v[-1][0]), x88)
            pp = []
            with torch.inference_mode():
                for start in range(0, len(gt), 65536):
                    p, _ = model.probabilities(torch.from_numpy(x88[start:start+65536]).to(device), torch.from_numpy(extra[start:start+65536]).to(device))
                    pp.append(p.cpu().numpy())
            prob = np.concatenate(pp)
            if len(path) == 1:
                prob.fill(0)
                prob[:, 0] = 1
            cm, sc, ee = score(prob, gt)
            cms.append(cm)
            scores.append(sc)
            pooled += cm
            windows += 1
            if not np.any(gt):
                zero += 1
                empty += ee
        records.append(dict(root_id=row['root_id'], scene_id=row['scene_id'], cm=np.mean(cms, axis=0), scores=np.mean(scores, axis=0), empty=empty, zero_cases=zero))
    result = aggregate(records)
    best = max(result['rules'], key=lambda k: result['rules'][k]['label_aware_delta_iou'])
    result.update(best_rule=best, windows=windows, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), pooled={k: extended_rates(cm) for k, cm in zip(result['rules'], pooled)}, pooled_confusion=pooled.tolist(), seconds=time.perf_counter()-tick, candidate_keys_checked=True)
    save(OUT/'pilot'/((name or saved['config']['name']+'_epoch%02d'%saved['epoch'])+'.json'), result)
    print(name, best, result['rules'][best]['label_aware_delta_iou'], result['rules'][best]['revelation_iou'], 'seconds', result['seconds'], flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--name')
    evaluate(**vars(p.parse_args()))
