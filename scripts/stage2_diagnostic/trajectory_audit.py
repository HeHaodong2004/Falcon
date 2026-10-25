"""Evaluation-only trajectory-conditioning audit (protocol first).

No fitting, checkpoint writes, production patches, simulator work or process
management. Execution is deliberately unavailable until the shared diagnostic
helper and this protocol have been reviewed. Only legal recorded observations
may provide A/B targets; the neutral path has no counterfactual label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/stage2_diagnostic_20261025/trajectory'
CHECKPOINT_SHA256 = '1ba6844d8b7eb336c46b41125eefae16e413c4ba408da9b0a1dc47ebcef66bfd'

PROTOCOL = {
    'schema': 'stage2_diagnostic_trajectory_conditioning_v1',
    'state': 'PROTOCOL_ONLY_NO_EXPERIMENT_AUTHORIZED_BY_THIS_FILE',
    'checkpoint_sha256': CHECKPOINT_SHA256,
    'objective': 'Determine whether predictions distinguish legally recorded alternative motion segments, separating candidate gating, legal-map crop changes and direct network conditioning. No Query utility/ranking optimization.',
    'restrictions': [
        'No fitting, new checkpoint, production source edit, simulator collection, protected payload or background-process modification.',
        'No experiment until saved protocol is reviewed; no GPU launch until coordinated with the parent and Priority A has released the device.',
        'Original data, model, thresholds, raw argmax and deployed API remain unchanged.',
    ],
    'source': {
        'analysis_split': 'All 2110 existing development Roots / 108 scenes eligible; no protected confirmation payload.',
        'primary_windows': 'The exact 47128 frozen mixed windows recorded in final_segment_dev per-Root artifacts; reconstruct from their actual start/end node identities.',
        'source_files': ['legal_root.json.gz', 'queries.json.gz', 'outcomes.json.gz', 'future_observations.json.gz'],
        'identities': 'Retain source Root/scene/Query aliases, member SHA256, selected node IDs, actual pose/time arrays, input graph and trajectory hashes, script/helper/source hashes and checkpoint identity.',
        'start_graph': 'Causal replay of recorded legal history and this start node ancestry; shared start reconstructed once; do not merge sibling observations.',
        'targets': 'Terminal legal state/height for cells UNKNOWN at the common start. Known-at-start correction events are audited outside scope. No interpolated observation boundary.',
    },
    'pair_selection': {
        'maximum_primary_pairs_per_root': 1,
        'eligibility': 'Same Root and exact same saved start node; different terminal branches; exact terminal timestamp/horizon; prefer/require identical complete saved timestamp schedule for the strict primary cohort.',
        'nontrivial_path': 'At least one corresponding XYZ or wrapped-yaw difference exceeds 1e-6; distinct node IDs alone do not establish a trajectory contrast.',
        'deterministic_pose_only_score': 'Among strict eligible pairs choose largest mean planar separation in metres plus mean absolute wrapped-yaw separation divided by pi. Tie-break by SHA256(seed:Root:start:endA:endB). No target or model output enters selection.',
        'seed': 202610252,
        'coverage': 'Use all valid Roots (at most2110 pairs); scene-balanced analysis. Report availability/failures by Root and scene, start type, duration and movement. Do not imply all original windows participate in a pair.',
        'supplement': 'If strict frozen-window pairs cover fewer than256 Roots or32 scenes, propose a separately frozen supplementary cohort using only actual DAG descendants at exact common saved times. Report it separately; do not replace/merge with primary or interpolate.',
    },
    'candidate_domains': {
        'natural': 'C_A and C_B are each trajectory original initially-UNKNOWN 11m candidate domain, without pruning.',
        'fixed_union': 'U=C_A union C_B, derived solely from initial legal graph and the two supplied paths.',
        'intersection': 'I=C_A intersection C_B is the primary in-natural-support diagnostic for network conditioning. U\\I direct-network outputs are explicitly feature-domain extrapolation.',
        'score_extension': 'E=U union actual targetA keys union actual targetB keys is scoring only. OutsideU both predictions remain deterministicUNKNOWN; these truth keys never enter feature/candidate construction. Report U and E denominators and every coverage miss.',
    },
    'prediction_modes': {
        'production_mechanics': 'Call unchanged model on each real A/B path; pad probabilities to U with[1,0,0] outside own natural domain. This includes candidate gating and must not alone establish network conditioning.',
        'direct_fixed_union': 'Infer the unchanged network on every U cell for both paths with the identical common legal-map crop and no trajectory-dependent candidate mask. Preserve every supplied time/pose. Use unpruned descriptors even outside the individual 11m support; flag those rows as extrapolation.',
        'common_crop': 'One crop containing U and both paths with16-cell margin for both contexts; same supplied graph only. Assert finite crop size and containment. No target-derived crop. Use a common legal-map encoding; quantify common-crop versus normal per-path-crop differences on each own natural candidate set.',
        'crop_audit': 'Report per-feature-column and probability changes on own-domain cells, including a same-path natural-crop fixed-inference versus production numerical control. Crop intervention and execution/block-shape differences must not be attributed to trajectory conditioning.',
        'direct_intersection': 'Restrict direct-fixed-U outputs and paired targets to I, using the same arrays. This removes candidate-gating/extrapolation contributions; retain common-crop audit qualification.',
        'neutral': 'Repeat the initialXYZ/yaw at the exact recorded timestamps of the paired schedule. No new sensor observations are invented. This is an OOD/input-neutralization sensitivity diagnostic with NO recorded counterfactual target, NOT a trained baseline or evidence that true neutral observations are UNKNOWN.',
    },
    'primary_metrics': {
        'paired_proper_gain': '0.5*(L(pA,yB)+L(pB,yA)-L(pA,yA)-L(pB,yB)) on identical cells; positive gain means own recorded branch targets are better matched than crossed targets. Report logloss with frozen1e-9clip and three-class summed Brier.',
        'domains': 'Report production U/E gain, direct U/E gain (extrapolation-qualified), and direct I gain. Also report revealed-target union and true-label-changed-cell diagnostics so persistent UNKNOWN cannot hide effects.',
        'hard_maps': 'Own and crossed label-aware GraphDelta IoU, per-class confusion/precision/recall/F1; raw argmax unchanged; empty-union IoU0 convention explicit.',
        'sensitivity': 'Per-class mean and quantiles of absolute probability change, mean total variation, expected FREE/BLOCKED/revelation counts, hard labeled-set and soft per-class-min/max overlaps; all on stated common domains.',
        'mechanics': 'Candidate Jaccard, exclusive-domain sizes and proper-gain decomposition I vsU\\I; distinguish predicted changes from actual target differences.',
        'spatial_shift': 'Expected revelation-mass shift toward own exclusive2m near-path regions and exclusive10m/60degree heading-opportunity regions. Report raw and per-cell-normalized shifts and empty supports. These are input-geometry proxies, not optical visibility ground truth; heading sectors support in-place rotation cases.',
        'neutral_labels': 'No neutral accuracy/calibration claim. If neutral predictions are compared to A/B labels, name them mismatched-target/OOD diagnostics rather than supervised neutral outcomes.',
        'probabilistic_integrity': 'Verify probability finiteness/bounds/row sums, unchanged checkpoint, input immutability and same-candidate output alignment; report numerical tolerance separately from task accuracy.',
    },
    'aggregation': {
        'unit': 'One selected pair per Root; equal Root within scene then equal scene. Pair owns both correlated branches; no cell/window independence claims.',
        'uncertainty': '2000 paired scene-cluster bootstrap draws, seed202610252; preserve all nested pair/Root dependencies and same draws for mode comparisons.',
        'cohorts': 'Saved duration<=.5,.5-1,1-2,>2; root vsintermediate starts; rotation/translation/mixed; identical vsdifferentGT only as post-selection descriptive strata. No threshold tuning or model selection.',
    },
    'visual_selection': {
        'fixed_cases': 'Six valid Roots with smallest SHA256(seed:Root) selected before seeing losses.',
        'extremes': 'Three largest and three smallest direct-I logloss conditioning gains, selected after analysis and explicitly labeled observed extremes, not representative samples. Retain full ordered table and selection rule.',
        'display': 'Same initial legal map/domain; both supplied paths; both actual new-observation targets; own/cross probabilities and hard updates; neutral panel visibly labeled NO RECORDED NEUTRAL GT/OOD. Show successes and failures.',
    },
    'availability_status': 'Source inspection confirms per-node parent/time/pose/observation records and frozen window IDs exist. Exact eligible-pair/schedule counts are pending a protocol-reviewed CPU metadata pass; no pair-count or result is assumed.',
}


def write_protocol() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol_v1.json'
    value = dict(PROTOCOL, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # Never replace a saved plan after any diagnostic result exists.
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return path



def _root_availability(row):
    """Pose/time-only pairing: recorded labels never enter selection."""
    import gzip
    import itertools
    from collections import defaultdict
    import numpy as np
    from eventwm_graph.stage2_segment.data import load
    from .common import EVALUATION, digest
    source = EVALUATION / (row['root_id'] + '.json.gz')
    try:
        with gzip.open(source, 'rt') as stream:
            frozen = json.load(stream)
        assert frozen['root_id'] == row['root_id'] and frozen['scene_id'] == row['scene_id']
        # Strip target/model fields before grouping or selecting anything.
        windows = [{k: rec[k] for k in ('start_node', 'end_node', 'duration', 'poses', 'query_aliases')}
                   for rec in frozen['records']]
        horizon_groups = defaultdict(list)
        for w in windows:
            horizon_groups[(w['start_node'], float(w['duration']).hex(), w['poses'])].append(w)
        potential = [ws for ws in horizon_groups.values() if len(ws) >= 2]
        result = dict(root_id=row['root_id'], scene_id=row['scene_id'], frozen_windows=len(windows),
                      horizon_candidate_groups=len(potential),
                      horizon_candidate_pairs=sum(len(ws)*(len(ws)-1)//2 for ws in potential),
                      exact_schedule_groups=0, exact_schedule_pairs=0, physical_duplicate_pairs=0,
                      frozen_root_artifact_sha256=digest(source), pair=None)
        if not potential:
            result['status'] = 'NO_SHARED_START_HORIZON_IN_FROZEN_WINDOWS'
            return result
        legal, queries, outcomes, nodes, children = load(row)
        ancestry = {None: []}
        for node in nodes:
            ancestry[node['node_id']] = ancestry[node['parent']] + [node['node_id']]
        schedules = defaultdict(list)
        for ws in potential:
            for w in ws:
                before = ancestry[w['start_node']]
                end = ancestry[w['end_node']]
                assert end[:len(before)] == before
                ids = end[len(before):]
                start_ob = legal['history'][-1] if w['start_node'] is None else nodes[w['start_node']]['observation']
                path = np.asarray([[start_ob['timestamp'], *start_ob['robot_pose']],
                                   *[[nodes[n]['observation']['timestamp'], *nodes[n]['observation']['robot_pose']] for n in ids]], np.float64)
                assert len(path) == w['poses'] and path[-1, 0]-path[0, 0] == w['duration']
                assert np.isfinite(path).all() and np.all(np.diff(path[:, 0]) > 0)
                item = dict(w, node_ids=ids, path=path)
                schedules[(w['start_node'], tuple(float(t).hex() for t in path[:, 0]))].append(item)
        winner = None
        for (start, schedule), items in schedules.items():
            if len(items) < 2:
                continue
            result['exact_schedule_groups'] += 1
            for a, b in itertools.combinations(sorted(items, key=lambda x: (-1 if x['end_node'] is None else x['end_node'])), 2):
                pa, pb = a['path'], b['path']
                xyz = np.linalg.norm(pa[:, 1:4]-pb[:, 1:4], axis=1)
                angle = np.abs(np.arctan2(np.sin(pa[:, 4]-pb[:, 4]), np.cos(pa[:, 4]-pb[:, 4])))
                if not np.any((xyz > 1e-6) | (angle > 1e-6)):
                    result['physical_duplicate_pairs'] += 1
                    continue
                result['exact_schedule_pairs'] += 1
                planar = np.linalg.norm(pa[:, [1, 3]]-pb[:, [1, 3]], axis=1)
                diversity = float(planar.mean() + angle.mean()/np.pi)
                tie = hashlib.sha256(('202610252:' + row['root_id'] + ':' + str(start) + ':' + str(a['end_node']) + ':' + str(b['end_node'])).encode()).hexdigest()
                order = (-diversity, tie)
                if winner is None or order < winner[0]:
                    winner = (order, a, b, float(planar.mean()), float(angle.mean()))
        if winner is None:
            result['status'] = 'NO_DISTINCT_POSE_PAIR_ON_EXACT_SCHEDULE'
            return result
        order, a, b, planar, angle = winner
        result['pair'] = dict(root_id=row['root_id'], scene_id=row['scene_id'], start_node=a['start_node'],
            end_a=a['end_node'], end_b=b['end_node'], node_ids_a=a['node_ids'], node_ids_b=b['node_ids'],
            query_aliases_a=a['query_aliases'], query_aliases_b=b['query_aliases'],
            trajectory_a=a['path'].tolist(), trajectory_b=b['path'].tolist(), duration=a['duration'], poses=a['poses'],
            selection_pose_diversity=-order[0], selection_tie_sha256=order[1],
            mean_planar_separation_m=planar, mean_absolute_heading_separation_rad=angle,
            source_member_sha256={k: row['files'][k] for k in ('legal_root.json.gz', 'queries.json.gz', 'outcomes.json.gz', 'future_observations.json.gz')},
            frozen_root_artifact_sha256=result['frozen_root_artifact_sha256'])
        result['status'] = 'SELECTED_PRIMARY_PAIR'
        return result
    except Exception as error:
        return dict(root_id=row['root_id'], scene_id=row['scene_id'], status='AUDIT_FAILURE', error=repr(error), pair=None)


def availability(workers=4):
    """Approved CPU metadata pass; no map prediction, labels or GPU calls."""
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    import gzip
    import time
    import resource
    from eventwm_graph.stage2.data import source_rows
    from .common import OUT as SHARED_OUT, digest, save
    dest = OUT / 'availability_v1'
    dest.mkdir(parents=True, exist_ok=False)
    rows, manifest = source_rows('dev')
    global_protocol = json.loads((SHARED_OUT/'protocol.json').read_text())
    assert len(rows) == 2110 and len({r['scene_id'] for r in rows}) == 108
    assert digest(manifest) == global_protocol['source_manifest_sha256']
    selection_scenes = set(global_protocol['threshold_selection_scenes'])
    validation_scenes = set(global_protocol['diagnostic_validation_scenes'])
    assert not selection_scenes & validation_scenes
    assert {r['scene_id'] for r in rows} == selection_scenes | validation_scenes
    identity = dict(protocol_sha256=digest(OUT/'protocol_v1.json'),
                    global_protocol_sha256=digest(SHARED_OUT/'protocol.json'), source_manifest_sha256=digest(manifest),
                    script_sha256=digest(Path(__file__)), helper_sha256=digest(Path(__file__).with_name('common.py')),
                    workers=workers, scope='Approved CPU-only pose/time metadata availability; no model calls or target-dependent selection')
    save(dest/'identity.json', identity)
    tick = time.perf_counter(); results = []; counts = Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(_root_availability, rows)):
            results.append(result); counts[result['status']] += 1
            if i % 64 == 0:
                progress = dict(roots=i+1, total=len(rows), counts=dict(counts), seconds=time.perf_counter()-tick)
                save(dest/'progress.json', progress); print(json.dumps(progress), flush=True)
    pairs = [r['pair'] for r in results if r['pair'] is not None]
    for pair in pairs:
        pair['diagnostic_scene_split'] = 'threshold_selection70' if pair['scene_id'] in selection_scenes else 'diagnostic_validation38'
    with gzip.open(dest/'root_accounting.json.gz', 'wt') as stream:
        json.dump(results, stream)
    save(dest/'pairs.json', pairs)
    stats = dict(identity=identity, roots=len(rows), scenes=108, counts=dict(counts), selected_pairs=len(pairs),
        paired_scenes=len({p['scene_id'] for p in pairs}),
        selected_pair_splits=dict(Counter(p['diagnostic_scene_split'] for p in pairs)),
        selected_pair_start_types=dict(Counter('root' if p['start_node'] is None else 'intermediate' for p in pairs)),
        exact_schedule_candidate_groups=sum(r.get('exact_schedule_groups', 0) for r in results),
        exact_schedule_candidate_pairs=sum(r.get('exact_schedule_pairs', 0) for r in results),
        physical_duplicate_pairs=sum(r.get('physical_duplicate_pairs', 0) for r in results),
        failures=[{k: v for k, v in r.items() if k not in ('pair',)} for r in results if r['status']=='AUDIT_FAILURE'],
        supplemental_trigger=len(pairs) < 256 or len({p['scene_id'] for p in pairs}) < 32,
        target_or_model_outputs_used_in_selection=False, protected_payloads_opened=False,
        seconds=time.perf_counter()-tick, parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(dest/'statistics.json', stats)
    print(json.dumps(stats), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-protocol', action='store_true')
    parser.add_argument('--show-protocol', action='store_true')
    parser.add_argument('--availability', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--check-math', action='store_true')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--name', default='primary_v1')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.write_protocol:
        print(write_protocol())
    elif args.show_protocol:
        print(json.dumps(PROTOCOL, indent=2))
    elif args.availability:
        availability(args.workers)
    elif args.check_math:
        from .trajectory_metrics import check_math
        print(json.dumps(check_math()))
    elif args.evaluate:
        from .trajectory_execution import run
        run(args.device,args.workers,args.name,args.limit)
    elif args.summarize:
        from .trajectory_execution import summarize
        summarize(args.name)
    else:
        parser.error('Protocol-only phase: use --write-protocol; execution awaits shared helper/review.')


if __name__ == '__main__':
    main()
