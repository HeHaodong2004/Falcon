#!/usr/bin/env python3
"""Offline binary SR audit; never collects data or trains a model."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

from collector.core import build_labels, digest, read, write
from eventwm_scale import verify_root

REPO = Path(__file__).resolve().parents[1]


def sr_target(native, disc, static, completed, horizon_covered):
    if native or disc or static:
        return True, 0
    if completed and horizon_covered:
        return True, 1
    return False, None


def counts(rows):
    return dict(total=len(rows), valid=sum(r['sr_valid'] for r in rows),
                safe=sum(r['sr_label'] == 1 for r in rows),
                collision=sum(r['sr_label'] == 0 for r in rows),
                invalid=sum(not r['sr_valid'] for r in rows))


def breakdown(rows, key):
    return {str(v): counts([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})}


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def failed_roots(dataset, manifest):
    result = []
    for r in manifest['roots']:
        if r['state'] != 'failed':
            continue
        for a in r['attempts']:
            logs = sorted((dataset / a['path']).glob('audits/workers/*/run.log'))
            text = '\n'.join(p.read_text(errors='replace') for p in logs)
            evidence = []
            for p in logs:
                for i, line in enumerate(p.read_text(errors='replace').splitlines(), 1):
                    if any(x in line for x in ('RuntimeError: Prefix terminated before root',
                        'AssertionError: ESP_CHECK failed: No Stage Attributes exists',
                        'Stage Template managed object handle `` not found',
                        'Root replay mismatch', 'Collector integrity audit failed', 'Canonical/worker mismatch')):
                        evidence.append(dict(path=str(p.relative_to(dataset)), line=i, text=line))
            if any(x in text for x in ('Root replay mismatch', 'Collector integrity audit failed', 'Canonical/worker mismatch')):
                category = 'integrity_replay_failure'
            elif 'Prefix terminated before root' in text:
                category = 'prefix_termination_before_root'
            elif 'No Stage Attributes exists' in text:
                category = 'missing_assets'
            elif 'Stage Template managed object handle `` not found' in text:
                category = 'replicacad_config_initialization'
            else:
                category = 'collection_process_failure'
            executions = [dict(worker=p.parent.name, **read(p)) for p in
                          (dataset / a['path']).glob('audits/workers/*/execution.json')]
            result.append(dict(root_id=r['root_id'], scene=r['scene'], human_count=r['human_count'],
                root_profile=r['recipe'].get('sampling', {}).get('profile', 'unspecified'),
                attempt=a['number'], category=category,
                exact_failure_reason=evidence[-1]['text'] if evidence else a.get('error', 'unknown'),
                evidence=evidence, executions=executions,
                detail=('Environment episode ended; the saved failed-prefix artifacts do not identify static/human contact cause.'
                        if category == 'prefix_termination_before_root' else
                        'Unresolved ReplicaCAD stage/object/URDF references; terminal abort is an empty stage handle.'
                        if category == 'replicacad_config_initialization' else '')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=REPO / 'datasets/raw/eventwm_prod_v1')
    parser.add_argument('--report', type=Path, default=REPO / 'EVENTWM_SR_DATA_AUDIT.md')
    args = parser.parse_args()
    dataset = args.dataset.resolve(); manifest = read(dataset / 'manifest.json')
    output = dataset / 'audits'; output.mkdir(exist_ok=True)
    rows, root_rows, contradictions, checks = [], [], [], Counter()
    def check(name, condition, **context):
        checks[name] += 1
        if not condition:
            contradictions.append(dict(check=name, **context))
    completed = [r for r in manifest['roots'] if r['state'] == 'completed']
    query_ids = set(); primary_pairs = set(); audit_excluded = 0
    for root in completed:
        rid = root['root_id']; path = dataset / root['path']
        try:
            verify_root(dataset, root, deep=True)
            check('completion_marker_and_all_sealed_files', True, root_id=rid)
        except Exception as e:
            check('completion_marker_and_all_sealed_files', False, root_id=rid, error=str(e))
        saved = read(path / 'audits/audits.json')
        check('stored_collector_integrity_audits', all(v['status'] == 'PASS' for k, v in saved.items()
              if k != 'human_reactivity'), root_id=rid)
        queries = read(path / 'queries/queries.json'); labels = read(path / 'labels/labels.json')
        branches = [json.loads(line) for line in (path / 'branch_truth/branches.jsonl').read_text().splitlines()]
        lb = {l['branch_id']: l for l in labels}; bb = {b['branch_id']: b for b in branches}
        check('unique_branch_and_label_ids', len(lb) == len(labels) and len(bb) == len(branches), root_id=rid)
        primary_ids = {f'b{i:03}' for i in range(len(queries))}
        check('complete_primary_inventory', primary_ids == {b['branch_id'] for b in branches
              if re.fullmatch(r'b\d+', b['branch_id'])} and primary_ids <= lb.keys(), root_id=rid)
        check('labels_cover_all_saved_branches', set(lb) == set(bb), root_id=rid)
        audit_excluded += sum(b['branch_id'] not in primary_ids for b in branches)
        history = read(path / 'legal_observations/history.json'); tracks = history[-1]['human_tracks']
        visible_history = any(t['visible_now'] for h in history for t in h['human_tracks'])
        visible = sum(t['visible_now'] for t in tracks); lost = sum(not t['visible_now'] for t in tracks)
        category = ('visible_and_lost_at_root' if visible and lost else 'visible_at_root' if visible else
                    'lost_at_root' if lost else 'observed_history_no_current_track' if visible_history else 'no_human_observed')
        start = len(rows)
        for i, qrow in enumerate(queries):
            q = qrow['query']; qid = q['query_id']; bid = f'b{i:03}'
            ctx = dict(root_id=rid, query_id=qid, branch_id=bid)
            check('unique_primary_query_ids_global', qid not in query_ids, **ctx); query_ids.add(qid)
            check('unique_primary_root_query_pairs', (rid, qid) not in primary_pairs, **ctx); primary_pairs.add((rid, qid))
            if bid not in bb or bid not in lb:
                check('missing_primary_branch_or_label', False, **ctx); continue
            b, l = bb[bid], lb[bid]
            check('root_query_branch_alignment', b['root_id'] == q['root_id'] == rid and
                  b['query_id'] == l['query_id'] == qid and b['branch_id'] == l['branch_id'] == bid, **ctx)
            check('fixed_query_hash', b['query_hash_before'] == b['query_hash_after'] == qrow['sha256'] == digest(q), **ctx)
            check('saved_label_equals_truth_recomputation', build_labels(b) == l, **ctx)
            check('completion_and_termination_agreement', b['completed'] == l['completed'] and
                  l['censored'] == (not b['completed']) and b['termination_reason'] == l['termination_reason'], **ctx)
            actual_duration = b['truth'][-1]['timestamp'] - b['truth'][0]['timestamp']
            tol = q['clock_quantization_tolerance']
            check('duration_metadata_agreement', abs(b['observed_duration'] - actual_duration) < 1e-9 and
                  abs(l['observed_duration'] - actual_duration) < 1e-9 and
                  b['planned_duration'] == l['planned_duration'] == q['planned_duration'], **ctx)
            controls = b['controls']; expected = q['executable_controls']
            control_match = len(controls) == len(expected) and all(c['index'] == j and
                            c['issued_command'] == expected[j] for j, c in enumerate(controls))
            horizon = bool(control_match and abs(actual_duration - q['planned_duration']) < tol)
            native, disc, static = (l['native_robot_human_contact'], l['research_disc_contact'], l['static_contact'])
            human = native or disc; collision = human or static
            valid, sr = sr_target(native, disc, static, b['completed'], horizon)
            check('human_union_agreement', l['human_collision_event'] == human == l['event_observed'], **ctx)
            check('no_safe_collision', not (sr == 1 and collision), **ctx)
            check('collision_label_requires_collision_evidence', sr != 0 or collision, **ctx)
            check('safe_requires_execution_completion', sr != 1 or b['completed'], **ctx)
            check('safe_requires_full_horizon', sr != 1 or horizon, **ctx)
            check('declared_completed_covers_horizon', not b['completed'] or horizon, **ctx)
            check('native_collision_termination_evidence', b['termination_reason'] != 'robot_human_collision' or native, **ctx)
            check('static_collision_termination_evidence', b['termination_reason'] != 'static_collision' or static, **ctx)
            variant = ('WAIT' if q['direction'] == 'WAIT' else
                       f"wait={q['wait_duration']:g},speed={q['speed_scale']:g}")
            rows.append(dict(**ctx, scene=root['scene'], human_count=root['human_count'],
                root_profile=root['recipe'].get('sampling', {}).get('profile', 'unspecified'),
                root_path=root['path'], planned_duration=q['planned_duration'], observed_duration=actual_duration,
                execution_completed=b['completed'], termination_reason=b['termination_reason'],
                full_horizon_observed=horizon, clock_quantization_tolerance=tol,
                executed_controls=len(controls), planned_controls=len(expected),
                native_human_collision=native, research_disc_collision=disc, human_collision=human,
                static_collision=static, collision_any=collision, sr_valid=valid, sr_label=sr,
                first_native_event_time=l['first_native_event_time'], first_disc_event_time=l['first_disc_event_time'],
                first_static_event_time=l['first_static_event_time'],
                spatial_path_id=digest(q['spatial_path']), local_target=q['local_target'],
                spatial_path=q['spatial_path'], direction=q['direction'], wait_time=q['wait_duration'],
                effective_wait_time=q['effective_wait_duration'], speed_scale=q['speed_scale'], query_variant=variant,
                human_observed_in_history=visible_history, visible_tracks_at_root=visible, lost_tracks_at_root=lost,
                visibility_history_category=category))
        rr = rows[start:]; c = counts(rr)
        root_rows.append(dict(root_id=rid, scene=root['scene'], human_count=root['human_count'],
            root_profile=root['recipe'].get('sampling', {}).get('profile', 'unspecified'), **c,
            mixed_sr=bool(c['safe'] and c['collision']), all_safe=bool(rr) and c['safe'] == len(rr),
            all_collision=bool(rr) and c['collision'] == len(rr), contains_invalid=bool(c['invalid'])))
        print(rid, c, flush=True)
    total = counts(rows)
    check('primary_count_matches_manifest', len(rows) == sum(r['stats']['primary_branches'] for r in completed))
    check('requested_dataset_primary_count', len(rows) == 1504)
    failures = failed_roots(dataset, manifest)
    check('failed_root_count', len({f['root_id'] for f in failures}) == sum(r['state'] == 'failed' for r in manifest['roots']) == 52)
    combinations = []
    for n in (False, True):
        for d in (False, True):
            for s in (False, True):
                if n or d or s:
                    combinations.append(dict(native=n, disc=d, static=s,
                        count=sum((r['native_human_collision'], r['research_disc_collision'], r['static_collision']) == (n,d,s) for r in rows)))
    collision = dict(human_union=sum(r['human_collision'] for r in rows),
                     static_union=sum(r['static_collision'] for r in rows),
                     human_only=sum(r['human_collision'] and not r['static_collision'] for r in rows),
                     static_only=sum(r['static_collision'] and not r['human_collision'] for r in rows),
                     human_and_static=sum(r['human_collision'] and r['static_collision'] for r in rows),
                     native_human_union=sum(r['native_human_collision'] for r in rows),
                     research_disc_union=sum(r['research_disc_collision'] for r in rows),
                     exclusive_combinations=combinations)
    invalid = {k: sum(not r['sr_valid'] and r['termination_reason'] == k for r in rows)
               for k in ('execution_failure','timeout','environment_termination','collection_error')}
    invalid['other'] = total['invalid'] - sum(invalid.values())
    groups = defaultdict(list)
    for row in rows:
        groups[(row['root_id'], row['spatial_path_id'])].append(row)
    path_rows = []
    for (rid, pid), rs in sorted(groups.items()):
        variants = {(r['wait_time'], r['effective_wait_time'], r['speed_scale']) for r in rs if r['sr_valid']}
        qualifies = len(variants) >= 2
        changes = qualifies and {r['sr_label'] for r in rs if r['sr_valid']} == {0, 1}
        path_rows.append(dict(root_id=rid, scene=rs[0]['scene'], spatial_path_id=pid,
            spatial_path=rs[0]['spatial_path'], valid_variant_count=len(variants), qualifies=qualifies, sr_changes=changes,
            variants=[{k:r[k] for k in ('query_id','query_variant','wait_time','effective_wait_time','speed_scale',
                       'planned_duration','observed_duration','sr_valid','sr_label','termination_reason',
                       'human_collision','static_collision')} for r in rs]))
    visibility = {name: counts([r for r in rows if predicate(r)]) for name, predicate in (
        ('no human observed in history', lambda r: not r['human_observed_in_history']),
        ('human observed in history', lambda r: r['human_observed_in_history']),
        ('visible track at root', lambda r: r['visible_tracks_at_root'] > 0),
        ('lost track at root', lambda r: r['lost_tracks_at_root'] > 0))}
    failure_groups = Counter((r['scene'],r['human_count'],r['root_profile'],r['category']) for r in failures)
    summary = dict(schema_version=1, dataset=str(dataset),
        manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest(),
        semantics='collision=(native_human OR declared_research_disc OR declared_static); collision => valid SR=0; otherwise completed AND full_horizon => valid SR=1; otherwise invalid/null',
        horizon_policy='All fixed controls issued exactly and abs(observed-planned) < stored clock_quantization_tolerance (0.008001s); native clock authoritative.',
        total=total, sr_valid_rate=total['valid']/total['total'], completed_roots=len(completed),
        audit_repeat_branches_excluded=audit_excluded, collision_breakdown=collision, invalid_breakdown=invalid,
        by_scene=breakdown(rows,'scene'), by_human_count=breakdown(rows,'human_count'),
        by_root_profile=breakdown(rows,'root_profile'), by_visibility=visibility,
        by_visibility_exclusive=breakdown(rows,'visibility_history_category'),
        root_signal={k:sum(r[k] for r in root_rows) for k in ('mixed_sr','all_safe','all_collision','contains_invalid')},
        roots=root_rows, path_group_count=len(path_rows),
        path_groups_with_two_valid_variants=sum(g['qualifies'] for g in path_rows),
        path_groups_with_different_sr=sum(g['sr_changes'] for g in path_rows),
        failed_root_count=len({f['root_id'] for f in failures}), failed_attempt_count=len(failures),
        failed_root_taxonomy=dict(Counter(f['category'] for f in failures)),
        failed_root_groups=[dict(scene=s,human_count=h,root_profile=p,category=c,count=n)
                            for (s,h,p,c),n in sorted(failure_groups.items())],
        check_counts=dict(checks), contradictions=contradictions,
        data_consistency='FAIL' if contradictions else 'PASS')
    write(output/'sr_branch_audit.json',rows); csv_write(output/'sr_branch_audit.csv', rows)
    write(output/'sr_root_audit.json',root_rows); csv_write(output/'sr_root_audit.csv',root_rows)
    write(output/'sr_path_variant_audit.json',path_rows)
    write(output/'sr_failed_root_audit.json',failures); csv_write(output/'sr_failed_root_audit.csv',failures)
    write(output/'sr_summary.json',summary)
    write_report(args.report, summary, path_rows)
    print(json.dumps({k: summary[k] for k in ('total','collision_breakdown','invalid_breakdown','root_signal',
          'path_groups_with_two_valid_variants','path_groups_with_different_sr','failed_root_taxonomy','data_consistency')}, indent=2))
    if contradictions:
        raise SystemExit(1)


def write_report(path, s, groups):
    lines = ['# EventWM SR production data audit', '',
        'Scope: 48 completed roots; 1,504 primary branches. Audit/repeat branches and failed-root attempts do not enter SR statistics. No model training, collection, collector edits, or label rewrites were performed.', '',
        '**Frozen SR semantics:** native human contact OR declared research human-disc contact OR declared static contact gives valid SR=0, regardless of early termination. With no collision, SR=1 requires completed execution and the full fixed query. All other cases are invalid/null, never a third class.', '',
        'Full horizon means every planned fixed control was issued, plus duration agreement within the collector’s stored 0.008001 s native-clock quantization tolerance. This preserves the existing clock contract. Collision declarations are taken from saved labels and independently recomputed from saved branch truth with the unchanged `build_labels` function.', '',
        'The previous 174-sample completed-only event index is **not an SR index** and must not be used to count usable SR data. Its exclusion of early collision terminations is inapplicable here.', '',
        '## Counts by data condition', '']
    def table(title, mapping):
        lines.extend([f'### {title}', '', '| Condition | Total | SR-valid | Safe | Collision | Invalid |',
                      '|---|---:|---:|---:|---:|---:|'])
        for key,c in mapping.items():
            lines.append(f"| {key} | {c['total']} | {c['valid']} | {c['safe']} | {c['collision']} | {c['invalid']} |")
        lines.append('')
    safe_scenes = ', '.join(f"{scene} ({c['safe']})" for scene, c in s['by_scene'].items() if c['safe'])
    changing_scenes = ', '.join(sorted({g['scene'] for g in groups if g['sr_changes']})) or 'none'
    lines.extend([f"Valid labels occur in {sum(c['valid'] > 0 for c in s['by_scene'].values())} scenes and "
                  f"{sum(c['valid'] > 0 for c in s['by_root_profile'].values())} profiles. "
                  f"Safe outcomes occur in: {safe_scenes}. Timing/speed-changing path groups occur in: {changing_scenes}.", ''])
    table('Scene',s['by_scene']); table('Human count',s['by_human_count']); table('Root sampling profile',s['by_root_profile'])
    table('Visibility history',s['by_visibility'])
    lines.extend(['Visibility rows overlap: visible/lost-at-root are subsets of observed-history, and one root can contain both visible and lost tracks. The first two rows partition the dataset.', ''])
    table('Exclusive visibility categories',s['by_visibility_exclusive'])
    lines.extend(['## Collision evidence and invalid branches', '',
        '| Native human | Research disc | Static | Branches |', '|---|---|---|---:|'])
    for c in s['collision_breakdown']['exclusive_combinations']:
        lines.append(f"| {c['native']} | {c['disc']} | {c['static']} | {c['count']} |")
    c=s['collision_breakdown']
    lines.extend(['', f"Human union: {c['human_union']}; native union: {c['native_human_union']}; research-disc union: {c['research_disc_union']}. The seven rows above are mutually exclusive, so all overlaps are explicit.", '',
        'Invalid breakdown: '+', '.join(f'{k}={v}' for k,v in s['invalid_breakdown'].items())+'.', '',
        '## Root and timing/speed learning signal', '',
        '| Scene | Roots | Mixed SR | All-safe | All-collision | With invalid |', '|---|---:|---:|---:|---:|---:|'])
    for scene in s['by_scene']:
        rs=[r for r in s['roots'] if r['scene']==scene]
        lines.append(f"| {scene} | {len(rs)} | {sum(r['mixed_sr'] for r in rs)} | {sum(r['all_safe'] for r in rs)} | {sum(r['all_collision'] for r in rs)} | {sum(r['contains_invalid'] for r in rs)} |")
    lines.extend(['', 'All-safe/all-collision require every candidate to have that valid label. Mixed means both valid labels occur; invalid presence is an independent flag.', '',
        f"Spatial-path groups are scoped to the same root and the exact canonical `spatial_path` SHA256; variants differ in requested/effective wait or speed. {s['path_groups_with_two_valid_variants']} groups have at least two valid variants; **{s['path_groups_with_different_sr']} change SR across variants**. WAIT-only singleton groups are not eligible.", '',
        '| Scene | Groups with ≥2 valid variants | Groups changing SR |', '|---|---:|---:|'])
    for scene in s['by_scene']:
        gs=[g for g in groups if g['scene']==scene]
        lines.append(f"| {scene} | {sum(g['qualifies'] for g in gs)} | {sum(g['sr_changes'] for g in gs)} |")
    lines.extend(['', 'Examples (all variants share the listed root and exact path):', '', '| Root | Path SHA256 prefix | Variants → SR |', '|---|---|---|'])
    examples=[]
    for g in groups:
        if g['sr_changes'] and g['root_id'] not in {e['root_id'] for e in examples}:
            examples.append(g)
        if len(examples)==4: break
    for g in examples:
        lines.append(f"| {g['root_id']} | {g['spatial_path_id'][:12]} | "+'; '.join(f"{v['query_variant']} → {v['sr_label']}" for v in g['variants'])+' |')
    lines.extend(['', '## Failed roots (separate from branch labels)', '',
        f"{s['failed_root_count']} failed roots, {s['failed_attempt_count']} failed attempts. Root-level failures contribute zero rows to primary SR statistics.", '',
        '| Scene | Human count | Profile | Failure category | Roots |', '|---|---:|---|---|---:|'])
    for r in s['failed_root_groups']:
        lines.append(f"| {r['scene']} | {r['human_count']} | {r['root_profile']} | {r['category']} | {r['count']} |")
    lines.extend(['',
        'Missing assets: MP3D `17DRP5sb8fy.glb` and `coda.glb` fail with `No Stage Attributes exists for requested scene`. ReplicaCAD: unresolved stage/object/URDF assets end with `Stage Template managed object handle `` not found in ManagedContainer, so aborting.` These are grouped by the terminal initialization failure, not double-counted as missing assets. Prefix failures end with `RuntimeError: Prefix terminated before root`; the retained artifacts do not identify a static-versus-human contact cause. Exact log text, path/line references, execution return codes, and per-root profile are in `sr_failed_root_audit.json`.', '',
        '## Consistency and artifacts', '',
        f"**{s['data_consistency']}**: {len(s['contradictions'])} contradictions. Rechecked completion markers and every sealed file for all completed roots, primary/label inventory, global query-ID uniqueness, root/query/branch IDs, fixed-query hashes, labels recomputed from raw truth, durations/control coverage, declared termination evidence, and all SR invariants. Existing collector integrity audits remain PASS (human reactivity is not an integrity gate). No data were silently corrected.", '',
        f"Excluded {s['audit_repeat_branches_excluded']} audit/repeat branches. The full per-root audit contains all 48 roots, including candidate counts and category flags.", '',
        'Artifacts under `datasets/raw/eventwm_prod_v1/audits/`:', '',
        '- `sr_branch_audit.csv` / `.json`: all primary branches, metadata, collision evidence, binary SR/validity, path/query descriptors.',
        '- `sr_root_audit.csv` / `.json`: every completed root and its learning signal.',
        '- `sr_path_variant_audit.json`: every path group and exact query variants/outcomes.',
        '- `sr_failed_root_audit.csv` / `.json`: every failed root with exact evidence.',
        '- `sr_summary.json`: counts, breakdowns, provenance, check counts, contradictions.', '',
        'Reproduce (offline audit only):', '', '```bash',
        '/research-user/miniforge3/envs/eventwm-falcon/bin/python scripts/audit_eventwm_sr.py', '```', ''])
    if s['contradictions']:
        lines.extend(['Contradictions:', '', '```json',json.dumps(s['contradictions'],indent=2),'```',''])
    t=s['total']; r=s['root_signal']
    lines.extend(['## Final summary', '', '```text',
        f"TOTAL PRIMARY BRANCHES: {t['total']}", f"SR-VALID: {t['valid']}",
        f"  SAFE (SR=1): {t['safe']}", f"  COLLISION (SR=0): {t['collision']}",
        f"SR-INVALID / UNKNOWN: {t['invalid']}", '', f"SR-VALID RATE: {100*s['sr_valid_rate']:.2f}%", '',
        'COLLISION BREAKDOWN:', f"  HUMAN: {c['human_only']} (human only; union={c['human_union']})",
        f"  STATIC: {c['static_only']} (static only; union={c['static_union']})",
        f"  HUMAN+STATIC: {c['human_and_static']}", '',
        'INVALID BREAKDOWN: '+', '.join(f'{k}={v}' for k,v in s['invalid_breakdown'].items()), '',
        f"MIXED-SR ROOTS: {r['mixed_sr']}", f"ALL-SAFE ROOTS: {r['all_safe']}",
        f"ALL-COLLISION ROOTS: {r['all_collision']}", f"ROOTS WITH INVALID QUERIES: {r['contains_invalid']}", '',
        f"TIMING/SPEED GROUPS WITH DIFFERENT SR: {s['path_groups_with_different_sr']} / {s['path_groups_with_two_valid_variants']} eligible groups", '',
        'FAILED ROOT TAXONOMY: '+', '.join(f'{k}={v}' for k,v in s['failed_root_taxonomy'].items()), '',
        f"DATA CONSISTENCY: {s['data_consistency']}", '```', ''])
    path.write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
