#!/usr/bin/env python3
"""Read-only verification of a stopped V3 mass collection and its retained index."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from collector.core import digest, read
from collector.v3 import social_outcome_labels
from collector.branch_storage import iter_branches, branch_path
from eventwm_mass_collection import aggregate, hashes, inspect_root
from eventwm_scale import verify_root


def verify_one(arguments):
    output, root, m = arguments
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    ids = set()
    counts = Counter()
    audited_roots = 0
    path = output / root['path']
    try:
        verify_root(output, dict(root, seals=read(output / root['seal_file'])), deep=True)
        stats, retained = inspect_root(path, root)
        for key in stats.keys() - {'bytes'}:
            require(stats[key] == root['stats'][key], f"{root['root_id']}: stats mismatch {key}")
        provenance = read(path / 'audits/provenance.json')
        require(provenance['git_head'] == m['frozen_commit'], f"{root['root_id']}: commit mismatch")
        require(all(m['source_hashes'].get(k) == v for k, v in provenance['source_sha256'].items()),
                f"{root['root_id']}: collector source mismatch")
        queries = {q['query']['query_id']: q for q in read(path / 'queries/queries.json')}
        require(branch_path(path).name == 'branches.jsonl.gz', f"{root['root_id']}: mixed branch encoding")
        rows = list(iter_branches(path))
        by_branch = {b['branch_id']: b for b in rows}
        expected_branches = {f'b{i:03}' for i in range(33)} | {'reverse_0', 'reverse_1', 'truncated_wait'}
        require(len(rows) == 36 and set(by_branch) == expected_branches, f"{root['root_id']}: branch inventory")
        fingerprint = read(path / 'roots/privileged_root.json')['sha256']
        require(all(b['root_fingerprint'] == fingerprint for b in rows), f"{root['root_id']}: canonical root fingerprint")
        derived = [social_outcome_labels(b, queries[b['query_id']]['query'], root['human_count']) for b in rows]
        require(derived == read(path / 'labels/labels.json'), f"{root['root_id']}: label recomputation")
        require(len({b['root_fingerprint'] for b in rows}) == 1, f"{root['root_id']}: root replay")
        for i in (0, 1):
            a, b = by_branch[f'b{i:03}'], by_branch[f'reverse_{i}']
            for key in ('truth', 'controls', 'legal_observations'):
                require(digest(a[key]) == digest(b[key]), f"{root['root_id']}: reverse replay {key}")
        for branch, label in zip(rows, derived):
            query = queries[branch['query_id']]
            require(branch['query_hash_before'] == branch['query_hash_after'] == query['sha256'] == digest(query['query']),
                    f"{root['root_id']}: branch query immutability")
            require(query['query']['root_id'] == root['root_id'], f"{root['root_id']}: query root provenance")
            collision = label['native_robot_human_contact'] or label['static_contact']
            complete = branch['completed'] and label['full_horizon_observed'] and branch['termination_reason'] == 'query_completed'
            require(label['sr_valid'] == bool(collision or complete), f"{root['root_id']}: SR validity")
            require(label['sr_label'] == (int(not collision) if collision or complete else None), f"{root['root_id']}: SR value")
            if branch['branch_id'].startswith('b'):
                if collision and not branch['completed']:
                    counts['retained_early_collision'] += int(label in retained)
                elif not collision and not complete:
                    counts['excluded_noncollision_early'] += 1
                    require(label not in retained, f"{root['root_id']}: retained unrelated early termination")
            else:
                counts['audit_branches'] += 1
            for row in branch['truth']:
                require(row['contacts'] == sorted(row['contacts'], key=lambda c: json.dumps(c, sort_keys=True, allow_nan=False)),
                        f"{root['root_id']}: unsorted contact serialization")
        shard = read(output / 'primary_indices' / f"{root['root_id']}.json")
        require(shard['samples'] == retained, f"{root['root_id']}: retained index mismatch")
        require(shard['role'] == root['role'] and shard['split_group'] == root['root_id'], f"{root['root_id']}: provenance split")
        current = {l['query_id'] for l in retained}
        require(not ids & current, f"{root['root_id']}: duplicate retained queries")
        ids.update(current)
        require(sorted(current) == root['query_ids'], f"{root['root_id']}: retained IDs mismatch")
        audited_roots += 1
        print(json.dumps({'verified_root': root['root_id'], 'retained_queries': len(retained)}), flush=True)
    except Exception as exc:
        errors.append(f"{root['root_id']}: {type(exc).__name__}: {exc}")
    return dict(root_id=root['root_id'], errors=errors, ids=ids, counts=counts, audited_roots=audited_roots)


def check(output, final=False, workers=4):
    m = read(output / 'manifest.json')
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    require(m['source_hashes'] == hashes(), 'current source hashes differ from manifest')
    require(m.get('schema_version') == 3 and m.get('branch_truth_encoding') == 'jsonl-gzip-v1', 'expected lossless binary SR storage schema v3')
    require(m.get('sr_contract_version') == 'binary-execution-safety-v1', 'SR version mismatch')
    require(m.get('contact_serialization') == 'sorted-exact-multiset-v1', 'contact format mismatch')
    require(not any(r['state'] == 'running' for r in m['roots']), 'workers remain running')
    require(not m['state'].startswith('global_'), 'unresolved global failure')
    recomputed = aggregate(m)
    require(read(output / 'aggregate.json') == recomputed, 'aggregate differs from manifest')
    ids = set()
    counts = Counter()
    audited_roots = 0
    roots = [root for root in m['roots'] if root['state'] == 'completed']
    source = {k: m[k] for k in ('frozen_commit', 'source_hashes')}
    arguments = [(output, root, source) for root in roots]
    from multiprocessing import Pool
    with Pool(processes=workers) as pool:
        for result in pool.imap_unordered(verify_one, arguments):
            errors.extend(result['errors'])
            require(not ids & result['ids'], f"{result['root_id']}: duplicate retained queries across roots")
            ids.update(result['ids'])
            counts.update(result['counts'])
            audited_roots += result['audited_roots']
    require(audited_roots > 0 and len(ids) > 0, 'no verified retained data')
    require(len(ids) == recomputed['primary_queries'], 'retained count mismatch')
    require(recomputed['static_violations'] == 0, 'static violations')
    if final:
        require(len(ids) >= max(300000, m['target_primary_queries']), 'target not reached')
        require(m['state'] == 'target_reached', 'run not at target')
        for key in ('families', 'human_counts', 'roles', 'root_categories', 'visible', 'lost', 'class_counts', 'binary_targets', 'sr_targets', 'variants'):
            require(bool(recomputed[key]), f'empty {key}')
        require(set(recomputed['human_counts']) >= {'0', '1', '2', '4', '6'}, 'missing human-count stratum')
        require(set(recomputed['sr_targets']) == {'0', '1'}, 'missing SR class')
        require(set(recomputed['root_categories']) >= {'zero', 'unseen', 'visible', 'lost', 'visible_lost'}, 'missing root category')
        require(counts['retained_early_collision'] > 0, 'no retained early collisions')
        require(recomputed['completed_scenes'] > 1, 'no broad scene coverage')
    return dict(status='PASS' if not errors else 'FAIL', final_gate=final, dataset=str(output.resolve()),
                commit=m['frozen_commit'], source_hashes=m['source_hashes'], audited_roots=audited_roots,
                retained_queries=len(ids), evidence_counts=dict(counts), aggregate=recomputed, errors=errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--workers', type=int, choices=range(1, 17), default=4)
    args = parser.parse_args()
    result = check(args.output, args.final, args.workers)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    sys.exit(result['status'] != 'PASS')
