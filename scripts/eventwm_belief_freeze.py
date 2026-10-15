#!/usr/bin/env python3
"""Incremental immutable V4 freeze; reads production, writes research only.

Reuse requires equality of both verified SQLite records and on-disk seals.
New roots use the canonical full-V4 legal/query/label audit. No branch-truth
copy or rescan is performed: its hash is inherited from the deep verifier.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import torch
import eventwm_full_v4 as full
import eventwm_v4_research as r
from eventwm_v4_scaling import fixture
from eventquerywm.inputs import tensorize
from eventquerywm.v2.labels import inspect_social_root, supervision
from collector.core import load_wm_input, digest

PARENT = r.PROJECT/'output/eventwm_full_v4_wm_20261013'
DEFAULT = r.PROJECT/'output/eventwm_350k_belief_snapshot_20261014'


def database_records(path):
    # This helper only reads frozen backup databases, never the live collector.
    # immutable avoids creating WAL/SHM sidecars beside a sealed research copy.
    wal=Path(str(path)+'-wal')
    assert not wal.exists() or wal.stat().st_size == 0, 'Frozen database has uncheckpointed WAL'
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro&immutable=1', uri=True) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert db.execute('SELECT DISTINCT dataset FROM roots').fetchall() == [(str(r.DATA),)]
        roots = {v[0]: v for v in db.execute('SELECT * FROM roots ORDER BY root_id')}
        provenance = {v[0]: v for v in db.execute('SELECT * FROM provenance ORDER BY root_id')}
        examples = defaultdict(list)
        for v in db.execute('SELECT root_id,branch_id,query_id,sr,social FROM examples ORDER BY root_id,branch_id,query_id'):
            examples[v[0]].append(v[1:])
        assert roots.keys() == provenance.keys() == examples.keys()
    return roots, provenance, examples


def protect_completed_snapshot(out):
    """Remove write bits from this completed research snapshot's own files.

    Hashes still establish identity: owner permissions are an accidental-write
    guard, not protection against an owner deliberately replacing a file.
    """
    required=('snapshot.json','verified.sqlite3','audit.json','snapshot_identity.json','cutoff.json')
    assert all((out/name).is_file() for name in required), 'Snapshot is incomplete'
    assert r.read(out/'status.json')['stage'] == 'frozen', 'Snapshot is not finalized'
    for path in out.iterdir():
        if path.is_file() and not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)


def parent_history_evidence(audit):
    """Preserve history evidence across legacy and repeated incremental freezes.

    Parent distributions cannot be silently recombined from summary quantiles.
    Retain their provenance tree instead; new-root measurements remain separate.
    """
    legacy={key:audit[key] for key in ('history_steps','history_span_s','history_dt_s') if key in audit}
    if legacy:
        return dict(legacy_statistics=legacy)
    if 'history_audit' in audit:
        return dict(incremental_parent=audit['history_audit'])
    raise ValueError('Parent snapshot lacks legal history audit evidence')


def evidence_check(root, record, provenance):
    rid, dataset, path, role, scene, seed, count, stats = record
    assert root['root_id'] == rid and root['state'] == 'completed'
    assert root['path'] == path and root['role'] == role
    assert root['physical_scene_id'] == scene and root['seed'] == seed
    # The verifier's stats are a compact outcome census, not pipeline stats.
    assert isinstance(json.loads(stats), dict)
    assert count == 33 and len(root['query_ids']) == count
    _, commit, seal_sha, complete_sha = provenance
    root_path = Path(os.path.abspath(r.DATA/path))
    assert r.sha(r.DATA/root['seal_file']) == seal_sha
    assert r.sha(root_path/'COMPLETE.json') == complete_sha
    seals = r.read(r.DATA/root['seal_file'])
    assert r.read(root_path/'COMPLETE.json') == dict(root_id=rid, seal_digest=digest(seals))
    assert 'branch_truth/branches.jsonl.gz' in seals
    root['source_commit'] = commit
    return root_path, seals


def audit_new(root, record, provenance, examples, fixed_validation):
    path, seals = evidence_check(root, record, provenance)
    rows = inspect_social_root(r.DATA, root)
    assert len(rows) == record[6] == 33
    group = fixture(r.DATA, root)
    split = 'validation' if root['root_id'] in fixed_validation else root['role']
    labels = full.branch_labels(r.read(path/'labels/labels.json'))
    public = {q['query']['query_id']: q['query'] for q in r.read(path/'queries/queries.json')}
    legal = load_wm_input(path, rows[0]['query_id'])
    tensor = tensorize(legal)
    times = np.array([h['timestamp'] for h in legal['history']])
    assert len(times) and np.isfinite(times).all() and (np.diff(times) > 0).all()
    category = root['stats']['legal_state']['category']
    visible = sum(t['visible_now'] and t['valid'] for t in legal['history'][-1]['human_tracks'])
    details = Counter()
    for row in rows:
        label = labels[row['branch_id'], row['query_id']]
        assert label['sr_valid'] and label['sr_label'] == row['sr_label']
        target = supervision(label)
        assert row['sr_label'] in (0, 1) and type(label['sr_label']) is int
        human, static = bool(label['native_human_collision']), bool(label['static_collision'])
        assert (row['sr_label'] == 0) == (human or static)
        if row['sr_label'] == 1:
            assert label['completed'] and label['full_horizon_observed']
            assert label['termination_reason'] == 'query_completed'
        if human:
            assert target['human_collision_time_valid']
            assert 0 <= target['human_collision_time_s'] <= target['observed_duration_s']+1e-5
        q = public[row['query_id']]
        row.update(**group, cohort='V4', split=split, source_commit=root['source_commit'],
                   category=category, human_count=root['human_count'], visible_now=visible,
                   ever_tracks=tensor.humans.shape[0],
                   speed_mps=q['generation_config']['speed']*q['speed_scale'], root_path=str(path))
        details['human_collision'] += human
        details['static_collision'] += static
        details['retained_noncollision_censored'] += not human and not static and not label['completed']
    assert sorted((v['branch_id'], v['query_id'], v['sr_label'], v['outcome_label']) for v in rows) == examples
    primaries = {v['query_id'] for v in rows}
    aliases = [v for (bid, qid), v in labels.items() if not bid.startswith('b') and qid in primaries]
    details.update(diagnostic_alias_labels=len(aliases), invalid_diagnostic_alias_labels=sum(not x['sr_valid'] for x in aliases))
    block = dict(root=root, rows=rows, split=split, cohort='V4', archive_path=str(path),
                 complete_sha256=provenance[3], seal_sha256=provenance[2],
                 evidence={n: seals[n] for n in ('VALIDATED.json', 'legal_observations/history.json',
                     'queries/queries.json', 'labels/labels.json', 'branch_truth/branches.jsonl.gz')})
    return block, dict(details), dict(steps=len(times), span_s=float(times[-1]-times[0]), dt_s=np.diff(times).tolist())


def isolate(blocks, parent, fixed_validation):
    reserved = full.reservation_groups()
    bank = parent['audit']['reserved_selector_bank']
    assert r.sha(bank['index']) == bank['sha256']
    bank_roots = {v['root_id'] for v in r.read(bank['index'])['rows'] if v['split'] == 'validation'}
    held = {b['root']['root_id'] for b in blocks if b['split'] != 'train'} | bank_roots
    identifiers = defaultdict(set)
    for b in blocks:
        if b['rows'][0].get('trajectory_id') in reserved:
            held.add(b['root']['root_id'])
        for field in ('episode_id', 'fixture_id', 'trajectory_id'):
            if b['rows'][0].get(field):
                identifiers[field, b['rows'][0][field]].add(b['root']['root_id'])
    while True:
        size = len(held)
        for ids in identifiers.values():
            if ids & held:
                held.update(ids)
        if len(held) == size:
            break
    for b in blocks:
        if b['split'] == 'train' and b['root']['root_id'] in held:
            b['split'] = 'quarantine_reserved_trajectory'
        for row in b['rows']:
            row['split'] = b['split']
    leakage = {}
    for field in ('root_id', 'query_id', 'episode_id', 'fixture_id', 'trajectory_id'):
        groups = {s: {row[field] for b in blocks if b['split'] == s for row in b['rows'] if row.get(field)} for s in full.SPLITS}
        leakage[field] = {a+'/'+b: len(groups[a] & groups[b]) for i, a in enumerate(full.SPLITS) for b in full.SPLITS[i+1:]}
        assert not any(leakage[field].values()), (field, leakage[field])
    qids = [v['query_id'] for b in blocks for v in b['rows']]
    assert len(qids) == len(set(qids))
    train = [b for b in blocks if b['split'] == 'train']
    assert not reserved & {b['rows'][0].get('trajectory_id') for b in train}
    assert not bank_roots & {b['root']['root_id'] for b in train}
    unseen = {b['root']['physical_scene_id'] for b in blocks if b['split'] == 'unseen_evaluation'}
    assert not unseen & {b['root']['physical_scene_id'] for b in train}
    assert fixed_validation == {b['root']['root_id'] for b in blocks if b['split'] == 'validation'}
    splits = {s: dict(roots=sum(b['split'] == s for b in blocks), queries=sum(len(b['rows']) for b in blocks if b['split'] == s),
                     scenes=len({b['root']['physical_scene_id'] for b in blocks if b['split'] == s})) for s in sorted({b['split'] for b in blocks})}
    return train, splits, leakage, bank, reserved


def freeze(args):
    out = args.output.resolve()
    assert out != r.DATA and r.DATA not in out.parents and out != args.parent.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'snapshot.json').exists(), 'Frozen snapshot already exists; never overwrite'
    torch.set_num_threads(1)
    started = time.time()
    parent_identity = r.read(args.parent/'snapshot_identity.json')
    assert r.sha(args.parent/'snapshot.json') == parent_identity['sha256']
    assert r.sha(args.parent/'verified.sqlite3') == parent_identity['source_index_sha256']
    parent = r.read(args.parent/'snapshot.json')
    previous = {b['root']['root_id']: b for b in parent['blocks']}
    old_roots, old_provenance, old_examples = database_records(args.parent/'verified.sqlite3')
    if not (out/'verified.sqlite3').exists():
        before = r.production_sample()
        r.save(out/'production_before.json', before)
        with r.ro(args.index) as source, sqlite3.connect(out/'verified.sqlite3') as target:
            source.execute('BEGIN')
            source.execute('SELECT count(*) FROM roots').fetchone()
            source.backup(target)
            # A standalone backup should not retain the source WAL policy.
            target.execute('PRAGMA journal_mode=DELETE')
        r.save(out/'cutoff.json', dict(timestamp=before['timestamp'], source_index=str(args.index),
            frozen_index_sha256=r.sha(out/'verified.sqlite3'), parent_snapshot_sha256=parent_identity['sha256']))
    cutoff = r.read(out/'cutoff.json')
    assert cutoff['parent_snapshot_sha256'] == parent_identity['sha256']
    assert cutoff['frozen_index_sha256'] == r.sha(out/'verified.sqlite3')
    before = r.read(out/'production_before.json')
    verified, provenance, examples = database_records(out/'verified.sqlite3')
    assert previous.keys() <= verified.keys(), 'Previously audited roots disappeared'
    with r.ro(r.DATA/'pipeline.sqlite3') as db:
        db.execute('BEGIN')
        roots = {rid: json.loads(db.execute('SELECT value FROM roots WHERE id=?', (rid,)).fetchone()[0]) for rid in verified}
    fixed_validation = {rid for rid, b in previous.items() if b['split'] == 'validation'}
    guard = full.WMGuard(out)
    blocks, excluded, new_history = [], [], []
    details = Counter()
    reused = newly_audited = 0
    print(json.dumps(dict(stage='cutoff', verified_roots=len(verified), verified_queries=sum(v[6] for v in verified.values()),
                         parent_roots=len(previous), fixed_validation_roots=len(fixed_validation), disk_free=shutil.disk_usage(out).free)), flush=True)
    journal = out/'incremental_audit.jsonl'
    cached = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                v = json.loads(line)
                cached[v['block']['root']['root_id']] = v
            except ValueError:
                raise ValueError('Incomplete audit journal; preserve it and use a new output')
    with journal.open('a') as stream:
        for number, (rid, record) in enumerate(verified.items()):
            guard.check()
            if (out/'STOP_RESEARCH').exists():
                raise RuntimeError('Research-only stop requested; cutoff and journal retained')
            try:
                root = roots[rid]
                path, seals = evidence_check(root, record, provenance[rid])
                prior = previous.get(rid)
                reuse = (prior is not None and old_roots[rid] == record and old_provenance[rid] == provenance[rid]
                         and old_examples[rid] == examples[rid] and prior['seal_sha256'] == provenance[rid][2]
                         and prior['complete_sha256'] == provenance[rid][3]
                         and all(seals[k] == value for k, value in prior['evidence'].items()))
                if reuse:
                    b = prior
                    assert sorted((v['branch_id'], v['query_id'], v['sr_label'], v['outcome_label']) for v in b['rows']) == examples[rid]
                    reused += 1
                else:
                    if rid in cached:
                        c = cached[rid]
                        b, d, h = c['block'], c['details'], c['history']
                        assert b['seal_sha256'] == provenance[rid][2] and b['complete_sha256'] == provenance[rid][3]
                    else:
                        b, d, h = audit_new(root, record, provenance[rid], examples[rid], fixed_validation)
                        stream.write(json.dumps(dict(block=b, details=d, history=h), allow_nan=False)+'\n')
                        stream.flush()
                    details.update(d)
                    new_history.append(h)
                    newly_audited += 1
                blocks.append(b)
            except (OSError, ValueError, AssertionError, KeyError) as exc:
                import traceback
                frame = traceback.extract_tb(exc.__traceback__)[-1]
                excluded.append(dict(root_id=rid, queries=record[6], reason=repr(exc), check=frame.line))
            if (number+1) % 100 == 0:
                status = dict(stage='incremental_audit', processed=number+1, total=len(verified), reused=reused,
                              newly_audited=newly_audited, excluded=len(excluded), seconds=time.time()-started)
                r.save(out/'status.json', status)
                print(json.dumps(status), flush=True)
    train, splits, leakage, bank, reserved = isolate(blocks, parent, fixed_validation)
    after = r.production_sample()
    elapsed = after['timestamp']-before['timestamp']
    rate = (after['progress']['primary_queries']-before['progress']['primary_queries'])/elapsed*3600 if elapsed else None
    all_rows = [v for b in blocks for v in b['rows']]
    sr = Counter(v['sr_label'] for v in all_rows)
    if not excluded:
        assert sr == Counter(v[2] for values in examples.values() for v in values)
    counts = dict(queries=dict(safe=sr[1], collision=sr[0]),
                  roots=dict(zero_visible_now=sum(b['rows'][0]['visible_now'] == 0 for b in blocks),
                             never_detected_humans=sum(b['rows'][0]['ever_tracks'] == 0 for b in blocks)),
                  categories=dict(Counter(v['category'] for v in all_rows)),
                  human_counts=dict(Counter(str(v['human_count']) for v in all_rows)))
    audit = dict(scope='Incremental immutable verified V4 snapshot; no live index consumed by training',
        source_commit=full.source_sha(), script_sha256=r.sha(Path(__file__)), cutoff=cutoff['timestamp'],
        source_index=str(args.index), frozen_index_sha256=cutoff['frozen_index_sha256'],
        parent_snapshot=str(args.parent/'snapshot.json'), parent_snapshot_sha256=parent_identity['sha256'],
        source_finalized_queries=sum(v[6] for v in verified.values()), source_finalized_roots=len(verified),
        retained_queries=len(all_rows), retained_roots=len(blocks), reused_roots=reused, independently_audited_new_roots=newly_audited,
        live_collection_at_cutoff=before, live_collection_after=after, publication_sample_qph=rate,
        splits=splits, counts=counts, leakage=leakage, unique_scenes=len({b['root']['physical_scene_id'] for b in blocks}),
        include_v3=False, invalid_exclusions=excluded, invalid_excluded_queries=sum(v['queries'] for v in excluded),
        reserved_selector_bank=bank, reserved_navigation_trajectories=len(reserved), reserved_navigation_overlap=0,
        reference_validation_unchanged=True, fixed_validation_roots=len(fixed_validation), unseen_scene_training_overlap=0,
        duration_s=full.distribution([v['planned_duration_s'] for v in all_rows]),
        observed_duration_s=full.distribution([v['observed_duration_s'] for v in all_rows]),
        commanded_speed_mps=full.distribution([v['speed_mps'] for v in all_rows]),
        new_root_audit_counts=dict(details), parent_audit_counts=parent['audit']['counts'],
        history_audit=dict(new_roots=new_history, reused_parent_history_statistics=parent_history_evidence(parent['audit'])),
        label_join='(branch_id,query_id); canonical primary branches, exact verified-example parity',
        reuse_evidence='Exact old/new SQLite root, provenance and example records; current SHA256 of seal and COMPLETE; matching evidence hashes',
        new_evidence='Canonical inspect_social_root hashes legal history/query/labels/VALIDATED; validates primary index, query digests, event times and censoring',
        truth_integrity='Branch truth content hashes inherited from independently verified immutable seals; raw branch payloads not rescanned or copied',
        source_epochs=dict(Counter(b['root']['source_commit'] for b in blocks)), raw_copied=False,
        static_capability=parent['audit']['static_capability'], elapsed_seconds=time.time()-started,
        future_target_duration_coverage={str(h):sum(v['observed_duration_s'] >= h for v in all_rows) for h in (1,3,5)})
    ordered = sorted([b['root']['root_id'] for b in train], key=r.key)
    r.save(out/'snapshot.json', dict(version='eventwm-v4-research-snapshot-v1', blocks=blocks, audit=audit,
                                   subsets={'all':ordered, '30k':ordered[:909]}))
    identity = dict(sha256=r.sha(out/'snapshot.json'), source_index_sha256=cutoff['frozen_index_sha256'],
                    parent_snapshot_sha256=parent_identity['sha256'])
    r.save(out/'snapshot_identity.json', identity)
    r.save(out/'audit.json', audit)
    r.save(out/'status.json', dict(stage='frozen', identity=identity, splits=splits, excluded=excluded))
    protect_completed_snapshot(out)
    print(json.dumps(dict(stage='frozen', identity=identity, splits=splits, excluded=excluded)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT)
    p.add_argument('--parent', type=Path, default=PARENT)
    p.add_argument('--index', type=Path, default=full.INDEX)
    freeze(p.parse_args())


if __name__ == '__main__':
    main()
