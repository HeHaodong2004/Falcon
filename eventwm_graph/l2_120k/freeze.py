"""Immutable research copy; the only live database connection is mode=ro."""
import argparse
import collections
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

from eventwm_graph.l2_partial.snapshot import sha


CORE = ('map.py', 'observer.py', 'query.py', 'metrics.py', 'execution.py',
        'eventwm_graph/stage1/runtime.py', 'scripts/collector/observation.py',
        'falcon/additional_action.py', 'falcon/additional_metric.py')


def freeze(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    started = time.time()
    db = sqlite3.connect('file:' + str(source/'campaign.sqlite3') + '?mode=ro', uri=True)
    backup = sqlite3.connect(destination/'journal.sqlite3')
    db.backup(backup, pages=256, sleep=.05)
    db.close()
    assert backup.execute('pragma integrity_check').fetchone()[0] == 'ok'
    jobs = backup.execute("select scene,result from jobs where state='sealed' order by seed").fetchall()
    backup.close()
    old = json.loads(Path('output/l2_partial_graph_wm_explore_20261021/snapshot/manifest.json').read_text())
    reference = old['rows'][0]['versions']
    fresh_path = Path('docs/l2/partial_graph_wm_repair_20261021/artifacts/raw/untouched_candidates.json.gz')
    with gzip.open(fresh_path, 'rt') as f:
        fresh = json.load(f)
    # Previously opened fresh scenes must not silently enter fitting.
    opened = {r['scene_id'] for r in fresh['roots']}
    rows, protected, counts = [], [], collections.defaultdict(collections.Counter)
    sources, contracts, statistics = collections.Counter(), collections.Counter(), collections.Counter()
    for i, (scene, result) in enumerate(jobs):
        rec = json.loads(result)
        src = Path(rec['folder']).resolve()
        if not src.is_relative_to(source/'dataset'/'roots'):
            raise ValueError('sealed root outside campaign storage')
        mh = sha(src/'manifest.json')
        m = json.loads((src/'manifest.json').read_text())
        if not (m['sealed'] is True and m['root_id'] == rec['root_id'] and
                m['scene_id'] == scene and m['future_id'] == m['root_id'] and
                m['schema'] == 'observation_limited_query_conditioned_root_1' and
                m['independent_futures'] == 1 and m['queries'] == len(m['query_ids'])):
            raise ValueError('invalid sealed root identity')
        contract = {k: m['version'][k] for k in CORE}
        if any(contract[k] != reference[k] for k in CORE):
            raise ValueError('incompatible static/sensor/query/executor contract')
        contracts[hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()] += 1
        sources[m['split']] += m['queries']
        statistics.update({k: v for k, v in m['statistics'].items() if isinstance(v, (int, float))})
        h = int(hashlib.sha256(('l2-explore-20261021:' + scene).encode()).hexdigest()[:8], 16) % 100
        split = 'train' if h < 70 else 'dev' if h < 85 else 'holdout'
        reason = None
        if m['split'] != 'train': reason = 'protected_source_' + m['split']
        elif scene in opened: reason = 'previously_opened_fresh_scene'
        elif split == 'holdout': reason = 'protected_research_holdout'
        row = dict(root_id=m['root_id'], scene_id=scene, split=split, source_split=m['split'],
                   queries=m['queries'], query_ids=m['query_ids'], schema=m['schema'],
                   versions=m['version'], collection_config=m['collection_config'],
                   independent_futures=1, source=str(src), manifest_sha256=mh)
        if reason:
            protected.append({**row, 'exclusion': reason})
            counts[reason].update(roots=1, queries=m['queries'])
            continue
        dest = destination/'roots'/m['root_id']
        dest.mkdir(parents=True)
        hashes = {}
        for name, expected in {**m['files'], 'manifest.json': mh}.items():
            if Path(name).name != name or (src/name).is_symlink():
                raise ValueError('noncanonical payload name')
            before = sha(src/name)
            shutil.copyfile(src/name, dest/name)
            copied, after = sha(dest/name), sha(src/name)
            if before != expected or copied != expected or after != expected:
                raise ValueError('sealed payload checksum mismatch')
            if os.stat(src/name).st_ino == os.stat(dest/name).st_ino:
                raise ValueError('mutable inode alias')
            os.chmod(dest/name, 0o444)
            hashes[name] = copied
        if sha(src/'manifest.json') != mh: raise ValueError('manifest changed while copying')
        row['files'] = hashes
        rows.append(row)
        counts[split].update(roots=1, queries=m['queries'])
        if i % 500 == 0:
            print(json.dumps({'scanned': i, 'copied': len(rows), 'seconds': time.time()-started}), flush=True)
    for split in ('train', 'dev'):
        counts[split]['scenes'] = len({r['scene_id'] for r in rows if r['split'] == split})
    assert not ({r['scene_id'] for r in rows if r['split']=='train'} &
                {r['scene_id'] for r in rows if r['split']=='dev'})
    manifest = dict(schema='l2_120k_readonly_snapshot_1', snapshot_version=2,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source=str(source),
        source_roots=len(jobs), source_outcomes=sum(json.loads(v)['outcomes'] for _, v in jobs),
        source_scenes=len({s for s, _ in jobs}), source_split_outcomes=dict(sources),
        source_statistics=dict(statistics), core_contracts=dict(contracts),
        journal_sha256=sha(destination/'journal.sqlite3'), splits=dict(counts),
        split_rule=old['split_rule'], old_manifest_sha256=sha(Path('output/l2_partial_graph_wm_explore_20261021/snapshot/manifest.json')),
        confirmation='NONE; source/protected exclusions never copied; evidence is development only',
        rows=rows, protected=protected, copy_seconds=time.time()-started)
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2))
    for name in ('manifest.json', 'journal.sqlite3'): os.chmod(destination/name, 0o444)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('rows','protected')}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('destination')
    freeze(**vars(p.parse_args()))
