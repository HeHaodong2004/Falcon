"""Read only public Root seals; no protected or privileged payload access."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import orjson
from eventwm_graph.stage2.data import source_rows
from eventwm_graph.stage2_segment.common import OUT,dump,sha


def audit(row):
    path=Path(row['folder'])/'manifest.json';raw=path.read_bytes();expected=row['files'].get('manifest.json')
    if expected is not None and hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Root seal hash mismatch: '+row['root_id'])
    m=orjson.loads(raw)
    if not m['sealed'] or m['root_id']!=row['root_id'] or m['scene_id']!=row['scene_id'] or m['split']!=row.get('source_split',row['split']) or m['queries']!=row['queries']:
        raise ValueError('Root seal identity mismatch: '+row['root_id'])
    for member in ('legal_root.json.gz','queries.json.gz','outcomes.json.gz','future_observations.json.gz'):
        if m['files'][member]!=row['files'][member]:raise ValueError('Root member identity mismatch')
    return dict(root_id=row['root_id'],scene_id=row['scene_id'],split=row['split'],source_split=m['split'],queries=row['queries'],seal_sha256=sha(path),seal_hash_in_snapshot=expected is not None)


def main():
    train,trainpath=source_rows('train');dev,devpath=source_rows('dev')
    assert len({r['root_id'] for r in train})==len(train) and len({r['root_id'] for r in dev})==len(dev)
    assert not {r['root_id'] for r in train}&{r['root_id'] for r in dev}
    assert not {r['scene_id'] for r in train}&{r['scene_id'] for r in dev}
    dest=OUT/'root_seal_audit_corrected';dest.mkdir(exist_ok=False);tick=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,r in enumerate(pool.map(audit,train+dev,chunksize=64)):
            records.append(r)
            if i%4096==0:print(json.dumps(dict(roots=i+1,total=len(train)+len(dev),seconds=time.perf_counter()-tick)),flush=True)
    import gzip
    with gzip.open(dest/'root_identities.json.gz','wt') as f:json.dump(records,f)
    dump(dest/'complete.json',dict(passed=True,training_roots=len(train),training_scenes=len({r['scene_id'] for r in train}),training_queries=sum(r['queries'] for r in train),
        dev_roots=len(dev),dev_scenes=len({r['scene_id'] for r in dev}),dev_queries=sum(r['queries'] for r in dev),
        original_train_manifest_sha256=sha(trainpath),original_dev_manifest_sha256=sha(devpath),
        sealed_root_hashes_available=sum(r['seal_hash_in_snapshot'] for r in records),protected_payloads_opened=False,privileged_payloads_opened=False,
        source_to_analysis_split_counts=dict(Counter(r['source_split']+'->'+r['split'] for r in records)),
        independent_root_ids_unique=True,scene_and_root_boundaries_disjoint=True,seconds=time.perf_counter()-tick))


if __name__=='__main__':main()
