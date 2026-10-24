"""Reuse byte-identical audited parent witnesses; transform only new Roots.

The existing positive_history.chunk and last_event_metadata.chunk functions do
all scientific transformations. This is manifest/cache execution plumbing only.
"""
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
from eventwm_graph.l2_two_stage import positive_history as ph,last_event_metadata as em
from eventwm_graph.l2_two_stage.common import OUT,dump,sha

CACHE='train_final_collection_20261024'
PARENT='train_campaign_20261023'


def chunk(bounds):return ph.chunk(bounds),em.chunk(bounds)


def main(resume_pid):
    prepared_path=OUT/'prepared'/CACHE/'manifest.json'
    while not prepared_path.exists():time.sleep(5)
    prepared=json.loads(prepared_path.read_text());snapshot_path=Path(prepared['snapshot'])
    snapshot=json.loads(snapshot_path.read_text());expected={r['root_id'] for r in snapshot['rows']}
    assert sha(snapshot_path)==prepared['identity']['snapshot_sha256']
    assert len(expected)==len(snapshot['rows'])==prepared['roots']
    assert all(r['split']=='train' and r['source_split']=='train' for r in snapshot['rows'])
    manifests={};parent_roots=None
    for folder,module in [('positive_history',ph),('last_event_metadata',em)]:
        path=OUT/folder/PARENT/'manifest.json';m=json.loads(path.read_text())
        assert m['schema']==module.SCHEMA and m['identity']['source_sha256']==sha(module.__file__)
        roots={r['root_id'] for shard in m['shards'] for r in shard['roots']}
        assert len(roots)==sum(len(s['roots']) for s in m['shards'])==m['roots'] and roots.issubset(expected)
        if parent_roots is None:parent_roots=roots
        else:assert roots==parent_roots
        manifests[folder]=(path,m)
    new=expected-parent_roots;anchors=collections.defaultdict(list)
    parent_prepared=json.loads((OUT/'prepared'/PARENT/'manifest.json').read_text())
    assert prepared['shards'][:len(parent_prepared['shards'])]==parent_prepared['shards']
    old_shards={s['path'] for s in parent_prepared['shards']}
    for shard in prepared['shards']:
        if shard['path'] in old_shards:continue
        raw=(OUT/shard['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==shard['sha256']
        for row in np.load(io.BytesIO(raw),allow_pickle=True):
            assert row['root_id'] in new
            anchors[row['root_id']].append({k:row[k] for k in ('anchor_index','timestamp','query_id','scene_id','keys')})
    assert set(anchors)==new
    jobs=[(row,anchors[row['root_id']]) for row in snapshot['rows'] if row['root_id'] in new]
    for folder,module in [('positive_history',ph),('last_event_metadata',em)]:
        path,parent=manifests[folder];dest=OUT/folder/CACHE;dest.mkdir(parents=True,exist_ok=True)
        identity=dict(schema=module.SCHEMA,prepared_manifest_sha256=sha(prepared_path),snapshot_manifest_sha256=sha(snapshot_path),
            source_sha256=sha(module.__file__),parent_manifest=str(path),parent_manifest_sha256=sha(path),
            extension_script_sha256=sha(__file__),reuse='Exact original Root/anchor payloads; unchanged transform function and source hashes')
        if (dest/'preparation_identity.json').exists():assert json.loads((dest/'preparation_identity.json').read_text())==identity
        else:dump(dest/'preparation_identity.json',identity)
        module.JOBS=jobs;module.DEST=dest;module.IDENTITY=identity
    em.PRIMARY=ph.DEST
    start=time.perf_counter();reports=[[],[]]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for a,b in pool.map(chunk,[(i,min(i+200,len(jobs))) for i in range(0,len(jobs),200)]):
            reports[0].append(a);reports[1].append(b)
            print(json.dumps(dict(new_roots=sum(len(r['roots']) for r in reports[0]),new_total=len(jobs),seconds=time.perf_counter()-start)),flush=True)
    for index,(folder,module) in enumerate([('positive_history',ph),('last_event_metadata',em)]):
        parent_path,parent=manifests[folder];counts=collections.Counter(parent['counts'])
        shards=list(parent['shards']);seen=set(parent_roots)
        for report in reports[index]:
            counts.update(report['counts'])
            for row in report['roots']:
                assert row['root_id'] not in seen;seen.add(row['root_id'])
            keys=('path','sha256','roots','counts') if index==0 else ('path','sha256','primary_shard_path','primary_shard_sha256','roots','counts')
            shards.append({k:report[k] for k in keys})
        assert seen==expected and counts['roots']==prepared['roots'] and counts['anchors']==prepared['counts']['anchors']
        value=dict(schema=module.SCHEMA,split='train',roots=len(seen),scenes=prepared['scenes'],counts=dict(counts),
            identity=module.IDENTITY,prepared_manifest=str(prepared_path),shards=shards,failures=[],seconds=time.perf_counter()-start,
            parent_manifest=str(parent_path),parent_manifest_sha256=sha(parent_path),parent_identity=parent['identity'],new_roots=len(jobs),
            transformation='Existing module.chunk without scientific modification; parent payloads referenced unchanged by checksum')
        if index:value.update(primary_manifest=str(ph.DEST/'manifest.json'),primary_manifest_sha256=sha(ph.DEST/'manifest.json'))
        dump(module.DEST/'manifest.json',value)
        for p in module.DEST.iterdir():
            if p.is_file():p.chmod(0o444)
        module.DEST.chmod(0o555)
    receipt=OUT.parents[0]/'research_transition_20261024/sidecar_extension_complete.json'
    dump(receipt,dict(parent_roots=len(parent_roots),new_roots=len(jobs),total_roots=len(expected),script_sha256=sha(__file__),
        scientific_modules={Path(m.__file__).name:sha(m.__file__) for m in (ph,em)},resumed_orchestrator_pid=resume_pid))
    args=(Path('/proc')/str(resume_pid)/'cmdline').read_bytes()
    assert b'scripts/final_stage1_research.py' in args
    os.kill(resume_pid,signal.SIGCONT)


if __name__=='__main__':main(int(sys.argv[1]))
