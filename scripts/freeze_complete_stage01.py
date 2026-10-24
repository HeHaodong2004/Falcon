"""Preserve ALL stopped collection splits, copying protected bytes opaquely.

Fitting Root copies are reused by checksum; all excluded/protected Roots get
independent read-only copies. This administrative snapshot is never a training
manifest and never deserializes any protected payload.
"""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'output/research_transition_20261024'
SOURCE=Path('/research-user/workspace/projects/eventwm-falcon-acceleration-v2/output/stage01_acceleration_v2_20261021/collection_user_authorized_5s_20261021_1')
FINAL=BASE/'output/l2_two_stage_20261023/final_collection_20261024/manifest.json'
DEST=OUT/'final_collection_all_splits'
FITTING=None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def status():
    s=json.loads(subprocess.check_output([str(Path.home()/'.local/bin/collectorctl'),'status'],text=True))
    assert s['health']['collection_status']==s['campaign']['collection_status']=='MANUALLY_STOPPED'
    assert s['service']['ActiveState']=='inactive' and s['service']['UnitFileState']=='disabled'
    assert s['service']['MainPID']=='0' and not s['campaign']['worker_pids'] and s['campaign']['active_worker_count']==0
    return s


def copy(job):
    seed,scene,raw=job;r=json.loads(raw);source=Path(r['folder']);m=json.loads((source/'manifest.json').read_text());rid=m['root_id']
    assert m['sealed'] and m['scene_id']==scene and m['root_id']==m['future_id'] and m['independent_futures']==1
    assert m['queries']==r['outcomes'] and m['schema']=='observation_limited_query_conditioned_root_1'
    expected={**m['files'],'manifest.json':sha(source/'manifest.json')}
    reused=rid in FITTING
    if reused:
        target=Path(FITTING[rid]['folder']);assert expected==FITTING[rid]['files']
    else:
        target=DEST/'roots'/rid;target.mkdir(parents=True,exist_ok=True)
        for name,digest in expected.items():
            assert Path(name).name==name and not (source/name).is_symlink()
            output=target/name
            if output.exists():
                assert sha(output)==digest
            else:
                temporary=target/(name+'.tmp');shutil.copyfile(source/name,temporary)
                assert sha(temporary)==digest and sha(source/name)==digest
                temporary.chmod(0o444);temporary.replace(output)
            assert (source/name).stat().st_ino!=output.stat().st_ino and output.stat().st_mode&0o222==0
        target.chmod(0o555)
    return dict(root_id=rid,scene_id=scene,seed=seed,source_split=m['split'],queries=m['queries'],folder=str(target),source_folder=str(source),
        fitting_eligible=reused,files=expected,protected_payloads_deserialized=False)


def main():
    global FITTING
    OUT.mkdir(parents=True,exist_ok=True);before=status();start=time.perf_counter()
    fitting=json.loads(FINAL.read_text());FITTING={r['root_id']:r for r in fitting['rows']}
    DEST.mkdir(exist_ok=True);index=DEST/'rows.jsonl.gz'
    if (DEST/'manifest.json').exists():raise RuntimeError('Complete collection already frozen; use its manifest')
    db=sqlite3.connect('file:'+str(SOURCE/'campaign.sqlite3')+'?mode=ro',uri=True)
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    jobs=db.execute("SELECT seed,scene,result FROM jobs WHERE state='sealed' ORDER BY seed").fetchall();db.close()
    counts={};scenes={};roots=outcomes=copied=0
    with (DEST/'rows.jsonl.gz.tmp').open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as stream:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for row in pool.map(copy,jobs):
                    roots+=1;outcomes+=row['queries'];copied+=not row['fitting_eligible'];split=row['source_split']
                    counts.setdefault(split,dict(roots=0,outcomes=0));counts[split]['roots']+=1;counts[split]['outcomes']+=row['queries']
                    scenes.setdefault(split,set()).add(row['scene_id']);stream.write((json.dumps(row,separators=(',',':'))+'\n').encode())
                    if roots%10000==0:print(json.dumps(dict(roots=roots,total=len(jobs),new_root_copies=copied,seconds=time.perf_counter()-start)),flush=True)
    (DEST/'rows.jsonl.gz.tmp').replace(index);index.chmod(0o444)
    after=status();assert before['campaign']==after['campaign'] and roots==after['campaign']['roots'] and outcomes==after['campaign']['outcomes']
    for split in counts:counts[split]['scenes']=len(scenes[split])
    manifest=dict(schema='final_complete_stage01_administrative_snapshot_1',immutable=True,training_manifest=False,
        roots=roots,outcomes=outcomes,scenes=len(set().union(*scenes.values())),splits=counts,
        rows=dict(path=str(index),sha256=sha(index)),new_excluded_root_copies=copied,reused_independent_fitting_copies=roots-copied,
        fitting_manifest=str(FINAL),fitting_manifest_sha256=sha(FINAL),
        historical_fitting_supplement='Retained separately in fitting manifest; source collection counts above exclude historical supplement.',
        source_journal_sha256=sha(SOURCE/'campaign.sqlite3'),collector_before=before,collector_after=after,
        protected_payloads_deserialized=False,source_sha256=sha(__file__),seconds=time.perf_counter()-start)
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(DEST/'manifest.json').chmod(0o444)
    (DEST/'roots').chmod(0o555);DEST.chmod(0o555)
    print(json.dumps({k:manifest[k] for k in ('roots','outcomes','scenes','new_excluded_root_copies','seconds')}),flush=True)


if __name__=='__main__':main()
