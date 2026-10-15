"""Freeze and extract a previously unused reserved forecasting bank.

Membership is selected from frozen metadata before forecasts are inspected.
No original reservation, stopped extension, or production file is modified.
The old extension's stronger recovery threshold also gates new raw extraction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.data import SNAPSHOT_HASH,SOURCE_HASH,cache_one
from eventforecastwm.guard import Guard

PROJECT=Path('/home/haodong/workspace/projects/eventwm-falcon')
OLD=PROJECT/'output/eventwm_350k_belief_20261014/cache_pilot128'


def select(blocks,old_config,count):
    excluded=set()
    for split,n in old_config['counts'].items():
        pool=[b for b in blocks if b['split']==split]
        pool.sort(key=lambda b:hashlib.sha256(f"{old_config['seed']}/{b['root']['root_id']}".encode()).hexdigest())
        excluded.update(b['root']['root_id'] for b in (pool if n<0 else pool[:n]))
    banks={};selected=[]
    for split in ('natural_reference','unseen_evaluation'):
        pool=[b for b in blocks if b['split']==split and b['root']['root_id'] not in excluded]
        pool.sort(key=lambda b:hashlib.sha256(('forecast-confirmation-v1/'+b['root']['root_id']).encode()).hexdigest())
        if len(pool)<count:raise ValueError('Insufficient untouched reservation: '+split)
        chosen=pool[:count];banks[split]=[b['root']['root_id'] for b in chosen];selected.extend(chosen)
    return selected,banks,excluded


def prepare(snapshot,output,count=256):
    out=Path(output)
    if out.exists():raise FileExistsError('Confirmation membership is immutable')
    if sha256(snapshot)!=SNAPSHOT_HASH:raise ValueError('Wrong frozen350k snapshot')
    source=read(snapshot);old_config=read(OLD/'config.json')
    chosen,banks,excluded=select(source['blocks'],old_config,count)
    blocks=[]
    for b in chosen:
        r=b['rows'][0];root=b['root']
        blocks.append(dict(root_id=root['root_id'],split=b['split'],scene=r['scene'],category=r['category'],
            episode_id=r['episode_id'],trajectory_id=r['trajectory_id'],archive_path=b['archive_path'],
            seal_path=str(Path(r['dataset'])/root['seal_file']),seal_sha256=b['seal_sha256'],
            complete_sha256=b['complete_sha256'],evidence=b['evidence'],
            rows=[{k:r[k] for k in ('branch_id','query_id','sr_valid','sr_label','observed_duration_s',
                'human_collision_time_valid','human_collision_time_s')} for r in b['rows']]))
    out.mkdir(parents=True);save_json(out/'blocks.json',blocks)
    membership=dict(snapshot_sha256=SNAPSHOT_HASH,blocks_sha256=sha256(out/'blocks.json'),banks=banks,
        excluded_previous_pilot_and_planned_extension=sorted(excluded),
        exclusion_config_sha256=sha256(OLD/'config.json'),selection='fixed root hash, no outcomes or model scores',
        scenes={k:len({b['scene'] for b in blocks if b['split']==k}) for k in banks},
        policy='Freeze model/seed/calibration checkpoint hashes before reading any reserved forecast metrics',
        old_stopped_extension_untouched=True,created_at=time.time())
    save_json(out/'membership.json',membership)
    for p in out.iterdir():p.chmod(0o444)
    print(json.dumps(dict(membership=str(out/'membership.json'),roots={k:len(v) for k,v in banks.items()},scenes=membership['scenes'])))


def recovery(shared):
    original=read(sorted(OLD.glob('preflight_*.json'))[0])['collector']
    evidence=read(shared);last=evidence['samples'][-3:];now=time.time()
    if (evidence['stopped'] or len(last)!=3 or now-last[-1]['timestamp']>120 or
        last[-1]['timestamp']-last[0]['timestamp']<120 or
        any(r['collector_pid']!=original['collector_pid'] or r['state']!='running' or
            r['timestamp']-r['heartbeat_timestamp']>120 or
            r.get('recent_qph') is None or r['recent_qph']<.85*original['qph'] for r in last)):
        raise RuntimeError('Reserved raw extraction requires three sustained recovered10minute windows above85percent original baseline')
    return dict(original_collector=original,shared_guard_sha256=sha256(shared),last_three_recovered_windows=last,
        old_extension_resumed=False,production_actions=[])


def build(protocol_dir,output):
    protocol=Path(protocol_dir);out=Path(output);membership=read(protocol/'membership.json')
    if sha256(protocol/'blocks.json')!=membership['blocks_sha256']:raise ValueError('Changed reserved membership')
    if (out/'manifest.json').exists():raise FileExistsError('Completed confirmation cache')
    out.mkdir(parents=True,exist_ok=True);guard=Guard(out)
    evidence=recovery(out.parent/'cache_full/guard.json')
    save_json(out/f'recovery_{time.time_ns()}.json',evidence)
    identity=dict(membership_sha256=sha256(protocol/'membership.json'),source_hash=SOURCE_HASH,
        confirmation_sha256=sha256(__file__),snapshot_sha256=membership['snapshot_sha256'])
    if (out/'identity.json').exists() and read(out/'identity.json')!=identity:raise ValueError('Exact reserved cache recovery identity changed')
    save_json(out/'identity.json',identity);blocks=read(protocol/'blocks.json');rows=[];started=time.time()
    for b in blocks:
        guard.check()
        if (out/'STOP_RESEARCH').exists():raise RuntimeError('Research-only reserved extraction stop')
        rows.append(cache_one(b,out))
        save_json(out/'progress.json',dict(completed=len(rows),total=len(blocks),seconds=time.time()-started))
    guard.check(True)
    save_json(out/'manifest.json',dict(**identity,roots=rows,banks=membership['banks']))
    for row in rows:(out/row['file']).chmod(0o444)
    (out/'manifest.json').chmod(0o444)
    print(json.dumps(dict(manifest=str(out/'manifest.json'),roots=len(rows),seconds=time.time()-started)))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--snapshot',required=True);a.add_argument('--output',required=True);a.add_argument('--count',type=int,default=256)
    a=sub.add_parser('build');a.add_argument('--protocol-dir',required=True);a.add_argument('--output',required=True)
    a=p.parse_args();kw=vars(a);command=kw.pop('command');(prepare if command=='prepare' else build)(**kw)


if __name__=='__main__':main()
