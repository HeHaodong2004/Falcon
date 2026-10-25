"""Append causal features to the exact immutable baseline sampled supervision.

Avoid replaying billions of already validated static updates. Every original
sample x47/y/height/weight is retained. Sample keys are uniquely recovered from
quantized relative coordinates AND the complete x47 is rechecked on every row.
Only legally given initial map and actual executed robot poses enter new inputs.
Canonical JSON fast readers skip static/camera payloads; unknown format falls
back to the standard decoder. Checksums are verified before either reader.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from numba import njit
from eventwm_graph.stage2.data import read_member,branches
from .features import PreparedGraph,Features,opportunity
from .data import sample
from .common import OUT,ORIGINAL,identity,dump,sha,snapshot

STATIC_ARRAY=re.compile(rb'"static_updates"\s*:\s*\[')

@njit(cache=True)
def array_ends(data,starts):
    ends=np.empty(len(starts),np.int64)
    for j in range(len(starts)):
        i=starts[j];depth=0;quoted=False;escaped=False;found=False
        while i<len(data):
            c=data[i]
            if quoted:
                if escaped:escaped=False
                elif c==92:escaped=True
                elif c==34:quoted=False
            elif c==34:quoted=True
            elif c==91:depth+=1
            elif c==93:
                depth-=1
                if depth==0:ends[j]=i+1;found=True;break
            i+=1
        if not found:raise ValueError('unbalanced static-update JSON array')
    return ends


def without_static_arrays(raw):
    """Remove unused static payloads while preserving the complete JSON tree.

    This is a decoder optimization for checksummed, already sealed/validated
    records, not a replacement for the original scientific validation rules.
    JSON strings/escapes are respected by the compiled balanced-array scanner.
    Every remaining key/value is decoded by the standard JSON implementation.
    """
    starts=np.asarray([m.end()-1 for m in STATIC_ARRAY.finditer(raw)],np.int64)
    ends=array_ends(np.frombuffer(raw,np.uint8),starts);parts=[];cursor=0
    for start,end in zip(starts,ends):
        if start<cursor:continue
        parts.extend((raw[cursor:start],b'[]'));cursor=int(end)
    parts.append(raw[cursor:]);return b''.join(parts)

ROWS=None
SHARDS=None
SOURCE=None
DEST=None


def payload(folder,name,expected):
    data=(Path(folder)/name).read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('immutable source checksum mismatch: '+name)
    return gzip.decompress(data)


def legal_input(row):
    # Decode the full legal Root with the original JSON reader. Nested history
    # track fields must never substitute for the actual initial robot pose.
    data=read_member(row['folder'],'legal_root.json.gz',row['files']['legal_root.json.gz'])
    graph=data['partial_graph'];start=data['history'][-1]
    if float(data['sensor']['max_range_m'])>10.00001:raise ValueError('unreviewed sensor')
    if float(start['timestamp'])!=float(graph['timestamp']):raise ValueError('legal Root clock mismatch')
    return graph,[float(start['timestamp']),*start['robot_pose']],False


def executed_inputs(row):
    queries=read_member(row['folder'],'queries.json.gz',row['files']['queries.json.gz'])
    outcomes=read_member(row['folder'],'outcomes.json.gz',row['files']['outcomes.json.gz'])
    raw=payload(row['folder'],'future_observations.json.gz',row['files']['future_observations.json.gz'])
    fallback=False
    try:nodes=json.loads(without_static_arrays(raw))
    except (ValueError,TypeError):nodes=json.loads(raw);fallback=True
    poses=[v['observation']['robot_pose'] for v in nodes];times=[float(v['observation']['timestamp']) for v in nodes];parents=[v['parent'] for v in nodes]
    if any(v['node_id']!=i for i,v in enumerate(nodes)):raise ValueError('node identity mismatch')
    if len(queries)!=len(outcomes) or len(queries)!=row['queries']:raise ValueError('Query cardinality mismatch')
    result=[]
    for q,o in zip(queries,outcomes):
        if q['query_id']!=o['query_id']:raise ValueError('Query identity mismatch')
        path=[];parent=None
        for node in o['node_ids']:
            if parents[node]!=parent:raise ValueError('executed ancestry mismatch')
            parent=node;path.append([times[node],*poses[node]])
        result.append((q['query_id'],path))
    return result,fallback


def recover_keys(x,start,resolution):
    encoded=x[:,36:38];decoded=encoded.astype(np.float32);initial=np.asarray(start,np.float32)[[1,3]]
    upper=np.nextafter(encoded,np.full(encoded.shape,np.inf,np.float16)).astype(np.float64)
    lower=np.nextafter(encoded,np.full(encoded.shape,-np.inf,np.float16)).astype(np.float64)
    error=5*np.maximum(upper-decoded,decoded-lower)+32*np.finfo(np.float32).eps*(np.abs(initial)+10*np.abs(decoded)+2)
    if not np.all(error<.49*resolution):raise ValueError('nonunique quantized key inversion')
    xy=decoded*10+initial;return np.floor(xy/resolution).astype(np.int32)


def augment_root(row,old):
    graph,start,legal_fallback=legal_input(row);paths,dag_fallback=executed_inputs(row);prepared=PreparedGraph(graph)
    n=len(old['y']);out=np.zeros((n,92),np.float16);out[:,:47]=old['x'];cursor=0;query_meta=[]
    try:
        for query,path in paths:
            begin=cursor
            while cursor<n and old['y'][cursor]>0 and cursor-begin<64:cursor+=1
            if cursor+64>n or np.any(old['y'][cursor:cursor+64]!=0):raise ValueError('unsupported frozen stratum layout')
            cursor+=64;stop=cursor;trajectory=np.asarray([start]+path,np.float64)
            context=Features(graph,trajectory,prepared);keys=recover_keys(old['x'][begin:stop],start,float(graph['resolution']))
            # Complete legacy feature identity, on ALL rows, resolves any tie,
            # quantization or Query alignment ambiguity before new features.
            legacy,_=context.transform(keys,'baseline')
            if not np.array_equal(legacy[:,:47].astype(np.float16),old['x'][begin:stop]):raise ValueError('legacy feature/Query alignment mismatch')
            known=set(map(tuple,prepared.keys))
            if any(tuple(k) in known for k in keys):raise ValueError('recovered sample was initially known')
            keep,view=opportunity(keys,trajectory,float(graph['resolution']))
            if not np.all(keep):raise ValueError('recovered sample outside original candidate support')
            out[begin:stop,47:67]=view.astype(np.float16);out[begin:stop,67:92]=prepared.map_features(keys).astype(np.float16)
            query_meta.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],query_id=query,start=begin,end=stop))
        if cursor!=n:raise ValueError('frozen Query rows not completely consumed')
        return out,query_meta,dict(fast_roots=1,rows_verified=n,legal_decoder_fallback=int(legal_fallback),dag_decoder_fallback=int(dag_fallback))
    except ValueError as error:
        # Supported scientific fallback: unchanged original target replay and
        # sampling, requiring every original supervision array to still match.
        arrays=[];query_meta=[];cursor=0
        for ex in branches(row):
            value,meta=sample(ex,Features(ex['graph'],ex['trajectory'],prepared))
            if value is None:continue
            size=len(value['y']);at=slice(cursor,cursor+size)
            for key in ('y','height','weight'):
                if not np.array_equal(value[key],old[key][at]):raise ValueError('frozen supervision mismatch in fallback '+key) from error
            if not np.array_equal(value['x'][:,:47],old['x'][at]):raise ValueError('frozen x47 mismatch in fallback') from error
            arrays.append(value['x']);query_meta.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],query_id=ex['query_id'],start=cursor,end=cursor+size));cursor+=size
        if cursor!=n:raise ValueError('fallback failed Root coverage') from error
        return np.concatenate(arrays),query_meta,dict(replayed_roots=1,rows_verified=n,fallback_reason=str(error))


def work(index):
    shard=SHARDS[index];path=DEST/('%06d.npz'%index);report=path.with_suffix('.json')
    if report.exists():
        value=json.loads(report.read_text());assert value['source']==SOURCE and sha(path)==value['sha256'];return value
    if sha(shard['path'])!=shard['sha256']:raise ValueError('frozen sampled cache checksum mismatch')
    with np.load(shard['path'],allow_pickle=False) as a:old={k:a[k] for k in a.files}
    output=np.empty((len(old['y']),92),np.float16);cursor=0;metas=[];counts={};reasons={}
    for root in shard['roots']:
        row=ROWS[root['root_id']];n=root['counts']['sampled_cells'];stop=cursor+n
        if row['scene_id']!=root['scene_id'] or row['queries']!=root['counts']['queries']:raise ValueError('frozen fitting Root identity mismatch')
        value,queries,ledger=augment_root(row,{k:v[cursor:stop] for k,v in old.items()});output[cursor:stop]=value
        for q in queries:q['start']+=cursor;q['end']+=cursor
        metas.extend(queries)
        for key,val in ledger.items():
            if key=='fallback_reason':reasons[val]=reasons.get(val,0)+1
            else:counts[key]=counts.get(key,0)+val
        cursor=stop
    assert cursor==len(old['y']);old['x']=output;tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**old);tmp.replace(path)
    value=dict(path=str(path),sha256=sha(path),source=SOURCE,rows=len(output),roots=[r['root_id'] for r in shard['roots']],queries=metas,counts=counts,fallback_reasons=reasons)
    dump(report,value);return value


def prepare(workers=16,limit=None):
    global ROWS,SHARDS,SOURCE,DEST
    final=ORIGINAL/'output/l2_two_stage_20261023/final_collection_20261024/manifest.json';rows=json.loads(final.read_text())['rows'];ROWS={r['root_id']:r for r in rows}
    path=ORIGINAL/'output/stage2_graphdelta_20261024/prepared/train/manifest.json';old=json.loads(path.read_text());SHARDS=old['shards']
    assert set(ROWS)=={r['root_id'] for shard in SHARDS for r in shard['roots']}
    actual_source=snapshot('validated exact-supervision augmentation')
    all_sources=identity();relevant={k:v for k,v in all_sources['files'].items() if Path(k).name in ('augment.py','features.py','data.py','common.py')}
    frozen=ORIGINAL/'output/stage2_graphdelta_20261024/frozen_sources/7803aa3bec784a8dec3e55306bf9e284263577f383f7aec39c1f38ff42c4b90f/manifest.json'
    baseline=json.loads(frozen.read_text())['files']
    for rel in ('eventwm_graph/stage2/features.py','eventwm_graph/stage2/data.py','eventwm_graph/stage2/api.py','eventwm_graph/stage2/common.py','eventwm_graph/partial/map.py'):
        actual=sha(Path(__file__).resolve().parents[2]/rel)
        assert actual==baseline[rel],('frozen augmentation dependency changed',rel)
        relevant[rel]=actual
    source_id=hashlib.sha256(json.dumps(relevant,sort_keys=True).encode()).hexdigest()
    SOURCE=dict(split='full',route='exact frozen sampled supervision plus validated causal feature augmentation',final_manifest_sha256=sha(final),frozen_sample_manifest_sha256=sha(path),sources=dict(files=relevant,sha256=source_id))
    DEST=OUT/'prepared'/('full_augmented_'+source_id[:12]);DEST.mkdir(parents=True,exist_ok=True);reports=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(work,range(min(limit or len(SHARDS),len(SHARDS)))):
            reports.append(result);print(json.dumps(dict(shards=len(reports),total=len(SHARDS),roots=sum(len(r['roots']) for r in reports),seconds=time.perf_counter()-start)),flush=True)
    if limit is None:
        value=dict(source=SOURCE,roots=len(rows),scenes=len({r['scene_id'] for r in rows}),shards=reports,sampled_cells=sum(r['rows'] for r in reports),seconds=time.perf_counter()-start,
            validation='every original sampled x47 recomputed exactly after fp16 encoding; y/height/weights preserved byteforbyte; strict DAG ancestry and clocks; all original shard SHA256 verified')
        assert value['sampled_cells']==old['counts']['sampled_cells'];dump(DEST/'manifest.json',value)
        (OUT/'augmentation_manifest_path.txt').write_text(str(DEST/'manifest.json')+'\n')
        # Only after external audit authorizes this execution route should the
        # orchestrator's full fitting manifest point to these exact shards.

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);p.add_argument('--limit',type=int);prepare(**vars(p.parse_args()))
