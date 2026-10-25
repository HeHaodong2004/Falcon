"""Owned hypothetical columns from already decoded candidate rows.

Sparse update dictionaries remain part of the API. This private path avoids
parsing those same dictionaries repeatedly to construct the normal full graph.
The independent legacy updater remains the correctness reference/fallback.
"""
from types import MappingProxyType
import numpy as np
from eventwm_graph.stage2_segment.native import Table, NativePartialGraph


def future_from_columns(base, keys, states, heights, now, pose, checkpoint):
    if base is None or not base.valid_topology:return None
    old=float(base.headers['timestamp']);now=float(now)
    if not np.isfinite(now) or now<old:raise ValueError('invalid hypothetical clock')
    if len(keys) and now==old:raise ValueError('starting-only path cannot reveal cells')
    if len(keys):
        if not np.all((states==1)|(states==2)) or not np.isfinite(heights).all():raise ValueError('invalid increment')
        biased=(keys.astype(np.int64)+np.int64(2147483648)).astype(np.uint64);codes=(biased[:,0]<<np.uint64(32))|biased[:,1]
        if np.any(codes[1:]<=codes[:-1]):raise ValueError('increment keys must be unique and ordered')
        pos=np.searchsorted(base.known_codes,codes);valid=pos<len(base.known_codes)
        if np.any(base.known_codes[pos[valid]]==codes[valid]):raise ValueError('known-cell update forbidden')
        t=base.table;names=list(t.sources);source='predicted_stage2_research'
        if source not in names:names.append(source)
        kk=np.concatenate([t.keys,keys.astype(np.int64)]);order=np.lexsort((kk[:,1],kk[:,0]))
        table=Table(kk[order],np.concatenate([t.states,states])[order],np.concatenate([t.heights,heights])[order],
            np.concatenate([t.first,np.full(len(keys),now)])[order],np.concatenate([t.last,np.full(len(keys),now)])[order],
            np.concatenate([t.source_ids,np.full(len(keys),names.index(source),np.int32)])[order],names)
    else:table=base.table
    headers=dict(base.headers)
    if now>old:
        headers.update(timestamp=now,ego_support=tuple(pose[:3]))
        visited=headers['visited']
        if not visited or np.linalg.norm((np.asarray(pose[:3])-np.asarray(visited[-1]['position']))[[0,2]])>=.55:
            headers['visited']=(*visited,MappingProxyType(dict(position=tuple(pose[:3]),timestamp=now)))
        if len(keys):headers['revision']+=1
    headers.update(hypothetical=True,prediction_checkpoint_sha256=checkpoint)
    return NativePartialGraph(headers,table)
