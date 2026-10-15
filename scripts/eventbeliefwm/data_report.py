"""Summarize compact root supervision without claiming independent query futures."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from .data import read, sha256


def report(cache):
    cache=Path(cache)
    manifest=read(cache/'manifest.json')
    summaries={}
    for split in sorted({r['split'] for r in manifest['roots']}):
        rows=[r for r in manifest['roots'] if r['split']==split]
        association=Counter();categories=Counter();human_counts=Counter();coverage=defaultdict(Counter)
        exogeneity_count=0;exogeneity_max=0.;events=observations=0
        errors=defaultdict(list);full_rows=[]
        for row in rows:
            path=cache/row['file']
            if sha256(path)!=row['sha256']:
                raise ValueError('Compact supervision changed: '+str(path))
            sample=torch.load(path,map_location='cpu')
            md=sample['metadata'];t=sample['targets']
            for name,value in t.items():
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError('Nonfinite target: '+name+' / '+row['root_id'])
            if (t['collision_interval'] & ~t['collision_valid']).any() or (t['collision_interval'].sum(-1)>1).any():
                raise ValueError('Invalid absorbing first-contact supervision')
            if not (sample['history_timestamps'][1:]>sample['history_timestamps'][:-1]).all():
                raise ValueError('Legal history clocks changed')
            association.update(md['association']);categories[row['strata']['category']]+=1
            human_counts[str(md['physical_humans_target_only'])]+=1
            exogeneity_count+=md['exogeneity_comparisons']
            exogeneity_max=max(exogeneity_max,md['exogeneity_max_difference_m'])
            events+=int(t['collision_interval'].sum());observations+=int(t['observed_tracks'].sum())
            if sample['base'].humans.shape[0]!=len(md['track_ids']):
                raise ValueError('Legal track slots changed')
            for horizon in (1.,3.,5.):
                at=round(horizon/manifest['dt'])-1
                if at>=len(t['times']):continue
                c=coverage[str(horizon)]
                c['physical_roots']+=int(t['physical_exposure'][at])
                c['identity_associated_roots']+=int(t['track_valid'][at].any())
                c['identity_associated_tracks']+=int(t['track_valid'][at].sum())
                c['all_physical_humans']+=int(t['all_human_valid'][at].sum())
                c['query_exposures']+=int(t['query_exposure'][:,at].sum())
                p=t['all_human_positions'][at]
                valid=t['all_human_valid'][at] & (p.abs()<8).all(-1)
                cells=torch.floor((p[valid]+8)/.5).long()
                c['foreground_cells']+=len(set(map(tuple,cells.tolist())))
                c['never_seen_physical_roots']+=int(sample['base'].humans.shape[0]==0 and bool(t['all_human_valid'][at].any()))
            for key in ('nominal_execution_error_m','query16_interpolation_error_m'):
                errors[key].append(md[key])
            full_rows.extend(md['rows'])
        error_summary={}
        for key,values in errors.items():
            count=sum(v['count'] for v in values)
            error_summary[key]=dict(samples=count,
                weighted_mean=sum(v['count']*v['mean'] for v in values if v['count'])/count if count else None,
                max=max((v['max'] for v in values if v['count']),default=None),
                root_p95_median=float(np.median([v['p95'] for v in values if v['count']])) if count else None)
        groups=defaultdict(list)
        for row in full_rows:groups[row['group_id']].append(row['sr_label'])
        summaries[split]=dict(roots=len(rows),queries=sum(r['queries'] for r in rows),
            scenes=len({r['strata']['scene'] for r in rows}),categories=dict(categories),
            physical_human_counts_target_only=dict(human_counts),
            identity_association=dict(association),horizon_coverage=dict(coverage),
            exogeneity_position_comparisons=exogeneity_count,exogeneity_max_difference_m=exogeneity_max,
            collision_events_within_cached_horizon=events,positive_tracked_detection_targets=observations,
            safe_collision_mixed_path_groups=sum(len(set(v))>1 for v in groups.values()),
            native_collision_queries=sum(not row['sr_label'] for row in full_rows),**error_summary)
    return dict(snapshot_sha256=manifest['snapshot_sha256'],extractor_sha256=manifest['extractor_sha256'],
        cache_manifest_sha256=sha256(cache/'manifest.json'),splits=summaries,
        compact_bytes=sum((cache/r['file']).stat().st_size for r in manifest['roots']),
        limitations=['Canonical .25s positions linearly interpolate native precontact states only.',
            'Exogeneity checks compare all available candidates at canonical times, not every native physics sample.',
            'Long horizons are a survivor-selected subset; unsupported and postterminal intervals remain masked.',
            'No static map or optical occlusion raster is present; geometry/support are explicitly invalid.',
            'Unknown human truth supervises occupancy only; no privileged physical count enters model inputs.',
            'Observation correction labels cover identities present in initial legal history; birth/reassociation remains separate.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=report(args.cache)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
