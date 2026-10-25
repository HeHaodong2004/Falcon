"""Stream exact derived-data accounting from immutable shard sidecars."""
from array import array
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from eventwm_graph.stage2_segment.common import OUT,dump,sha


def main():
    source=OUT/'prepared/full_mixed_ray';manifest=source/'manifest.json'
    while not manifest.exists():time.sleep(10)
    dest=OUT/'data_report';dest.mkdir(exist_ok=False);tick=time.perf_counter()
    counts=Counter();duration=Counter();motion=Counter();information=Counter();poses=Counter();root_mass=Counter()
    durations=array('d');candidate_sizes=array('d');targets=array('d');pose_sizes=array('d')
    roots=set();scenes=set();failures=[];shards=[];intermediate=0;aliases=0;known_windows=0;query_identities=0;ood=0;rss=0
    for path in sorted(source.glob('[0-9]*_[0-9]*.json')):
        report=json.loads(path.read_text());counts.update(report['counts']);failures.extend(report['failures'])
        rss=max(rss,report['peak_rss_kib']);local={}
        for row in report['segments']:
            rid=row['root_id'];scenes.add(row['scene_id']);entry=local.setdefault(rid,dict(windows=0,expected=row['segments_in_root'],queries=set()))
            entry['windows']+=1;entry['queries'].update(row['query_aliases'])
            seconds=row['duration'];new=row['new_free']+row['new_blocked'];candidates=row['candidate_cells']
            label=next((name for cutoff,name in ((.5000001,'<=0.5'),(1.0000001,'0.5-1'),(2.0000001,'1-2')) if seconds<=cutoff),'>2')
            duration[label]+=1;root_mass[label]+=1/row['segments_in_root']
            label='mixed' if row['movement']>.01 and row['rotation']>.01 else 'translation' if row['movement']>.01 else 'rotation' if row['rotation']>.01 else 'stationary'
            motion[label]+=1;information['0' if new==0 else '1-99' if new<100 else '>=100']+=1
            poses[row['poses']]+=1;durations.append(seconds);candidate_sizes.append(candidates);targets.append(new);pose_sizes.append(row['poses'])
            intermediate+=int(row['start_node'] is not None);aliases+=len(row['query_aliases']);known_windows+=int(row['known_terminal_differences']>0);ood+=row['out_of_domain_positive']
        for rid,entry in local.items():
            assert rid not in roots and entry['windows']==entry['expected'];roots.add(rid);query_identities+=len(entry['queries'])
        shards.append(dict(path=report['path'],sha256=report['sha256'],sidecar=str(path),sidecar_sha256=sha(path),counts=report['counts'],sampled_cells=sum(r['end']-r['start'] for r in report['segments'])))
    assert len(roots)==counts['roots'] and len(durations)==counts['candidate_segments'] and counts['roots']==105838
    def quantiles(values):return {str(q):float(np.quantile(values,q)) for q in (0,.05,.25,.5,.75,.95,.99,1)}
    total=sum(p.stat().st_size for p in source.iterdir() if p.is_file())
    identity=json.loads((source/'identity.json').read_text())
    dump(dest/'statistics.json',dict(valid_roots=len(roots),valid_queries=counts['queries'],unique_root_query_identities=query_identities,scenes=len(scenes),
        reconstructed_unique_future_states=counts['unique_frames'],reconstructed_root_and_future_states=counts['intermediate_states'],
        source_query_ancestry_frames=counts['query_ancestry_frames'],segments=len(durations),counts=dict(counts),
        segment_duration_counts=dict(duration),root_balanced_window_duration_mass={k:v/len(roots) for k,v in root_mass.items()},
        motion_counts=dict(motion),new_cell_counts=dict(information),saved_pose_counts=dict(poses),intermediate_start_segments=intermediate,
        query_window_alias_references=aliases,windows_with_known_terminal_state_or_height_differences=known_windows,
        out_of_domain_revelation_cells=ood,duration_quantiles_seconds=quantiles(np.asarray(durations)),
        candidate_cell_quantiles=quantiles(np.asarray(candidate_sizes)),new_cell_quantiles=quantiles(np.asarray(targets)),saved_pose_quantiles=quantiles(np.asarray(pose_sizes)),
        sampled_cell_rows=sum(s['sampled_cells'] for s in shards),shards=len(shards),npz_bytes=sum(Path(s['path']).stat().st_size for s in shards),
        directory_bytes_including_sidecars_and_manifest=total,max_worker_peak_rss_bytes=rss*1024,exclusions=failures,
        source_identity=identity,manifest=str(manifest),manifest_sha256=sha(manifest),statistics_seconds=time.perf_counter()-tick,
        independent_unit_note='Derived overlapping windows and shared prefixes are correlated; use original scenes and Roots, never segment count, as independence evidence'))
    dump(dest/'shard_manifest.json',dict(source_manifest=str(manifest),source_manifest_sha256=sha(manifest),shards=shards,root_ids=sorted(roots),scene_ids=sorted(scenes)))


if __name__=='__main__':main()
