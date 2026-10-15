"""Physical supervision census; metadata eligibility and measured support differ."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.guard import Guard


def summarize(protocol_dir,manifest=None,guard=None):
    directory=Path(protocol_dir);protocol=read(directory/'protocol.json');blocks=read(directory/'blocks.json')
    episodes=[b['episode_id'] for b in blocks];trajectories=[b['trajectory_id'] for b in blocks if b['trajectory_id'] is not None]
    max_exposure={b['root_id']:max(min(r['observed_duration_s'],r['human_collision_time_s']
                      if r['human_collision_time_valid'] else float('inf')) for r in b['rows']) for b in blocks}
    result=dict(snapshot_sha256=protocol['snapshot_sha256'],eligible_roots=len(blocks),eligible_queries=sum(len(b['rows']) for b in blocks),
        unique_episodes=len(set(episodes)),nonnull_scene_trajectory_ids=len(trajectories),unique_nonnull_scene_trajectory_ids=len(set(trajectories)),
        scenes=len({b['scene'] for b in blocks}),fit_roots=len(protocol['subsets']['full']),development_roots=len(protocol['development']),
        scene_counts=dict(Counter(b['scene'].split(':')[0] for b in blocks)),
        legal_categories=dict(Counter(b['category'] for b in blocks)),
        metadata_precontact_horizon_upper_bounds={str(h):sum(x>=h for x in max_exposure.values()) for h in (.25,1,3,5)},
        count_definition='33 queries per root are outcomes, not independent physical trajectories',
        limitations='Metadata exposure is only an eligibility upper bound; identity/history and native interpolation masks require extraction')
    if manifest:
        path=Path(manifest);source=read(path);entries=source['roots'];strata=Counter();physical_agents=0;windows=0;history=[]
        incomplete_scene_ticks=0;physical_ticks=0
        coverage={str(h):Counter() for h in (.25,1,3,5)};splits={}
        for row in entries:
            if guard:guard.check()
            payload=path.parent/row['file']
            if sha256(payload)!=row['sha256']:raise ValueError('Frozen physical payload changed')
            value=torch.load(payload,map_location='cpu');ws=value['windows'];windows+=len(ws)
            for v in ws:
                exposure=v['targets']['physical_exposure'];all_valid=v['targets']['all_human_valid']
                incomplete_scene_ticks+=int((exposure & ~all_valid.all(-1)).sum());physical_ticks+=int(exposure.sum())
            w=ws[0];p=w['base'].humans;humans=w['metadata']['physical_humans_target_only'];physical_agents+=humans
            strata['zero_human_roots']+=humans==0;strata['never_observed_roots']+=humans>0 and len(p)==0
            strata['lost_at_root']+=bool(len(p) and (p[:,-1,4]>.5).sum()<len(p))
            strata['dense_ge3_humans']+=humans>=3;history.append(w['metadata']['history_span_s'])
            split=splits.setdefault(value['split'],Counter());split['roots']+=1;split['windows']+=len(ws)
            for h in (.25,1,3,5):
                l=int(h*4);c=coverage[str(h)];matched=[];physical=[]
                for v in ws:
                    valid=v['targets']['track_valid'][:l].all(0);matched.append(int(valid.sum()))
                    physical.append(bool(v['targets']['physical_exposure'][:l].all()))
                c['physical_roots_at_origin']+=physical[0];c['identity_roots_at_origin']+=matched[0]>0
                c['identity_agents_at_origin']+=matched[0];c['physical_windows']+=sum(physical)
                c['identity_windows']+=sum(x>0 for x in matched);c['identity_agent_windows']+=sum(matched)
                c['roots_with_any_supported_window']+=any(x>0 for x in matched)
                split[f'identity_roots_{h}s']+=matched[0]>0;split[f'physical_roots_{h}s']+=physical[0]
                split[f'identity_roots_any_window_{h}s']+=any(x>0 for x in matched)
        result['measured_cache']=dict(manifest=str(path),manifest_sha256=sha256(path),roots=len(entries),windows=windows,
            physical_agent_trajectories_target_only=physical_agents,scenes=len({r['scene'] for r in entries}),
            strata=dict(strata),coverage={k:dict(v) for k,v in coverage.items()},splits={k:dict(v) for k,v in splits.items()},
            root_history_span_s=dict(min=min(history),median=float(np.median(history)),max=max(history)),
            exposed_physical_ticks=physical_ticks,incomplete_scene_identity_ticks=incomplete_scene_ticks,
            full_pool_complete=len(entries)==len(blocks),
            windows_are_correlated=True,train_sampling='uniform roots, then uniform valid recorded origin; missing futures masked')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol-dir',required=True);p.add_argument('--manifest');p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    result=summarize(a.protocol_dir,a.manifest,Guard(out.parent));save_json(out,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
