#!/usr/bin/env python3
"""Reproduce paired diagnostics from saved native rollouts; no policy rollout."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from eventwm_closed_loop import DEFAULT,read,save,summarize
from eventquerywm.navigation import NavigationConfig,RiskAwareSelector


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def analyze(args):
    summarize(args)
    comparison=read(args.output/'comparison.json')
    diagnostics={};latencies=[];planning=[];cold=[];rows=[]
    for tag in comparison['paired_episode_tags']:
        policies={}
        for policy in ('static','eventwm'):
            directory=args.output/'runs'/policy/tag
            m=read(directory/'metrics.json');obs=lines(directory/'observations.jsonl')
            decisions=lines(directory/'decisions.jsonl');actions=lines(directory/'actions.jsonl')
            assert all(a['timestamp']<b['timestamp'] for a,b in zip(obs,obs[1:]))
            assert len(actions)==m['navigation_steps'] and actions[-1]['episode_over']
            assert set(obs[0])=={'timestamp','robot_position','robot_yaw','previous_command',
                                'human_tracks','tracking_available','sensor_keys'}
            for frame in obs:
                for track in frame['human_tracks']:
                    assert track['first_seen_time']<=track['last_seen_time']<=frame['timestamp']
            families={};different=0;different_prefixes=0;previous=None;terminal_decision=None
            for decision in decisions:
                if not decision['candidates']:
                    continue
                arrived=[frame for frame in obs if frame['timestamp']<=decision['timestamp']+1e-8]
                assert abs(arrived[-1]['timestamp']-decision['timestamp'])<1e-8
                chosen=decision['selected'];candidate=decision['candidates'][chosen]
                families[candidate['family']]=families.get(candidate['family'],0)+1
                if policy=='eventwm':
                    # Instantaneous static counterfactual at the actual WM state,
                    # with the same previously executed family. This is not a
                    # separately rolled-out navigation outcome.
                    static=RiskAwareSelector(NavigationConfig(),False);static.last_family=previous
                    predicted=decision['safe_probabilities']
                    static_choice,_=static.select(decision['candidates'],[None]*len(predicted))
                    different+=int(static_choice!=chosen)
                    if static_choice!=chosen:
                        controls=candidate['query']['executable_controls']
                        other=decision['candidates'][static_choice]['query']['executable_controls']
                        limit=round(NavigationConfig().execution_s/candidate['query']['control_dt'])
                        a,b=np.asarray(controls[:limit]),np.asarray(other[:limit])
                        different_prefixes+=int(a.shape!=b.shape or not np.allclose(a,b,rtol=1e-5,atol=1e-5))
                    latencies.append(1000*decision['latency']['inference_s'])
                    planning.append(1000*decision['planning_s'])
                    if previous is None:
                        cold.append(1000*decision['latency']['inference_s'])
                    terminal_decision=dict(family=candidate['family'],safe_probability=predicted[chosen],
                        risk_range=[1-max(predicted),1-min(predicted)],timestamp=decision['timestamp'],
                        current_visible_tracks=sum(t['visible_now'] for t in arrived[-1]['human_tracks']),
                        current_valid_tracks=sum(t['valid'] for t in arrived[-1]['human_tracks']),
                        observation_ages_s=[t['observation_age'] for t in arrived[-1]['human_tracks']],
                        reason=decision['reason'])
                previous=candidate['family']
            policies[policy]=dict(visible_observation_frames=sum(any(t['visible_now'] for t in f['human_tracks']) for f in obs),
                valid_velocity_frames=sum(any(t['velocity_valid'] for t in f['human_tracks']) for f in obs),
                max_tracks=max(len(f['human_tracks']) for f in obs),selected_families=families,
                choices_different_from_static_at_same_state=different,
                executed_prefixes_different_from_static_at_same_state=different_prefixes,last_decision=terminal_decision)
            rows.append(dict(tag=tag,policy=policy,success=m['navigation_success'],spl=m['spl'],psc=m['psc'],
                human_collision=m['human_collision'],steps=m['navigation_steps'],navigation_time_s=m['navigation_time_s'],
                path_length_m=m['path_length_m'],initial_goal_distance_m=m['initial_goal_distance_m'],
                inference_ms=m['wm_inference_ms'],planning_ms=m['planning_ms'],**{
                    k:policies[policy][k] for k in ('visible_observation_frames','valid_velocity_frames','max_tracks',
                        'choices_different_from_static_at_same_state',
                        'executed_prefixes_different_from_static_at_same_state')}))
        diagnostics[tag]=policies
    def distribution(values):
        return dict(count=len(values),mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),
                    p95_ms=float(np.percentile(values,95)),max_ms=float(max(values))) if values else None
    result=dict(paired_episodes=len(diagnostics),episodes=diagnostics,
        wm_inference=distribution(latencies),wm_planning=distribution(planning),first_inference=distribution(cold),
        legal_input_checks_passed=True,
        caveats=['Partial-query execution does not supply full-query risk calibration labels.',
                 'Initial-distance/path-length ratio is uninformative for collision-truncated paths; use official SPL.',
                 'Static counterfactual choices use WM-visited states, not a counterfactual rollout.'])
    save(args.output/'analysis.json',result)
    with (args.output/'paired_episodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'}))
    return result


def plot(args):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(args.output/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tags=read(args.output/'comparison.json')['paired_episode_tags']
    fig,axes=plt.subplots(2,4,figsize=(12,6),squeeze=False)
    for ax,tag in zip(axes.ravel(),tags):
        for policy,color in (('static','tab:blue'),('eventwm','tab:orange')):
            actions=lines(args.output/'runs'/policy/tag/'actions.jsonl')
            ax.plot([r['timestamp'] for r in actions],[r['goal_distance'] for r in actions],color=color,label=policy)
            end=actions[-1];collision=end['official_metrics']['human_collision']
            ax.scatter([end['timestamp']],[end['goal_distance']],color=color,marker='x' if collision else 'o',s=40)
        ax.set_title(tag.split('_')[0]+' '+tag.split('_')[1],fontsize=10)
        ax.set_xlabel('Native simulation time (s)');ax.set_ylabel('Goal distance (m)');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Matched native rollouts; x = human collision, circle = goal success')
    fig.tight_layout();fig.savefig(args.output/'goal_progress.png',dpi=160);plt.close(fig)


def video(args,tag):
    """Offline evaluation visualization from actual poses and ray diagnostics."""
    import cv2
    import habitat_sim
    import imageio.v2 as imageio
    cv2.setNumThreads(1)
    directory=args.output/'runs/eventwm'/tag
    m=read(directory/'metrics.json');actions=lines(directory/'actions.jsonl')
    observations=read(directory/'perception_audit.json')
    decisions=lines(directory/'decisions.jsonl')
    from hashlib import sha256
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(directory/'resolved_config.yaml').habitat.simulator.agents.agent_0
    # Cache identity equals the private simulator's native scene and settings.
    row=read(directory/'provenance.json')['row']
    scene=str(Path(row['scene_path']))
    identity=sha256((scene+repr((cfg.radius,cfg.height,cfg.max_climb,cfg.max_slope))).encode()).hexdigest()[:20]
    path=args.output/'navmeshes'/(identity+'.navmesh')
    pf=habitat_sim.PathFinder();assert pf.load_nav_mesh(str(path)), str(path)
    lower,_=pf.get_bounds();resolution=.05
    mask=pf.get_topdown_view(resolution,m['start_position'][1])
    base=np.full((*mask.shape,3),255,np.uint8);base[mask]=[180,180,180]
    def point(position):
        return int((position[0]-lower[0])/resolution),int((position[2]-lower[2])/resolution)
    humans={round(x['timestamp'],8):x['humans'] for x in observations}
    dt=read(directory/'clock.json')['control_dt']
    trace=[point(m['start_position'])]
    destination=args.output/'videos'/f'{tag}_eventwm_diagnostic.mp4';destination.parent.mkdir(exist_ok=True)
    with imageio.get_writer(str(destination),fps=1/(2*dt),codec='libx264',ffmpeg_params=['-threads','1']) as out:
        for action in actions:
            trace.append(point(action['robot_position']))
            if action['step']%2:
                continue
            image=base.copy();cv2.polylines(image,[np.asarray(trace)],False,(20,100,255),2)
            cv2.circle(image,point(m['goal']),4,(40,190,40),-1)
            cv2.circle(image,trace[-1],3,(20,100,255),-1)
            for human in humans[round(action['timestamp'],8)]:
                torso=human['witness'].get('torso')
                if torso is not None:
                    cv2.circle(image,point(torso),3,(220,60,60) if human['visible'] else (220,160,80),-1)
            past=[d for d in decisions if d.get('selected') is not None and d['step']<action['step']]
            chosen=past[-1] if past else None
            label=f'{tag} step {action["step"]}; '+('initialization' if chosen is None else
                  chosen['candidates'][chosen['selected']]['family']+f' P(safe)={chosen["safe_probabilities"][chosen["selected"]]:.3f}')
            image=cv2.resize(image,(640,512),interpolation=cv2.INTER_NEAREST)
            banner=np.full((64,640,3),25,np.uint8)
            for y,text in ((20,label),(45,'Actual native poses; privileged evaluation overlay; no rendered RGB')):
                cv2.putText(banner,text,(5,y),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            frame=np.concatenate([banner,image]);out.append_data(frame)
            if action['step']==2*(len(actions)//4):
                imageio.imwrite(destination.with_suffix('.png'),frame)
    print(str(destination))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--video-tag')
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();analyze(args)
    if args.plot:
        plot(args)
    if args.video_tag:
        video(args,args.video_tag)
