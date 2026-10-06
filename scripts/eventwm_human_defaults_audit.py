#!/usr/bin/env python3
"""Read-only comparison to pinned upstream FALCON human defaults."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
from collector.core import read,write

UPSTREAM='d15ef512c4b1451f9daeaf6d4d7ecc702ed57430'
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
FILES=['falcon/additional_action.py','falcon/rearrange_sim_v2.py',
       'habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d.yaml',
       'habitat-lab/habitat/config/default_structured_configs.py',
       'habitat-lab/habitat/tasks/rearrange/social_nav_v2/socialnav_task.py']


def action_differences(defaults,resolved):
    """Compare every field, including missing actions or added overrides."""
    differences={}
    for key,expected in defaults.items():
        actual=resolved.get(key,{})
        diff={field:[expected.get(field),actual.get(field)]
              for field in sorted(set(expected)|set(actual))
              if field not in expected or field not in actual or expected[field]!=actual[field]}
        if diff:differences[key]=diff
    return differences


def audit(dataset):
    import habitat_sim
    import falcon
    from habitat_baselines.config.default import get_config
    from habitat.core.registry import registry
    from omegaconf import OmegaConf
    import inspect
    cfg=get_config('social_nav_v2/falcon_hm3d.yaml')
    defaults=OmegaConf.to_container(cfg.habitat.task.actions,resolve=True)
    keys=[k for k in defaults if 'oracle_nav_randcoord_action_obstacle' in k]
    defaults={k:defaults[k] for k in keys}
    if len(defaults)!=6:raise ValueError('Expected six benchmark human action slots')
    source={}
    for name in FILES:
        actual=(REPO/name).read_bytes();upstream=subprocess.check_output(['git','show',UPSTREAM+':'+name],cwd=REPO)
        source[name]=dict(sha256=hashlib.sha256(actual).hexdigest(),matches_upstream=actual==upstream)
    manifest=read(dataset/'manifest.json');differences={};native=[];checked=0
    for root in manifest['roots']:
        if root['state']!='completed':continue
        path=dataset/root['path'];resolved=OmegaConf.to_container(OmegaConf.load(path/'audits/workers/root/resolved_config.yaml').habitat.task.actions,resolve=True)
        diff=action_differences(defaults,resolved)
        if diff:differences[root['root_id']]=diff
        checked+=1
        if root['human_count']:
            native.append(dict(root_id=root['root_id'],human_count=root['human_count'],
                **read(path/'audits/audits.json')['human_reactivity']))
    action=registry.get_task_action('OracleNavRandCoordAction_Obstacle')
    registered_source=Path(inspect.getsourcefile(action)).resolve()
    native_pass=(all(x['matches_upstream'] for x in source.values()) and checked>0 and not differences
                 and registered_source==(REPO/'falcon/additional_action.py').resolve())
    return dict(upstream_commit=UPSTREAM,benchmark_config='social_nav_v2/falcon_hm3d.yaml',
        registered_action_source=str(registered_source),source_comparison=source,
        default_human_actions=defaults,checked_completed_roots=checked,human_action_differences=differences,
        native_reactivity_audits=native,
        first_human_trajectory_comparison=dict(human_present_roots=len(native),
            max_deviation_m=max((r['details']['max_deviation_m'] for r in native),default=None),
            diagnostic_pass_count=sum(r['status']=='PASS' for r in native),
            interpretation='No observed robot reactivity is expected, not a benchmark-fidelity failure'),
        benchmark_default_human_config_identified=native_pass,
        robot_reactive_human_behavior_expected=False,
        custom_human_controller_or_action_config_modifications=not all(x['matches_upstream'] for x in source.values()) or bool(differences),
        source_findings=dict(avoidance_agent_range='range(1, num_articulated_agents); robot is agent 0',
            avoidance_condition='other human, geodesic distance <2m and forward dot >0.5; requires previous positions',
            optional_robot_distance_switch=[v['human_stop_and_walk_to_robot_distance_threshold'] for v in defaults.values()],
            optional_switch_limitation='disabled by default; enabling requires exactly two allocated agents and changes speed activation',
            limitation='controlled episode fixtures remain; equality of controllers does not establish benchmark episode-distribution equivalence'),
        benchmark_native_dynamics_pass=native_pass,
        semantics='Human-human avoidance; robot agent 0 excluded; ego-independent with respect to robot, not other humans',
        status='PASS' if native_pass else 'FAIL')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=audit(args.dataset);write(args.output,r)
    print({k:r[k] for k in ('checked_completed_roots','benchmark_default_human_config_identified','robot_reactive_human_behavior_expected','custom_human_controller_or_action_config_modifications','status')})
    if not r['benchmark_native_dynamics_pass']:raise SystemExit(1)

if __name__=='__main__':main()
