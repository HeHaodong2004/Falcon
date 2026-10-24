"""Finish the fixed repair campaign in two bounded, one-thread job chains.

Run from the research worktree. No collection command or mutable source reader
is invoked. Start `fit-evaluate` and `report-profile-fresh` at most once each.
Waiting jobs use no numerical worker. The second chain waits for feature fitting
and observation extraction, so two heavy research CPUs is the maximum.
Artifacts are published exclusively by the underlying modules; failures stop
the chain and are preserved for diagnosis.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
OLD=Path('output/l2_partial_graph_wm_explore_20261021')
NEW=Path('output/l2_partial_graph_wm_repair_20261021')
SNAPSHOT=OLD/'snapshot'
DYNAMICS=OLD/'learned_posdiff/checkpoint.pt'
POPULATION=OLD/'train.npy.population.json'
EVENT=NEW/'native_events_v4.json'
COHORT=NEW/'development_cohort.json'


def context(label):
    source=Path('/research-user/workspace/projects/eventwm-falcon-acceleration-v2/output/stage01_acceleration_v2_20261021/collection_user_authorized_5s_20261021_1/status.json')
    status=json.loads(source.read_text())
    value={'time_unix':time.time(),'read_only_collector_status':str(source),
           'collector':{k:status.get(k) for k in ['status','status_reason','outcomes','roots','active_worker_count',
                         'collector_restart_count','supervisor_pid','validated_outcomes_per_second_recent']},
           'observation_checkpoint_sha256':hashlib.sha256((NEW/'observation_fit.json').read_bytes()).hexdigest()}
    path=NEW/(label+'.json')
    with path.open('x') as stream:json.dump(value,stream,indent=2)


def wait_for(paths):
    start=time.monotonic()
    while not all(Path(path).is_file() for path in paths):
        if time.monotonic()-start>7200:
            raise TimeoutError('Required completed research artifacts did not arrive: '+str(paths))
        time.sleep(2)


def run(module,args,label):
    command=[sys.executable,'-m','eventwm_graph.l2_partial.'+module,*map(str,args)]
    log=NEW/(label+'.log')
    if log.exists():raise FileExistsError(log)
    print(json.dumps({'start':label,'command':command,'time_unix':time.time()}),flush=True)
    with log.open('x') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    print(json.dumps({'finished':label,'exit_code':result.returncode,'time_unix':time.time()}),flush=True)
    if result.returncode:raise subprocess.CalledProcessError(result.returncode,command)


def fit_evaluate():
    wait_for([NEW/'events_train_v4.npy.manifest.json',NEW/'events_dev_v4.npy.manifest.json'])
    run('repair_events',['fit',NEW/'events_train_v4.npy',NEW/'events_dev_v4.npy',EVENT,
                        '--epochs','10','--batch-roots','4','--initial-checkpoint',OLD/'native_events_final.json'],
                        'native_events_v4_fit')
    run('repair_evaluation',['run',SNAPSHOT,DYNAMICS,POPULATION,EVENT,COHORT,NEW/'population_evaluation.json',
                            '--variants',NEW/'evaluation_variants.json','--verify-invariance-roots','3'],
                            'population_evaluation')


def report_profile_fresh():
    wait_for([EVENT,NEW/'observations_dev_v4.npy.manifest.json'])
    run('repair_risk',['report',NEW/'events_dev_v4.npy',NEW/'event_checkpoints.json',NEW/'risk_development.json'],
                     'risk_development')
    run('repair_observation_evaluation',['report',NEW/'observations_dev_v4.npy',NEW/'events_dev_v4.npy',
                                        NEW/'event_checkpoints.json',NEW/'observation_calibration.json'],
                                        'observation_calibration')
    context('latency_final_context_before')
    run('repair_profile',[SNAPSHOT,NEW/'latency_final_active.json','--checkpoint',DYNAMICS,
                         '--population',POPULATION,'--event-checkpoint',EVENT,
                         '--model-kwargs',NEW/'model_kwargs.json','--cohort',COHORT,'--limit','60',
                         '--repeats','10','--cold-repeats','1','--rl-roots','3','--rl-repeats','100'],
                         'latency_final_active')
    context('latency_final_context_after')
    run('repair_probability_diagnostics',[NEW/'probability_diagnostics_v5_final.json','--snapshot',SNAPSHOT,
                         '--checkpoint',DYNAMICS,'--population',POPULATION,
                         '--observation-checkpoint',NEW/'observation_fit.json','--event-checkpoint',EVENT,
                         '--cohort',NEW/'smoke_cohort10.json','--limit','10'],
                         'probability_diagnostics_v5_final')
    # This is the one-time boundary before opening any new evaluation payload.
    # Checkpoint selection has ended; the source/checkpoint guard rejects edits.
    run('repair_fresh',['freeze',NEW/'fresh_config.json',NEW/'fresh_design.json'],'fresh_freeze')
    run('repair_fresh',['snapshot',NEW/'fresh_design.json',NEW/'fresh_snapshot'],'fresh_snapshot')
    run('repair_fresh',['evaluate',NEW/'fresh_design.json',NEW/'fresh_snapshot',NEW/'fresh_evaluation'],
                     'fresh_evaluation')


def scene_evaluate():
    event=NEW/'native_events_v5_scene.json'
    wait_for([event])
    run('repair_evaluation',['run',SNAPSHOT,DYNAMICS,POPULATION,event,COHORT,NEW/'population_evaluation.json',
                            '--variants',NEW/'evaluation_variants.json','--verify-invariance-roots','3'],
                            'population_evaluation_v5')


def scene_reports_profile_fresh():
    event=NEW/'native_events_v5_scene.json'
    wait_for([event,NEW/'observations_dev_v4.npy.manifest.json'])
    run('repair_risk',['report',NEW/'events_dev_v4.npy',NEW/'event_checkpoints_v5.json',NEW/'risk_scene_calibrated.json'],
                     'risk_scene_calibrated')
    run('repair_observation_evaluation',['report',NEW/'observations_dev_v4.npy',NEW/'events_dev_v4.npy',
                                        NEW/'event_checkpoints_v5.json',NEW/'observation_calibration_scene.json'],
                                        'observation_calibration_scene')
    context('latency_v5_context_before')
    run('repair_profile',[SNAPSHOT,NEW/'latency_v5_active.json','--checkpoint',DYNAMICS,
                         '--population',POPULATION,'--event-checkpoint',event,
                         '--model-kwargs',NEW/'model_kwargs.json','--cohort',COHORT,'--limit','60',
                         '--repeats','10','--cold-repeats','1','--rl-roots','3','--rl-repeats','100'],
                         'latency_v5_active')
    context('latency_v5_context_after')
    run('repair_probability_diagnostics',[NEW/'probability_diagnostics_v5_final.json','--snapshot',SNAPSHOT,
                         '--checkpoint',DYNAMICS,'--population',POPULATION,
                         '--observation-checkpoint',NEW/'observation_fit.json','--event-checkpoint',event,
                         '--cohort',NEW/'smoke_cohort10.json','--limit','10'],
                         'probability_diagnostics_v5_final')
    run('repair_fresh',['freeze',NEW/'fresh_config_v5.json',NEW/'fresh_design.json'],'fresh_freeze')
    run('repair_fresh',['snapshot',NEW/'fresh_design.json',NEW/'fresh_snapshot'],'fresh_snapshot')
    run('repair_fresh',['evaluate',NEW/'fresh_design.json',NEW/'fresh_snapshot',NEW/'fresh_evaluation'],
                     'fresh_evaluation')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['fit-evaluate','report-profile-fresh','scene-evaluate','scene-report-profile-fresh'])
    args=parser.parse_args()
    os.chdir(REPO)
    for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[variable]='1'
    os.setpriority(os.PRIO_PROCESS,0,19)
    subprocess.run(['ionice','-c','3','-p',str(os.getpid())],check=True)
    {'fit-evaluate':fit_evaluate,'report-profile-fresh':report_profile_fresh,
     'scene-evaluate':scene_evaluate,'scene-report-profile-fresh':scene_reports_profile_fresh}[args.stage]()
