"""Prepare a finite fresh-run queue from explicit completed campaign recipes.

Only configurations are copied. Large immutable data/prepared caches and named
external controls stay referenced. This command never starts a worker, changes
a completed run, freezes data, or touches the production collector.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


CAMPAIGN=Path('/home/haodong/workspace/projects/eventwm-falcon/output/eventwm_forecast_performance_v2_20261015')
REFERENCE=CAMPAIGN.with_name('eventwm_forecast_full_20261014')
REFERENCE_SOURCE=CAMPAIGN.with_name('eventwm_forecast_full_checkout')
SOURCE=Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rewrite(value,mapping):
    if isinstance(value,dict):return {k:rewrite(v,mapping) for k,v in value.items()}
    if isinstance(value,list):return [rewrite(v,mapping) for v in value]
    if isinstance(value,str):
        for old,new in sorted(mapping.items(),key=lambda pair:len(pair[0]),reverse=True):
            if value==old or value.startswith(old+'/'):return new+value[len(old):]
        label,separator,path=value.partition('=')
        if separator and path.startswith('/'):
            return label+'='+rewrite(path,mapping)
    return value


def producer_dependencies(value,producers,own_run):
    if isinstance(value,dict):values=value.values()
    elif isinstance(value,list):values=value
    elif isinstance(value,str):
        label,separator,path=value.partition('=');path=path if separator and path.startswith('/') else value
        return {receipt for run,receipt in producers.items() if run!=own_run and
                (path==run or path.startswith(run+'/'))}
    else:return set()
    return set().union(*(producer_dependencies(v,producers,own_run) for v in values))


def prepare(spec_path,output):
    spec=read(spec_path);out=Path(output).resolve()
    if CAMPAIGN not in out.parents or out.exists():
        raise ValueError('Use a new directory below the versioned v2 campaign root')
    workflow=read(spec['workflow']);wanted=set(spec['include_runs'])
    original_owner=read(spec['workflow_launch'])
    if original_owner['module']!='eventforecastwm.workflow':raise ValueError('Workflow launcher expected')
    if not wanted or len(wanted)!=len(spec['include_runs']):raise ValueError('Explicit unique run names required')
    available=[*workflow['jobs'],*spec.get('reference_jobs',[])]
    jobs=[j for j in available if Path(read(j['config'])['run']).name in wanted]
    if {Path(read(j['config'])['run']).name for j in jobs}!=wanted:
        raise ValueError('Requested run is absent from the source workflow')
    if len(jobs)!=len(wanted):raise ValueError('Ambiguous duplicate run names')
    configurations={};mapping={};external={};producers={}
    for job in jobs:
        launch_path=Path(job['config']).resolve();launch=read(launch_path)
        old=Path(launch['run']).resolve()
        if not (old/'result.json').is_file():raise ValueError('Reproduction recipe must be a completed run: '+str(old))
        if CAMPAIGN not in old.parents and REFERENCE not in old.parents:
            raise ValueError('Only completed v2 or immutable v1 reference jobs may be reproduced here')
        mapping[str(old)]=str(out/old.name);configurations[str(launch_path)]=launch
        producers[str(out/old.name)]=str(out/old.name/'result.json')
        arguments=launch['args']
        for index,argument in enumerate(arguments[:-1]):
            if argument=='--config':
                path=Path(arguments[index+1]).resolve()
                if not any(root/'configs' in path.parents for root in (SOURCE,REFERENCE_SOURCE)) or path.suffix!='.json':
                    raise ValueError('Expected a committed JSON experiment configuration')
                configurations[str(path)]=read(path)
    names=[Path(path).name for path in configurations]
    if len(names)!=len(set(names)):raise ValueError('Configuration filenames collide')
    mapping.update({path:str(out/'configs'/Path(path).name) for path in configurations})
    aliases=spec.get('artifact_aliases',{})
    for alias,target in aliases.items():
        if digest(alias)!=digest(target):raise ValueError('Artifact alias is not byte-identical to its producer')
        destination=rewrite(target,mapping)
        if destination==target:raise ValueError('Artifact alias requires a selected fresh producer')
        mapping[alias]=destination
    for job in jobs:
        for dependency in job.get('requires',[]):
            if rewrite(dependency,mapping)==dependency:
                path=Path(dependency)
                if not path.is_file():raise FileNotFoundError('Missing frozen external dependency: '+dependency)
                external[str(path)]=digest(path)
    transformed={mapping[path]:rewrite(value,mapping) for path,value in configurations.items()}
    resolved_jobs=rewrite(jobs,mapping)
    for job in resolved_jobs:
        launch=transformed[job['config']];inputs=[launch];arguments=launch['args']
        for index,argument in enumerate(arguments[:-1]):
            if argument=='--config':inputs.append(transformed[arguments[index+1]])
        # Input aliases can introduce a fresh prerequisite absent from an old
        # queue that referenced an already completed release. Derive it before
        # launching so calibration cannot race its fresh physical producer.
        job['requires']=sorted(set(job.get('requires',[]))|producer_dependencies(inputs,producers,launch['run']))
    # Keep the established compact cache and shared adaptive guard. This queue
    # must have sole research ownership when launched, just like the original.
    owner=dict(workflow)
    owner.update(run=str(out/'workflow'),jobs=resolved_jobs,max_hours=16,jobs_per_gpu=1,
        gpu_launch_free_gib=8,training_priority_runs=[str(out/name) for name in sorted(wanted)
            if read(next(j['config'] for j in jobs if Path(read(j['config'])['run']).name==name))['module']
            in ('eventforecastwm.performance_v2.train','eventforecastwm.train_posterior','eventforecastwm.train_cached')])
    owner_path=out/'configs/workflow.json';launch_path=out/'configs/workflow_launch.json'
    launch=dict(original_owner,run=owner['run'],args=['--config',str(owner_path)])
    def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
    (out/'configs').mkdir(parents=True)
    for path,value in configurations.items():
        copied=rewrite(value,mapping)
        if 'module' in copied:copied['environment']=dict(original_owner.get('environment',{}),**copied.get('environment',{}))
        write(mapping[path],copied)
    write(owner_path,owner);write(launch_path,launch)
    commands={action:shlex.join(['bash',str(SOURCE/'scripts/run_eventwm_forecast.sh'),action,str(launch_path)])
              for action in ('launch','status','stop','resume')}
    receipt=dict(status='reproduction_prepared_not_launched',spec=spec,spec_sha256=digest(spec_path),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        source_sha256=digest(__file__),source_configuration_sha256={p:digest(p) for p in configurations},
        generated_configuration_sha256={str(p):digest(p) for p in sorted((out/'configs').glob('*.json'))},
        external_prerequisite_sha256=external,run_mapping={k:v for k,v in mapping.items() if k not in configurations and k not in aliases},
        artifact_aliases=aliases,runtime_environment=original_owner.get('environment',{}),
        commands=commands,production_actions=[],large_data_copied=False,
        ownership='Stop the current research owner and verify all children exited before launching this queue.',
        scope='Fresh fitting with unchanged scientific settings; external data/caches and explicitly named controls are reused. Not a bitwise cross-machine guarantee.')
    write(out/'preparation.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.spec,args.output),indent=2))


if __name__=='__main__':main()
