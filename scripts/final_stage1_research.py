"""One final fixed-recipe fit; execution plumbing only, no model selection."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'output/l2_two_stage_20261023'
DEST=BASE/'output/research_transition_20261024'
SPLIT='train_final_collection_20261024'
PYTHON='/research-user/miniforge3/envs/eventwm-falcon/bin/python'
ORIGINAL='cfe4c5ebecafbac92f4704a9b6efc6a01bc1ca786a45470516c5b540a01b8778'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def command(module,args,log,env=None,cwd=BASE):
    handle=(DEST/log).open('a')
    cmd=[PYTHON,'-m',module]+args
    print(json.dumps(dict(command=cmd,log=log)),flush=True)
    return subprocess.Popen(cmd,cwd=cwd,env=env,stdout=handle,stderr=subprocess.STDOUT)


def wait(proc):
    if proc.wait()!=0:raise RuntimeError('Execution failed; inspect named log. No data/model changes or retries authorized by this runner.')


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
             PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='1',EVENTWM_TWO_STAGE_BASE=str(BASE),
             PYTHONPATH=str(BASE)+':'+str(OUT/'deps'))
    while not (OUT/'final_collection_20261024/manifest.json').exists():time.sleep(5)
    snapshot=json.loads((OUT/'final_collection_20261024/manifest.json').read_text())
    assert snapshot['final_collection'] and snapshot['operator_after']['health']['collection_status']=='MANUALLY_STOPPED'
    if not (OUT/'prepared'/SPLIT/'manifest.json').exists():
        wait(command('eventwm_graph.l2_two_stage.extend_prepare',
            ['--snapshot','final_collection_20261024','--parent','train_campaign_20261023','--name',SPLIT,'--workers','4'],
            'stage1_extend_prepare.log',env))
    processes=[]
    for module,folder,workers in [('positive_history','positive_history','4'),('future_population_targets','population_future_targets','2')]:
        if not (OUT/folder/SPLIT/'manifest.json').exists():
            processes.append(command('eventwm_graph.l2_two_stage.'+module,['--cache',SPLIT,'--workers',workers],module+'.log',env))
    for proc in processes:wait(proc)
    if not (OUT/'last_event_metadata'/SPLIT/'manifest.json').exists():
        wait(command('eventwm_graph.l2_two_stage.last_event_metadata',['--cache',SPLIT,'--workers','4'],'last_event_metadata.log',env))
    # Preserve the established full-data optimizer/model implementation exactly.
    # The sole executable change redirects its old hardcoded OFFLINE target path.
    original=OUT/'frozen_sources'/ORIGINAL
    stage=DEST/'training_source'
    if not stage.exists():
        shutil.copytree(original,stage)
        p=stage/'eventwm_graph/l2_two_stage/fit_conditioned.py'
        text=p.read_text();old="attach_targets(records,'train_campaign_20261023')"
        assert text.count(old)==1
        p.chmod(0o644);p.write_text(text.replace(old,'attach_targets(records,sidecar)'));p.chmod(0o444)
        files={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*.py')}
        identity=dict(schema='final_stage1_original_recipe_source_1',parent=ORIGINAL,
            sole_change="fit_conditioned.attach_targets: old hardcoded cache -> explicit sidecar argument",
            files=files,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip())
        (stage/'FINAL_SOURCE_IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n')
    env['PYTHONPATH']=str(stage)+':'+str(OUT/'deps')
    run=OUT/'experiments/final_stage1_research_20261024'
    if (run/'complete.json').exists():
        cfg=json.loads((run/'config.json').read_text())
        assert cfg['data_manifest_sha256']==sha(OUT/'prepared'/SPLIT/'manifest.json')
        assert cfg['epochs']==8 and cfg['full_fit'] and cfg['sidecar']==SPLIT
        print('Verified identical final fit already complete; reused.',flush=True)
    elif run.exists():
        raise RuntimeError('Existing incomplete final run requires explicit diagnosis; never silently repeat a fit.')
    else:
        wait(command('eventwm_graph.l2_two_stage.fit_conditioned',[
            'final_stage1_research_20261024','--split',SPLIT,'--sidecar',SPLIT,'--epochs','8','--full-fit',
            '--seed','413','--batch-size','48','--learning-rate','.0002','--history-fraction','.5',
            '--association-guidance','--initial',str(OUT/'experiments/pilot_o_motion_guided_source_first/best.pt')],
            'final_stage1_fit.log',env,Path('/tmp')))
    (DEST/'STAGE1_FIT_COMPLETE.json').write_text(json.dumps(dict(checkpoint=str(run/'last.pt'),sha256=sha(run/'last.pt')))+'\n')


if __name__=='__main__':main()
