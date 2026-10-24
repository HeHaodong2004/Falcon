#!/usr/bin/env python
"""Install a separately owned user service after both scientific gates pass."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eventwm_graph.config import read_json,write_json,file_digest
from eventwm_graph.partial.collect import verify_gate
from stage01_freeze import freeze


def deploy(config_path,service_name):
    cfg=read_json(config_path)
    verify_gate(cfg['stage0_gate']);verify_gate(cfg['stage1_gate'],'stage1')
    if cfg.get('automatic_training') is not False:raise ValueError('No training authorization')
    if not service_name.startswith('eventwm-partial-stage01-'):raise ValueError('Separate service namespace required')
    output=Path(cfg['output']).resolve();output.mkdir(parents=True,exist_ok=True)
    for key in ('stage0_gate','stage1_gate'):
        source=Path(cfg[key]).resolve();target=output/'acceptance'/(key+'.json');target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and file_digest(target)!=file_digest(source):raise ValueError('Existing deployed gate differs')
        if not target.exists():shutil.copyfile(source,target)
        cfg[key]=str(target)
    frozen=freeze(output/'source')
    cfg['output']=str(output);cfg['dataset']=str(Path(cfg['dataset']).resolve())
    configuration=output/'deployment.json';write_json(configuration,cfg)
    unit_dir=Path.home()/'.config/systemd/user';unit_dir.mkdir(parents=True,exist_ok=True)
    unit=unit_dir/(service_name+'.service')
    text=f'''[Unit]
Description=Observation-limited graph-query collection; no training
After=default.target
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory={ROOT}
Environment=OMP_NUM_THREADS=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=MAGNUM_LOG=quiet
Environment=HABITAT_SIM_LOG=quiet
ExecStart={sys.executable} -u {frozen}/scripts/stage01_campaign.py --config {configuration}
Restart=on-failure
RestartSec=15
KillMode=mixed
TimeoutStopSec=900
Nice=10
CPUQuota={cfg['workers']*100}%
MemoryHigh=64G
MemoryMax=96G
StandardOutput=append:{output}/service.log
StandardError=append:{output}/service.log

[Install]
WantedBy=default.target
'''
    if unit.exists() and unit.read_text()!=text:
        raise ValueError('Existing unit differs; use a distinct campaign service name')
    unit.write_text(text)
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','enable','--now',unit.name],check=True)
    result={'service':unit.name,'unit':str(unit),'configuration':str(configuration),
            'frozen_source':str(frozen),'automatic_training':False}
    write_json(output/'deployment_receipt.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--service-name',required=True)
    args=p.parse_args();print(deploy(args.config,args.service_name))
