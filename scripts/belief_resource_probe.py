#!/usr/bin/env python3
"""Read-only host/collector probe. Never signals a process or changes CUDA."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

PROJECT = Path('/home/haodong/workspace/projects/eventwm-falcon')


def sample(cuda=False):
    result = {'timestamp': time.time(), 'pid': os.getpid(),
              'disk_free_bytes': shutil.disk_usage(PROJECT).free,
              'production_actions': []}
    database = Path('/tmp/eventwm_v4_300k/pipeline.sqlite3')
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        meta = {key: json.loads(value) for key, value in db.execute(
            "SELECT key,value FROM meta WHERE key IN ('manifest','progress')")}
    manifest = meta['manifest']
    runner = manifest.get('runner', {})
    result['collector'] = {'runner': runner, 'state': manifest.get('state'),
                           'progress': meta['progress'], 'source_commit': manifest.get('frozen_commit')}
    result['collector']['pid_exists'] = Path('/proc/' + str(runner.get('pid'))).exists()
    result['nvidia_kernel'] = Path('/proc/driver/nvidia/version').read_text().splitlines()[0]
    processes = []
    owners = []
    inaccessible = 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc/'cmdline').read_bytes().split(b'\0')
            argv = [part.decode(errors='replace') for part in argv if part]
            if argv and 'python' in Path(argv[0]).name:
                if any('eventwm' in arg or 'belief' in arg for arg in argv):
                    # Only operational command lines; never environment or shell wrappers.
                    processes.append({'pid': int(proc.name), 'argv': argv[:18]})
            devices = set()
            try:
                for fd in (proc/'fd').iterdir():
                    try:
                        target = os.readlink(fd)
                        if target.startswith('/dev/nvidia'):
                            devices.add(target)
                    except OSError:
                        pass
            except PermissionError:
                inaccessible += 1
            if devices:
                owners.append({'pid': int(proc.name), 'executable': argv[0].split(' --')[0] if argv else None,
                               'devices': sorted(devices)})
        except (OSError, ProcessLookupError):
            continue
    result['research_and_collector_processes'] = processes
    result['nvidia_device_owners'] = owners
    result['inaccessible_process_fd_directories'] = inaccessible
    if cuda:
        import torch
        result['torch_version'] = torch.__version__
        result['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        result['cuda_available'] = torch.cuda.is_available()
        result['gpus'] = []
        for device in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(device):
                    x = torch.ones(8, device=f'cuda:{device}')
                    value = float(x.sum())
                    torch.cuda.synchronize()
                    result['gpus'].append({'index': device, 'name': torch.cuda.get_device_name(),
                        'free_total_bytes': torch.cuda.mem_get_info(), 'tensor_sum': value})
            except Exception as error:
                result['gpus'].append({'index': device, 'error': repr(error)})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = sample(args.cuda)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    summary = {key: value for key, value in result.items()
               if key not in ('research_and_collector_processes', 'nvidia_device_owners')}
    summary['process_count'] = len(result['research_and_collector_processes'])
    summary['gpu_device_owner_pids'] = [p['pid'] for p in result['nvidia_device_owners']]
    print(json.dumps(summary, indent=2))
