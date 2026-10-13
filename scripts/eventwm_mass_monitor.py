#!/usr/bin/env python3
"""Read-only extended coverage/resource view. Never writes collection files."""
import argparse
from pathlib import Path
import statistics
import subprocess
from collector.core import read
from eventwm_mass_collection import status,DEFAULT_OUTPUT

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);args=p.parse_args()
    status(args)
    out=args.output.resolve()
    if (out/'manifest.json').exists():
        m=read(out/'manifest.json');a=read(out/'aggregate.json')
        counts=[a['scene_counts'].get(s['name'],0) for s in m['scenes']]
        print('DATASET FAMILY COUNTS:',a['families'])
        print('ROOTS/SCENE MIN / MEDIAN / MAX:',min(counts,default=0),statistics.median(counts) if counts else 0,max(counts,default=0))
        print('ROOT ROLES:',a['roles']);print('ROBOT STATE:',a['robot_state'])
        print('QUERY VARIANTS:',a['variants']);print('BINARY COLLISION TARGET COUNTS:',a['binary_targets'])
        print('INVALID:',a['invalid']);print('UNAVAILABLE / FAILURE REASONS:',a['failed_reasons'])
        try:
            size=int(subprocess.check_output(['du','-sb',str(out)],text=True,timeout=20).split()[0])
            print('DATASET SIZE (all retained attempts/metadata):',round(size/1e9,3),'GB')
        except (OSError,subprocess.SubprocessError):print('DATASET SIZE: scan unavailable; completed-root byte total shown above')
