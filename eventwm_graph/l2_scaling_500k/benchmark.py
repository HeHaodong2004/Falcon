"""Original complete workload, cohort, repetitions and deployed source bytes."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
from .common import BASE,OUT,PRIOR,sha,dump

def run(arm):
    import torch
    torch.set_num_threads(1)
    path=BASE/'docs/l2/intermittent_wm_20261022/reviews/reviewer_c_benchmark.py'
    spec=importlib.util.spec_from_file_location('frozen_complete_layer2_benchmark',path);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    dest=OUT/('benchmark_'+arm);dest.mkdir(exist_ok=False)
    (dest/'training').symlink_to((PRIOR if arm=='original' else OUT)/'training',target_is_directory=True)
    bench.OUT=dest
    checkpoint=(BASE/'docs/l2/intermittent_wm_20261022/artifacts/checkpoints/checkpoint.pt' if arm=='original' else OUT/'training/checkpoint.pt')
    assert checkpoint.exists()
    if arm=='original':assert sha(checkpoint)==sha(PRIOR/'training/checkpoint.pt')
    original_paths=bench.source_paths
    bench.source_paths=lambda:sorted(set(original_paths()+[Path(__file__)]))
    args=argparse.Namespace(mode='benchmark',runtime_frozen=True,checkpoint=str(checkpoint),tag='complete',limit=60,warm_repeats=5,cold_repeats=2,sensitivity_repeats=1)
    bench.run(args)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['original','new'],required=True);run(p.parse_args().arm)
