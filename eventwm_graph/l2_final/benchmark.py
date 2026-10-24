"""Retain the complete frozen workload while timing the repaired implementation."""
import argparse
import importlib.util
import inspect
from .common import BASE,OUT,PREVIOUS,sha


def run(arm,tag=''):
    path=BASE/'docs/l2/intermittent_wm_20261022/reviews/reviewer_c_benchmark.py'
    spec=importlib.util.spec_from_file_location('frozen_benchmark',path);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    dest=OUT/('benchmark_'+arm+('_'+tag if tag else ''));dest.mkdir(exist_ok=False)
    training=(PREVIOUS if arm=='B' else OUT)/'training';(dest/'training').symlink_to(training,target_is_directory=True)
    bench.OUT=dest;original=bench.source_paths
    # The old unrelated zero-weight-domain fixture manually adds slot1. Give
    # that synthetic construction its required allocator metadata; measured
    # cohorts, one_call, repetitions, people and world budgets remain identical.
    source=inspect.getsource(bench.functional_checks)
    assert 'mix=b.fork(components=' in source
    source=source.replace('mix=b.fork(components=','mix=b.fork(next_slot=2,components=')
    exec(compile(source,str(__import__('pathlib').Path(__file__)), 'exec'),bench.__dict__)
    bench.source_paths=lambda:sorted(set(original()+list((BASE/'eventwm_graph/l2_intermittent').glob('*.py'))+[__import__('pathlib').Path(__file__)]))
    # The frozen benchmark requires a checkpoint path inside this worktree.
    local=dest/'checkpoint.pt';__import__('shutil').copyfile(training/'checkpoint.pt',local)
    assert sha(local)==sha(training/'checkpoint.pt')
    args=argparse.Namespace(mode='benchmark',runtime_frozen=True,checkpoint=str(local),tag='complete',limit=60,
                            warm_repeats=5,cold_repeats=2,sensitivity_repeats=1)
    bench.run(args)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['B','C'],required=True);parser.add_argument('--tag',default='');a=parser.parse_args();run(a.arm,a.tag)
