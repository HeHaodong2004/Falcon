"""Use the same verified past-only preprocessing sidecars during evaluation."""
import argparse
from pathlib import Path
import sys
from eventbeliefwm.data import sha256,read
from eventbeliefwm.experiment import save_json
from eventforecastwm import benchmark
from eventforecastwm.prepared_cache import prepare


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--prepared-cache',required=True)
    a,remaining=p.parse_known_args();manifest=remaining[remaining.index('--manifest')+1]
    output=Path(remaining[remaining.index('--output')+1]);output.parent.mkdir(parents=True,exist_ok=True)
    provenance=dict(directory=a.prepared_cache,manifest_sha256=sha256(manifest),
        sources={name:sha256(Path(__file__).parent/name) for name in ('benchmark_cached.py','prepared_cache.py')})
    receipt=output.parent/'preprocessing.json'
    if receipt.exists() and read(receipt)!=provenance:raise ValueError('Evaluation preprocessing identity changed')
    if not receipt.exists():save_json(receipt,provenance)
    original=benchmark.prepare_roots
    benchmark.prepare_roots=lambda model,roots,guard=None:prepare(model,roots,manifest,a.prepared_cache,original,guard)
    sys.argv=[sys.argv[0],*remaining];benchmark.main()


if __name__=='__main__':main()
