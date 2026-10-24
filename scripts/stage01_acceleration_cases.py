#!/usr/bin/env python
"""Two bounded correctness cases; never included in throughput measurements."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eventwm_graph.config import file_digest, read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    from eventwm_graph.partial.acceleration import Backend, MODE, source_identity
    from eventwm_graph.partial.demo import scene_manifest
    from eventwm_graph.benchmark.map import load_certified_map
    cfg = read_json(ROOT/'configs/stage01_partial_graph_production.json')
    cfg['acceleration'] = dict(mode=MODE, persistent_scene=False)
    backend = Backend(cfg)
    cases = [dict(scene='mp3d_rPc6DW4iMge', seed=310201081, node=12, humans=1, steps=120),
             dict(scene='hm3d_GLAQ4DNUx5U', seed=310201062, node=34, humans=0, steps=2)]
    results = []
    for case in cases:
        with backend.configuration.scope():
            hmap = load_certified_map(scene_manifest(case['scene']))
        result = backend.collect_root(hmap, args.output/'dataset', case['seed'], cfg['stage0_gate'],
            24, case['steps'], node_index=case['node'], human_count=case['humans'], scenario='approach', pilot=True)
        results.append(dict(case=case, result=result))
    write_json(args.output/'cases.json', dict(schema='stage01_acceleration_bounded_cases_1',
        status='PASS', implementation=source_identity(), script_sha256=file_digest(__file__), cases=results,
        scope='Explicit accepted MP3D social and HM3D empty/timeout initializations; correctness only, no throughput claim',
        workers_configuration=28, training_started=False, production_started=False))
    print([(r['case'], r['result']['statistics']) for r in results])


if __name__ == '__main__':
    main()
