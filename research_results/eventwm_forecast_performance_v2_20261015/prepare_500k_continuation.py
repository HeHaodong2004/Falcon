"""Configuration-only continuation; requires an audited >=495k protocol.

Run with PYTHONPATH=scripts. No worker is launched and no production file is
written. Existing data/scaling/geometry implementations perform extraction.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from eventforecastwm.performance_v2.prepare_reproduction import prepare, rewrite
from eventforecastwm.workflow import active

SOURCE = Path(__file__).resolve().parents[2]
CAMPAIGN = SOURCE.parent / 'eventwm_forecast_performance_v2_20261015'
REFERENCE = SOURCE.parent / 'eventwm_forecast_full_20261014'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def presentations(steps, roots, batch=64):
    epochs, remainder = divmod(steps, math.ceil(roots / batch))
    return epochs * roots + min(remainder * batch, roots)


def exposure_steps(roots):
    target = math.ceil(1562598 * roots / 8965)
    low, high = 0, math.ceil(target / roots) * math.ceil(roots / 64)
    while low < high:
        middle = (low + high) // 2
        if presentations(middle, roots) < target:
            low = middle + 1
        else:
            high = middle
    return low


def validate(protocol_path):
    protocol = read(protocol_path)
    snapshot = Path(protocol['snapshot'])
    audit = read(snapshot.parent / 'audit.json')
    if audit['retained_queries'] < 495000:
        raise ValueError('The audited 495000-query gate is closed')
    if digest(snapshot) != protocol['snapshot_sha256']:
        raise ValueError('Snapshot changed')
    if read(snapshot.parent / 'status.json')['stage'] != 'frozen':
        raise ValueError('Snapshot is not sealed')
    parent_path = REFERENCE / 'protocol/protocol.json'
    parent = read(parent_path)
    if protocol['parent_protocol_sha256'] != digest(parent_path):
        raise ValueError('Wrong 350k parent protocol')
    if protocol['development'] != parent['development']:
        raise ValueError('Development membership changed')
    if protocol['subsets']['full350matched'] != parent['subsets']['full']:
        raise ValueError('Original fitting membership changed')
    fitting = protocol['subsets']['full']
    held = read(CAMPAIGN / 'confirmation_protocol/membership.json')['banks']
    if len(set(fitting)) != len(fitting) or set(fitting) & set(protocol['development']):
        raise ValueError('Repeated fitting root or development leakage')
    if set(fitting) & {rid for bank in held.values() for rid in bank}:
        raise ValueError('Confirmation entered fitting')
    if len(fitting) <= 8965:
        raise ValueError('No additional independent fitting roots')
    return protocol


def generate(args):
    protocol_path = args.protocol.resolve()
    protocol = validate(protocol_path)
    owner_path = Path((CAMPAIGN / 'CURRENT_OWNER').read_text().strip())
    owner = read(owner_path)
    if active(owner['run']):
        raise RuntimeError('Stop the current owner and verify all children exited first')
    owner_config = read(owner['args'][1])
    if any(active(read(job['config'])['run']) for job in owner_config['jobs']):
        raise RuntimeError('A previous research worker is still active')
    cache = args.cache.resolve()
    geometry = args.geometry.resolve()
    if CAMPAIGN not in cache.parents or CAMPAIGN not in geometry.parents:
        raise ValueError('Use separate versioned campaign cache paths')
    if geometry.exists() and not (geometry / 'manifest.json').exists():
        raise FileExistsError('Use new geometry or a completed verified geometry cache')
    if (cache / 'manifest.json').exists():
        cached = read(cache / 'manifest.json')
        if cached['protocol_sha256'] != digest(protocol_path):
            raise ValueError('Completed expanded cache has another protocol')
        if {r['root_id'] for r in cached['roots']} != set(protocol['development'] + protocol['subsets']['full']):
            raise ValueError('Completed expanded cache membership mismatch')
    if geometry.exists():
        if read(geometry / 'manifest.json')['source_manifest_sha256'] != digest(cache / 'manifest.json'):
            raise ValueError('Completed expanded geometry has another physical cache')
    reuse = read(cache / 'incremental_reuse.json')
    if reuse['parent_manifest_sha256'] != digest(REFERENCE / 'cache_full/manifest.json'):
        raise ValueError('Incremental compact-cache inheritance changed')
    recipe = 'reproduction_primary.json' if args.family == 'primary' else 'reproduction_selected_family.json'
    prepared = prepare(SOURCE / 'configs/eventwm_performance_v2' / recipe, args.output)
    output = args.output.resolve()
    replacements = {
        str(REFERENCE / 'protocol/protocol.json'): str(protocol_path),
        str(REFERENCE / 'cache_full/manifest.json'): str(cache / 'manifest.json'),
        str(REFERENCE / 'geometry_full/manifest.json'): str(geometry / 'manifest.json'),
        str(REFERENCE / 'prepared_075'): str(output / 'prepared_075'),
        str(REFERENCE / 'prior_cache_full'): str(output / 'prior_cache'),
        str(REFERENCE / 'anonymous_prior_cache'): str(output / 'prior_cache'),
    }
    steps = 24576 if args.budget == 'updates' else exposure_steps(len(protocol['subsets']['full']))
    for path in (output / 'configs').glob('*.json'):
        value = rewrite(read(path), replacements)
        if value.get('objective') == 'distribution' and value.get('steps') == 24576:
            value['steps'] = steps
        path.write_text(json.dumps(value, indent=2) + '\n')
    needs_geometry = any(str(geometry / 'manifest.json') in p.read_text()
                         for p in (output / 'configs').glob('*.json'))
    if needs_geometry and not geometry.exists():
        geometry.mkdir(parents=True)
        inherited = read(REFERENCE / 'geometry_full/manifest.json')['roots']
        allowed = set(protocol['development']) | set(protocol['subsets']['full'])
        for row in inherited:
            if row['root_id'] not in allowed:
                continue
            source = REFERENCE / 'geometry_full' / row['file']
            if digest(source) != row['sha256']:
                raise ValueError('Inherited geometry bytes changed')
            os.link(source, geometry / row['file'])
    environment = read(output / 'configs/workflow_launch.json')['environment']
    jobs = []
    for name, directory, module, arguments in [
        ('data', cache, 'eventforecastwm.data', ['build', '--protocol-dir', str(protocol_path.parent), '--output', str(cache), '--workers', '1']),
        ('geometry', geometry, 'eventforecastwm.geometry', ['--manifest', str(cache / 'manifest.json'), '--output', str(geometry)]),
    ]:
        if name == 'geometry' and not needs_geometry:
            continue
        # A completed immutable manifest is the data-stage receipt. The link
        # stays dangling until the established builder writes that manifest.
        receipt = directory / 'result.json'
        if receipt.exists() or receipt.is_symlink():
            if not receipt.is_symlink() or os.readlink(receipt) != 'manifest.json':
                raise ValueError('Unexpected data-stage completion receipt')
        else:
            receipt.symlink_to('manifest.json')
        path = output / 'configs' / (name + '_launch.json')
        path.write_text(json.dumps(dict(run=str(directory), module=module, args=arguments, environment=environment), indent=2) + '\n')
        jobs.append(dict(config=str(path), gpu=None, requires=[] if name == 'data' else [str(cache / 'manifest.json')]))
    workflow_path = output / 'configs/workflow.json'
    workflow = read(workflow_path)
    for job in workflow['jobs']:
        launch = read(job['config'])
        inputs = json.dumps(launch)
        if '--config' in launch['args']:
            inputs += json.dumps(read(launch['args'][launch['args'].index('--config') + 1]))
        job['requires'] = sorted(set(job.get('requires', [])) | {str(cache / 'manifest.json')})
        if str(geometry / 'manifest.json') in inputs:
            job['requires'].append(str(geometry / 'manifest.json'))
    workflow['jobs'] = jobs + workflow['jobs']
    workflow['training_priority_runs'].insert(0, str(cache))
    workflow['cpu_jobs'] = 1
    workflow_path.write_text(json.dumps(workflow, indent=2) + '\n')
    result = dict(status='scaling_continuation_prepared_not_launched', timestamp=time.time(),
        budget=args.budget, family=args.family, optimizer_steps_per_seed=steps, fitting_roots=len(protocol['subsets']['full']),
        planned_presentations_per_seed=presentations(steps, len(protocol['subsets']['full'])),
        protocol_sha256=digest(protocol_path), source_sha256=digest(__file__),
        base_preparation_sha256=digest(output / 'preparation.json'),
        current_config_sha256={str(p): digest(p) for p in sorted((output / 'configs').glob('*.json'))},
        commands=prepared['commands'], production_actions=[], raw_data_copied=False,
        scope='Configuration and identity checks only. Actual expanded-data training/evaluation has not run. No new untouched-confirmation claim for already opened banks.')
    (output / 'scaling_preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--geometry', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--budget', choices=('updates', 'exposure'), default='updates')
    parser.add_argument('--family', choices=('primary', 'probability_alternative'), default='primary')
    args = parser.parse_args()
    if args.self_test:
        assert presentations(24576, 8965) == 1562598
        assert exposure_steps(8965) == 24576
        for roots in (8966, 10000, 13000, 20000):
            step = exposure_steps(roots)
            target = math.ceil(1562598 * roots / 8965)
            assert presentations(step, roots) >= target > presentations(step - 1, roots)
        try:
            validate(REFERENCE / 'protocol/protocol.json')
        except ValueError as error:
            assert 'gate is closed' in str(error)
        else:
            raise AssertionError('350k must not pass the scaling gate')
        print('PASS: epoch-tail exposure budgets and real 350k gate rejection; no files written')
        return
    if any(getattr(args, key) is None for key in ('protocol', 'cache', 'geometry', 'output')):
        parser.error('--protocol, --cache, --geometry and --output are required')
    generate(args)


if __name__ == '__main__':
    main()
