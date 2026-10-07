#!/usr/bin/env python3
"""Portable preparation, preflight, execution and verification for controlled FALCON."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
COUNTS = (0, 1, 2, 4, 6)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError(f'Unsafe scene/tag name: {name}')
    return name


def resolve(root, value):
    return (root / value).resolve()


def scene_path(root, scene):
    return resolve(root, scene['local_scene']) if 'local_scene' in scene else resolve(REPO, scene['scene_id'])


def validate(root):
    scenes, cases = read(root / 'scenes.json'), read(root / 'cases.json')
    if len({s['name'] for s in scenes}) != len(scenes):
        raise ValueError('Duplicate scenes')
    if len(cases) != len(scenes) * len(COUNTS) or len({c['tag'] for c in cases}) != len(cases):
        raise ValueError('Missing or duplicate matrix entries')
    for scene in scenes:
        group = [c for c in cases if c['scene'] == scene['name']]
        if sorted(c['humans'] for c in group) != list(COUNTS):
            raise ValueError(f"Wrong count matrix for {scene['name']}")
        base = None
        for case in group:
            safe_name(case['tag'])
            ep = read(root / case['episode'])
            if ep['info']['human_num'] != case['humans'] or ep['info']['policy_seed'] != case['seed']:
                raise ValueError(f"Count/seed mismatch: {case['tag']}")
            if ep['scene_id'] != scene['scene_id']:
                raise ValueError('Scene mismatch')
            for h in range(6):
                for k in range(3):
                    for suffix in ('position', 'rotation'):
                        ep['info'][f'human_{h}_waypoint_{k}_{suffix}']
            shared = copy.deepcopy(ep)
            shared.pop('episode_id')
            shared['info'].pop('human_num')
            if base is not None and base != shared:
                raise ValueError(f"Uncontrolled inputs within {scene['name']}")
            base = shared
    return scenes, cases


def prepare(args):
    root, reference = args.output, args.reference.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'{root} is not empty; select a new output directory')
    scenes = read(reference / 'scenes.json')
    root.mkdir(parents=True, exist_ok=True)
    provenance = {}
    cases = []
    for scene in scenes:
        safe_name(scene['name'])
        filename = Path(scene['scene_id']).name
        for subdir in ('scene_configs', 'scene_instances'):
            source = reference / subdir / filename
            if source.exists():
                dest = root / 'scene_instances' / filename
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(source, dest)
                scene['local_scene'] = str(dest.relative_to(root))
                provenance[str(source.relative_to(reference))] = digest(source)
                break
        source = reference / 'navmeshes' / (filename.split('.')[0] + '.navmesh')
        dest = root / 'navmeshes' / source.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(source, dest)
        provenance[str(source.relative_to(reference))] = digest(source)
        scene['navmesh'] = str(dest.relative_to(root))
    for source in sorted((reference / 'matrix_episodes').glob('*.json')):
        ep = read(source)
        scene = next(s for s in scenes if s['name'] == ep['info']['scene_name'])
        tag = safe_name(source.stem)
        dest = root / 'episodes' / source.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(source, dest)
        provenance[str(source.relative_to(reference))] = digest(source)
        cases.append(dict(index=int(tag.split('_')[0]), tag=tag,
                          scene=scene['name'], humans=ep['info']['human_num'],
                          seed=ep['info']['policy_seed'], episode=str(dest.relative_to(root)),
                          episode_sha256=digest(dest), initial_geodesic_m=ep['info']['geodesic_distance']))
    write(root / 'scenes.json', scenes)
    write(root / 'cases.json', cases)
    write(root / 'provenance.json', dict(reference_bundle=reference.name, input_sha256=provenance))
    validate(root)
    print(f'Prepared {len(cases)} exact reference inputs in {root}')


def selected(args, cases):
    names = set(args.scenes.split(',')) if args.scenes else {c['scene'] for c in cases}
    counts = set(map(int, args.humans.split(',')))
    tags = set(args.tags.split(',')) if args.tags else {c['tag'] for c in cases}
    if names - {c['scene'] for c in cases} or tags - {c['tag'] for c in cases} or counts - set(COUNTS):
        raise ValueError('Unknown scene/tag or unsupported human count')
    result = [c for c in cases if c['scene'] in names and c['humans'] in counts and c['tag'] in tags]
    if not result:
        raise ValueError('Selection contains no cases')
    return result


def preflight(root, scenes, cases, checkpoint):
    missing = []
    for scene in scenes:
        if scene['name'] not in {c['scene'] for c in cases}:
            continue
        paths = [scene_path(root, scene), root / scene['navmesh']]
        if scene['scene_dataset'] != 'default':
            paths.append(resolve(REPO, scene['scene_dataset']))
        for path in paths:
            if not path.is_file():
                missing.append(str(path))
    for path in (checkpoint, REPO / 'data/robots/hab_spot_arm',
                 REPO / 'data/robots/spot_data/spot_walking_trajectory.csv',
                 REPO / 'data/humanoids/humanoid_data'):
        if not path.exists():
            missing.append(str(path))
    for command in ('ffmpeg', 'ffprobe'):
        if shutil.which(command) is None:
            missing.append(f'executable: {command}')
    if missing:
        raise ValueError('Missing required assets/tools:\n  ' + '\n  '.join(sorted(set(missing))))


def verify_case(root, case, attempt):
    metrics = read(attempt / 'metrics.json')
    audit = read(attempt / 'crowd_audit.json')
    ep = read(root / case['episode'])
    actions = read(attempt / 'workdir/output/actions.json')
    read(attempt / 'workdir/output/result.json')
    if len(actions) != 1:
        raise ValueError('Expected one evaluated episode')
    runtime_scene, episode_id, _ = next(iter(actions)).split('|')
    values = next(iter(actions.values()))
    steps = int(metrics['metrics']['num_steps'])
    if not (audit['verified'] and audit['requested_humans'] == audit['actual_active_humans'] == case['humans']):
        raise ValueError('Runtime crowd verification failed')
    if audit['episode_id'] != ep['episode_id'] or episode_id != ep['episode_id'] or runtime_scene != audit['scene_id']:
        raise ValueError('Runtime scene/episode mismatch')
    scene = next(s for s in read(root / 'scenes.json') if s['name'] == case['scene'])
    if Path(runtime_scene).resolve() != scene_path(root, scene):
        raise ValueError('Runtime scene differs from selected scene')
    if metrics['tag'] != case['tag'] or metrics['humans'] != case['humans']:
        raise ValueError('Metrics belong to another case/count')
    if len(values) != steps or metrics['frames'] != steps + 1 or steps > metrics['step_limit']:
        raise ValueError('Action/step/frame counts disagree')
    video = attempt / 'four_panel.mp4'
    probe = read_probe(video)
    if int(probe['nb_read_frames']) != metrics['frames'] or (probe['width'], probe['height']) != (1536, 480):
        raise ValueError('Encoded video does not match recorded frames/layout')
    return dict(tag=case['tag'], scene=case['scene'], humans=case['humans'], seed=case['seed'],
                episode_id=episode_id, steps=steps, frames=metrics['frames'],
                video=str(video.relative_to(root)), video_sha256=digest(video),
                attempt=str(attempt.relative_to(root)), **metrics['metrics'])


def verify_completed(root, case, record):
    if record['signature']['episode_sha256'] != digest(root / case['episode']):
        raise ValueError('Episode changed since completion')
    row = verify_case(root, case, root / record['attempt'])
    if row != record['result']:
        raise ValueError('Completed metrics/video changed since verification')
    return row


def read_probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate',
        '-of', 'json', str(path)]))['streams'][0]


def run(args):
    scenes, all_cases = validate(args.output)
    cases = selected(args, all_cases)
    checkpoint = resolve(REPO, args.checkpoint)
    if not args.dry_run:
        preflight(args.output, scenes, cases, checkpoint)
    checkpoint_hash = digest(checkpoint) if not args.dry_run else None
    source_hashes = {str(p.relative_to(REPO)): digest(p) for p in
                     [Path(__file__), *sorted((REPO / 'scripts/controlled').glob('*.py'))]}
    local_diff = subprocess.check_output(['git', 'diff'], cwd=REPO)
    source_hashes['local_tracked_diff'] = hashlib.sha256(local_diff).hexdigest()
    failed = []
    for case in cases:
        scene = next(s for s in scenes if s['name'] == case['scene'])
        signature = dict(episode_sha256=digest(args.output / case['episode']),
                         navmesh_sha256=digest(args.output / scene['navmesh']),
                         scene=scene, sources=source_hashes, step_limit=args.steps,
                         checkpoint_sha256=checkpoint_hash)
        if 'local_scene' in scene:
            signature['scene_instance_sha256'] = digest(scene_path(args.output, scene))
        case_root = args.output / 'runs' / case['tag']
        completed = case_root / 'complete.json'
        if completed.exists() and not args.rerun and not args.dry_run:
            record = read(completed)
            if record['signature'] != signature:
                raise ValueError(f"Changed run inputs for {case['tag']}; use --rerun")
            verify_completed(args.output, case, record)
            print('VERIFIED, SKIP', case['tag'], flush=True)
            continue
        attempt = case_root / (time.strftime('%Y%m%dT%H%M%S') + f'_{time.time_ns() % 1000000000:09}')
        cmd = [sys.executable, '-u', str(REPO / 'scripts/controlled/run_case.py'),
               '--output', str(args.output), '--tag', case['tag'], '--attempt', str(attempt),
               '--steps', str(args.steps), '--checkpoint', str(checkpoint)]
        print('CUDA_VISIBLE_DEVICES=' + os.environ.get('CUDA_VISIBLE_DEVICES', '1') + ' ' + shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        attempt.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ)
        env.setdefault('CUDA_VISIBLE_DEVICES', '1')
        env.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
        env.setdefault('OMP_NUM_THREADS', '1')
        env['PYTHONPATH'] = str(REPO) + os.pathsep + env.get('PYTHONPATH', '')
        start = time.time()
        with (attempt / 'run.log').open('w') as log:
            result = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        execution = dict(command=cmd, returncode=result.returncode, seconds=time.time()-start,
                         cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'])
        write(attempt / 'execution.json', execution)
        try:
            if result.returncode:
                raise ValueError(f'Process exited {result.returncode}')
            row = verify_case(args.output, case, attempt)
            write(completed, dict(attempt=str(attempt.relative_to(args.output)),
                                  signature=signature, result=row))
            print('COMPLETE', case['tag'], flush=True)
        except Exception as exc:
            failed.append(case['tag'])
            print(f"FAILED {case['tag']}: {exc}; log: {attempt / 'run.log'}", flush=True)
    if not args.dry_run:
        summarize(args.output, all_cases)
    if failed:
        raise ValueError('Failed cases: ' + ', '.join(failed))


def summarize(root, cases):
    rows = []
    for case in cases:
        path = root / 'runs' / case['tag'] / 'complete.json'
        if path.exists():
            rows.append(read(path)['result'])
    write(root / 'results.json', rows)
    if rows:
        with (root / 'results.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check', 'run', 'verify', 'generate'])
    parser.add_argument('--output', type=Path, default=REPO / 'evaluation/controlled50')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--scene-manifest', type=Path)
    parser.add_argument('--seed-base', type=int, default=2300)
    parser.add_argument('--scenes', default='')
    parser.add_argument('--humans', default='0,1,2,4,6')
    parser.add_argument('--tags', default='')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--checkpoint', default='pretrained_model/falcon_noaux_25.pth')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.command == 'prepare':
        if not args.reference:
            parser.error('prepare requires --reference (directory containing scenes.json)')
        prepare(args)
    elif args.command == 'generate':
        if not args.scene_manifest:
            parser.error('generate requires --scene-manifest')
        args.scene_manifest = args.scene_manifest.resolve()
        sys.path.insert(0, str(REPO / 'scripts/controlled'))
        from generate import generate
        generate(args)
    elif args.command == 'run':
        run(args)
    else:
        scenes, cases = validate(args.output)
        cases = selected(args, cases)
        if args.command == 'check':
            preflight(args.output, scenes, cases, resolve(REPO, args.checkpoint))
            print(f'PASS: inputs and required asset paths for {len(cases)} cases')
        else:
            for case in cases:
                record = read(args.output / 'runs' / case['tag'] / 'complete.json')
                verify_completed(args.output, case, record)
            print(f'PASS: runtime crowd, actions, metrics and video for {len(cases)} cases')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, StopIteration) as exc:
        sys.exit(f'ERROR: {exc}')
