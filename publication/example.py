"""Run the frozen LDM on authored synthetic legal history, with no target cache."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from verify_artifacts import verify_checkpoint, MANIFEST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Authorized external checkpoint from artifact_manifest.json')
    parser.add_argument('--role', choices=list(json.loads(MANIFEST.read_text())['models']), default='primary')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    proof = verify_checkpoint(args.checkpoint, args.role)
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root / 'scripts'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command = [sys.executable, '-B', '-m', 'eventforecastwm.inference',
        '--checkpoint', proof['path'], '--history', str(root/'publication/examples/history.json'),
        '--sensor', str(root/'publication/examples/sensor.json'), '--output', str(args.output.resolve()),
        '--device', args.device, '--cpu-preparation', '--horizon', '3', '--samples', '6', '--seed', '7']
    subprocess.run(command, cwd=root, env=env, check=True)
    print(json.dumps(dict(role=args.role, input_kind='authored synthetic legal history; no forecasting-accuracy claim',
                         checkpoint_sha256=proof['sha256'], output=str(args.output.resolve()))))


if __name__ == '__main__':
    main()
