"""The selected configuration loads (SHA-256 verified) and produces a valid 40x64 bank.

Run: python scripts/stage1_hwm_run.py tests/stage1_hwm/test_selected.py
"""
import json
import numpy as np
import torch

from eventwm_graph.stage1_hwm.api import load_selected, HumanWorlds
from eventwm_graph.stage1_hwm.extract import DEV_MANIFEST, DEV_ROOTS, read
from eventwm_graph.l2_two_stage.types import JointHumanWorlds


def main():
    torch.set_num_threads(1)
    model = load_selected();cfg = model.configuration
    row = [r for r in json.loads(DEV_MANIFEST.read_text())['rows'] if r['split'] == 'dev'][7]
    legal = read(DEV_ROOTS / row['root_id'] / 'legal_root.json.gz');h = legal['history']
    bank = model.predict(h[:-1], h[-1], legal['partial_graph'], worlds=cfg['worlds'], horizon=cfg['horizon_seconds'])
    assert isinstance(bank, HumanWorlds) and isinstance(bank, JointHumanWorlds)
    assert bank.positions.shape[:2] == (cfg['worlds'], cfg['time_samples']) and model.fast is not None
    print('selected %s (%s) loads and predicts a valid bank PASS' % (cfg['name'], cfg['checkpoint_sha256'][:12]))


if __name__ == '__main__':
    main()
