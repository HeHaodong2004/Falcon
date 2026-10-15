"""Data-free API smoke using analytic CV, explicitly not the trained LDM."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import torch
from eventforecastwm.data import nominal_query
from eventforecastwm.model import ForecastWorldModel
from eventquerywm.inputs import tensorize, collate


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    history = json.loads((ROOT/'publication/examples/history.json').read_text())
    legal = collate([tensorize(dict(history=history, history_images=None, query=nominal_query(history)))])
    clocks = torch.tensor([[h['timestamp'] for h in history]], dtype=torch.float64)
    acquisition = torch.tensor([[[h['human_tracks'][i]['last_seen_time'] for h in history]
                                  for i in range(2)]], dtype=torch.float64)
    model = ForecastWorldModel(kind='cv', noise='integrated').eval()
    with torch.no_grad():
        belief = model.encode_history(legal, history_timestamps=clocks, track_acquisition_times=acquisition)
        whole = model.imagine(belief, horizon_s=3.)
        left = model.imagine(belief, horizon_s=1.)
        right = model.imagine(left.next_belief, horizon_s=2.)
        first_rng = torch.Generator().manual_seed(19)
        second_rng = torch.Generator().manual_seed(19)
        paths, _, _ = whole.sample(6, first_rng)
        start, _, state = left.sample(6, second_rng)
        end, _, _ = right.sample(6, second_rng, initial_state=state)
        assert torch.equal(paths, torch.cat([start, end], 2))
        assert torch.isfinite(paths).all()
    print(json.dumps(dict(status='analytic_cv_interface_smoke_passed', trained_ldm=False,
        samples=list(paths.shape), legal_tracks=int(whole.valid.sum()), composition_exact=True)))


if __name__ == '__main__':
    main()
