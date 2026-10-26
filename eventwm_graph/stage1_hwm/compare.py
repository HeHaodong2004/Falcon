"""Side-by-side table of scored banks: python -m eventwm_graph.stage1_hwm.compare NAME [NAME ...]"""
import json
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parents[2] / 'output/stage1_hwm/eval'


def g(d, *path):
    for p in path:
        if d is None:return None
        d = d.get(p) if isinstance(d, dict) else None
    return d


ROWS = [
    ('path ES all', ('paths', 'all', 'path_es', 'mean')), ('path ES visible', ('paths', 'visible', 'path_es', 'mean')),
    ('path ES gap', ('paths', 'practical_gap', 'path_es', 'mean')), ('path ES reappear', ('paths', 'reappearance', 'path_es', 'mean')),
    ('path ES 0.5s', ('paths', 'horizons', '0.5', 'path_es', 'mean')), ('path ES 1s', ('paths', 'horizons', '1.0', 'path_es', 'mean')),
    ('path ES 2s', ('paths', 'horizons', '2.0', 'path_es', 'mean')), ('path ES 3s', ('paths', 'horizons', '3.0', 'path_es', 'mean')),
    ('joint ES', ('paths', 'all', 'joint_es', 'mean')),
    ('Cov90 all', ('paths', 'all', 'coverage90', 'mean')), ('Cov90 visible', ('paths', 'visible', 'coverage90', 'mean')),
    ('Cov90 gap', ('paths', 'practical_gap', 'coverage90', 'mean')), ('Cov90 reappear', ('paths', 'reappearance', 'coverage90', 'mean')),
    ('joint Cov90', ('paths', 'all', 'joint_coverage90', 'mean')),
    ('Cov90 obs-self gap all', ('paths', 'all', 'cov90_gap', 'mean')), ('Cov90 obs-self gap visible', ('paths', 'visible', 'cov90_gap', 'mean')),
    ('Cov90 obs-self gap gap', ('paths', 'practical_gap', 'cov90_gap', 'mean')), ('Cov90 self-ref all', ('paths', 'all', 'cov90_self39', 'mean')),
    ('count NLL', ('population', 'all', 'count_nll', 'mean')), ('count RPS', ('population', 'all', 'count_rps', 'mean')),
    ('occ Brier 2m', ('population', 'all', 'occupancy_brier', 'mean')), ('joint set kernel', ('population', 'all', 'joint_set_kernel', 'mean')),
    ('count NLL valid-empty', ('population', 'valid_empty', 'count_nll', 'mean')),
    ('count NLL native-occl', ('population', 'native_occlusion', 'count_nll', 'mean')),
    ('count PIT KS', ('count_calibration', 'all', 'pit_ks')), ('count PIT KS H0', ('count_calibration', 'h0', 'pit_ks')),
    ('P0 forecast H0', ('count_calibration', 'h0', 'p0')), ('P0 observed H0', ('count_calibration', 'h0', 'y0')),
    ('P0 Brier', ('count_calibration', 'all', 'brier_p0', 'mean')),
    ('count CDF ECE k=0', ('count_calibration', 'all', 'cdf_reliability', '0', 'ece')),
    ('count CDF ECE k=1', ('count_calibration', 'all', 'cdf_reliability', '1', 'ece')),
    ('count CDF ECE k=2', ('count_calibration', 'all', 'cdf_reliability', '2', 'ece')),
    ('near occ Brier x100 1s', ('near_occupancy', '1.0', 'brier_x100', 'mean')), ('near occ Brier x100 2s', ('near_occupancy', '2.0', 'brier_x100', 'mean')),
    ('near occ ECE 1s', ('near_occupancy', '1.0', 'reliability', 'ece')),
]
ROWS += [('cand r0.5 Brier', ('candidates', 'r0.5', 'brier', 'mean')), ('cand r0.5 pair acc', ('candidates', 'r0.5', 'pairwise', 'accuracy')),
         ('cand r1 Brier', ('candidates', 'r1', 'brier', 'mean')), ('cand r1 pair acc', ('candidates', 'r1', 'pairwise', 'accuracy')),
         ('cand r1 ECE', ('candidates', 'r1', 'reliability', 'ece')), ('cand min-dist MAE', ('candidates', 'min_distance', 'mae', 'mean')),
         ('cand min-dist rho', ('candidates', 'min_distance', 'within_root_spearman', 'mean'))]
for probe in ('executed', 'stationary'):
    for H in (1, 2, 3):
        for r in ('0.5', '1', '2'):
            key = '%s_H%d_r%s' % (probe, H, r)
            ROWS.append(('%s H%d r%s Brier' % (probe[:4], H, r), ('risk', key, 'brier', 'mean')))
            ROWS.append(('%s H%d r%s ECE' % (probe[:4], H, r), ('risk', key, 'reliability', 'ece')))
    ROWS.append(('%s step Brier' % probe[:4], ('risk', probe + '_step', 'step_brier', 'mean')))
    ROWS.append(('%s social-time err' % probe[:4], ('risk', probe + '_step', 'social_time_error', 'mean')))


def main(names):
    data = [json.loads((EVAL / (n + '_summary.json')).read_text()) for n in names]
    width = max(len(r[0]) for r in ROWS)
    print(' ' * width + ''.join('%14s' % n[:13] for n in names))
    for label, path in ROWS:
        vals = [g(d, *path) for d in data]
        print(label.ljust(width) + ''.join('%14s' % ('-' if v is None else '%.5f' % v) for v in vals))


if __name__ == '__main__':
    main(sys.argv[1:])
