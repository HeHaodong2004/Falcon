"""Model-agnostic scoring of saved world banks on development anchors.

Legacy-compatible metrics reuse the legacy evaluator's functions and loops
(``l2_two_stage.evaluate``): tracked path Energy Score / Coverage90 / joint ES,
count RPS, 2 m occupancy Brier and the joint future-set kernel score, with the
same Root->scene reductions and scene bootstrap. Count NLL is exact for models
that expose their count law; legacy unbiased count estimates are read from the
reproduced legacy evaluation.

Additional downstream metrics (new, identical for every model):
  * risk events for two robot probes (the actually executed future path and a
    stationary robot): P(any human within r of the robot during (0,H]) for
    r in {0.5,1,2} m, H in {1,2,3} s -> Brier, log loss, pooled reliability;
  * per-native-step social (1 m) probability Brier and expected social time;
  * 0.5 m near-robot occupancy (10x10 m) Brier and pooled reliability;
  * tracked-path coverage at 50/80/90/95 % (calibration curve).
"""
import argparse
import collections
import glob
import gzip
import json
import math
import time
from pathlib import Path
import numpy as np

from eventwm_graph.l2_two_stage.evaluate import energy, coverage, summarize, occupancy, HORIZONS
from eventwm_graph.l2_two_stage.set_score import joint_future_set_kernel_score
from .data import Store

LEGACY = Path('/home/haodong/workspace/projects/eventwm-falcon-l2-two-stage-20261023')
LEGACY_OUT = LEGACY / 'output/l2_two_stage_20261023'
STRATA = LEGACY / 'docs/l2/two_stage_20261023/reviews/native_occlusion_development/anchors.jsonl.gz'
LEGACY_POP = Path(__file__).resolve().parents[2] / 'output/s1base/output/l2_two_stage_20261023/evaluation/repro_final_forecast'
RADII = (.5, 1., 2.)
RISK_H = (1., 2., 3.)
LEVELS = (.5, .8, .9, .95)
BINS = np.linspace(0, 1, 11)


def legacy_records():
    m = json.loads((LEGACY_OUT / 'prepared/dev_native_v2/manifest.json').read_text());out = {}
    for s in m['shards']:
        for r in np.load(LEGACY_OUT / s['path'], allow_pickle=True):out[(r['root_id'], int(r['anchor_index']))] = r
    return out


def native_strata():
    strata = {}
    if STRATA.exists():
        with gzip.open(STRATA, 'rt') as f:
            for line in f:
                row = json.loads(line);strata[(row['root_id'], row['anchor_index'])] = row
    return strata


def legacy_constants():
    """Model-independent constant-prior scores computed by the legacy run (same truth)."""
    rows = json.loads((LEGACY_POP / 'final_stage1_research_20261024_population.json').read_text())
    return {(r['root_id'], r['anchor'], r['stream'], r['horizon']): r for r in rows}


def iter_banks(path):
    for f in sorted(glob.glob(str(Path(path) / 'bank_*.npy'))):
        for row in np.load(f, allow_pickle=True):yield row


def nearest(times, wanted, tol=1e-6):
    i = np.searchsorted(times, wanted)
    lo = np.clip(i - 1, 0, len(times) - 1);hi = np.clip(i, 0, len(times) - 1)
    j = np.where(np.abs(times[lo] - wanted) <= np.abs(times[hi] - wanted), lo, hi)
    if np.any(np.abs(times[j] - wanted) > tol):raise ValueError('time not native')
    return j


class Reliability:
    def __init__(self):self.n = np.zeros(10);self.p = np.zeros(10);self.y = np.zeros(10)

    def add(self, p, y):
        b = np.clip((np.asarray(p) * 10).astype(int), 0, 9)
        np.add.at(self.n, b, 1);np.add.at(self.p, b, p);np.add.at(self.y, b, y)

    def summary(self):
        n = self.n.sum()
        ece = float(np.sum(np.abs(self.p - self.y)) / max(n, 1))
        return dict(ece=ece, n=int(n), bins=[dict(n=int(a), mean_p=float(b / a) if a else None, freq=float(c / a) if a else None)
                                            for a, b, c in zip(self.n, self.p, self.y)])


def score(bank_dir, out, name, set_score=True, limit=0):
    recs = legacy_records();strata = native_strata();consts = legacy_constants()
    store = Store('dev');root_index = {r: i for i, r in enumerate(store.root_ids)}
    anchor_of = {(store.root_ids[store.a['anchor_root'][i]], int(store.a['j'][i])): i for i in range(store.n)}
    path_rows = [];pop_rows = [];risk_rows = [];occ_rows = [];count_rows = [];query_rows = []
    qpath = Path('/var/tmp/falcon-stage1-claude/hwm_data/dev_queries.pkl')
    import pickle
    queries = pickle.load(open(qpath, 'rb')) if qpath.exists() else None
    rel = collections.defaultdict(Reliability);cov_levels = collections.defaultdict(list)
    started = time.perf_counter();count = 0
    for bank in iter_banks(bank_dir):
        r = recs[(bank['root_id'], int(bank['anchor']))];stream = int(bank['stream'])
        root = r['root_id'];meta = dict(root_id=root, scene_id=r['scene_id'], anchor=r['anchor_index'], stream=stream)
        st = strata.get((root, r['anchor_index']), {})
        if st:
            meta.update(native_occlusion=bool(st.get('has_native_occlusion')),
                        empty_without_in_frustum_human=bool(st.get('cohort_supported') and st.get('valid_empty') and not st.get('has_native_occlusion')))
        times = np.asarray(bank['times'], np.float64);positions = np.asarray(bank['positions'], np.float64)
        present = np.asarray(bank['present'], bool);key_cols = np.asarray(bank['key_cols'], int)
        W = len(positions);n = len(r['keys'])
        tracked = positions[np.arange(W)[:, None, None], np.arange(len(times))[None, :, None], key_cols[:, None, :]] if n else positions[:, :, :0]
        ri = root_index[root];T, truth = store.truth[ri]
        # ---------------- legacy path + population rows ----------------
        for h in HORIZONS:
            hidx = int(np.flatnonzero(np.isclose(np.array([.25, .5, .75, 1., 1.5, 2., 2.5, 3.]), h))[0])
            selected = np.flatnonzero((r['times'] > 0) & (r['times'] <= h + .028001))
            endpoint = r['times'][hidx]
            if endpoint <= 0:continue
            selected = selected[r['times'][selected] <= endpoint + 1e-7]
            timeidx = [int(np.flatnonzero(abs(times - r['times'][j]) < 1e-7)[0]) for j in selected]
            valid = np.flatnonzero((r['physical'] >= 0) & (r['target_mask'][:, hidx] > 0))
            joint_draw = [];joint_truth = []
            for person in valid:
                age = float(r['age'][person]);gap = .75 - 1e-7 <= age <= 1.25 + 1e-7 and r['anchor_index'] > 0
                base = dict(**meta, physical=int(r['physical'][person]), horizon=h, actual_horizon=float(endpoint), age=age,
                            visible=age < 1e-7, practical_gap=bool(gap), reappearance=bool(r['reappearance']))
                draws = tracked[:, timeidx, person];y = r['target'][person, selected]
                cov, distance, radius = coverage(draws, y)
                path_rows.append(dict(**base, path_es=energy(draws, y) / np.sqrt(len(timeidx)), coverage90=cov,
                                      error=distance / np.sqrt(len(timeidx)), radius90=radius / np.sqrt(len(timeidx))))
                x = draws.reshape(W, -1);center = x.mean(0);radii = np.linalg.norm(x - center, axis=-1)
                dist = np.linalg.norm(y.ravel() - center)
                for lv in LEVELS:cov_levels[(h, base['visible'], lv)].append(float(dist <= np.quantile(radii, lv)))
                joint_draw.append(draws);joint_truth.append(y)
            if joint_draw:
                draws = np.stack(joint_draw, axis=2);y = np.stack(joint_truth, axis=1)
                cov, _, _ = coverage(draws, y)
                path_rows.append(dict(**meta, physical=-1, horizon=h, visible=False, practical_gap=False, reappearance=bool(r['reappearance']),
                                      joint_es=energy(draws, y) / np.sqrt(len(timeidx) * len(joint_draw)), joint_coverage90=cov))
            ti = nearest(T, r['timestamp'] + float(endpoint));actual = truth[ti]
            pred = occupancy(positions[:, timeidx[-1]], present, r['robotxy'])
            target = occupancy(actual[None], np.ones((1, len(actual)), bool), r['robotxy'])
            counts = present.sum(-1);max_count = max(int(counts.max()), len(actual)) + 8
            cdf = np.mean(counts[:, None] <= np.arange(max_count)[None], axis=0)
            row = dict(**meta, horizon=h, count_rps=float(np.sum((cdf - (len(actual) <= np.arange(max_count))) ** 2)),
                       occupancy_brier=float(((pred - target) ** 2).mean()), valid_empty=not bool(np.any(r['age'] < 1e-7)),
                       unresolved_prefix=r['unresolved_prefix'], expected_count=float(counts.mean()), true_count=len(actual))
            c = consts.get((root, int(r['anchor_index']), stream, h))
            if c is not None:row.update(constant_count_nll=c['constant_count_nll'], constant_occupancy_brier=c['constant_occupancy_brier'])
            if 'count_logp' in bank and bank['count_logp'] is not None:
                lp = np.asarray(bank['count_logp']);row['count_nll'] = float(-lp[len(actual)]) if len(actual) < len(lp) else float('inf')
            elif c is not None:row['count_nll'] = c['count_nll']
            if set_score:
                native_indices = nearest(T, r['timestamp'] + r['times'][selected])
                s = joint_future_set_kernel_score(positions[:, timeidx], present, truth[native_indices], times=r['times'][selected])
                row['joint_set_kernel'] = s.score
            pop_rows.append(row)
        # ---------------- count calibration (bank-based, identical for every model) ----------------
        a_i = anchor_of[(root, int(r['anchor_index']))]
        y_count = truth.shape[1];counts = present.sum(-1)
        h0 = float(store.a['summary'][a_i][3]) == 0.
        F = lambda k:float(np.mean(counts <= k)) if k >= 0 else 0.
        u = F(y_count - 1) + np.random.default_rng(stream * 7919 + a_i).random() * (F(y_count) - F(y_count - 1))
        count_rows.append(dict(**meta, h0=h0, pit=u, p0=F(0), y0=float(y_count == 0), true_count=y_count, mean_count=float(counts.mean()),
                               **{'cdf%d' % k: F(k) for k in range(4)}, **{'ycdf%d' % k: float(y_count <= k) for k in range(4)}))
        for k in range(4):
            rel[('count_cdf', k)].add([F(k)], [float(y_count <= k)])
            if h0:rel[('count_cdf_h0', k)].add([F(k)], [float(y_count <= k)])
        # ---------------- downstream risk rows ----------------
        frames = store.frame[ri]
        future = (times > 1e-9) & (times <= 3.028001)
        ft = times[future];fidx = nearest(T, r['timestamp'] + ft)
        human = truth[fidx]                                                    # [T',H,2]
        covered = ft <= frames[-1, 0] - r['timestamp'] + 1e-6
        fr = nearest(frames[:, 0], r['timestamp'] + ft[covered])
        here_xz = frames[nearest(frames[:, 0], np.array([r['timestamp']]))][0, [1, 3]]
        bank_all = positions[:, future]
        for probe in ('executed', 'stationary'):
            if probe == 'executed':
                robot = frames[fr][:, [1, 3]];sub = covered;end = frames[-1, 0]
            else:
                robot = np.broadcast_to(here_xz, (len(ft), 2));sub = np.ones(len(ft), bool);end = T[-1]
            bank_f = bank_all[:, sub];fts = ft[sub];hum_s = human[sub]
            if not len(fts):continue
            dmodel = np.linalg.norm(bank_f - robot[None, :, None], axis=-1)
            dmodel = np.where(present[:, None, :], dmodel, np.inf).min(-1) if dmodel.shape[-1] else np.full(dmodel.shape[:2], np.inf)
            dtrue = np.linalg.norm(hum_s - robot[:, None], axis=-1).min(-1) if hum_s.shape[1] else np.full(len(fts), np.inf)
            for H in RISK_H:
                if end < r['timestamp'] + H - .03:continue
                sel = fts <= H + .028001
                for rad in RADII:
                    p = float(np.mean((dmodel[:, sel] < rad).any(1)));y = float((dtrue[sel] < rad).any())
                    pc = min(max(p, 1 / (2 * W)), 1 - 1 / (2 * W))
                    risk_rows.append(dict(**meta, probe=probe, H=H, radius=rad, p=p, y=y, brier=(p - y) ** 2,
                                          logloss=-(y * math.log(pc) + (1 - y) * math.log(1 - pc))))
                    rel[(probe, H, rad)].add([p], [y])
            # per-step 1 m social probability over the available (<=3 s) future
            ps = (dmodel < 1.).mean(0);ys = (dtrue < 1.).astype(float)
            dt = np.diff(np.r_[0., fts])
            risk_rows.append(dict(**meta, probe=probe, H=-1, radius=1., step_brier=float(np.mean((ps - ys) ** 2)),
                                  social_time_error=float(abs((ps * dt).sum() - (ys * dt).sum())), social_time_true=float((ys * dt).sum())))
            rel[(probe, 'step', 1.)].add(ps, ys)
        # ---------------- sibling candidate queries (Root anchors only) ----------------
        if int(r['anchor_index']) == 0 and queries is not None:
            for q in queries.get(root, ()):
                qt = q['poses'][:, 0] - r['timestamp']
                ok = (qt > 1e-9) & (qt <= min(3.028001, T[-1] - r['timestamp'] + 1e-6))
                if not ok.any():continue
                bi = nearest(times, qt[ok]);tix = nearest(T, q['poses'][ok, 0])
                robot = q['poses'][ok][:, [1, 3]]
                dm = np.linalg.norm(positions[:, bi] - robot[None, :, None], axis=-1)
                dm = np.where(present[:, None, :], dm, np.inf).min(-1).min(-1) if dm.shape[-1] else np.full(W, np.inf)
                dtv = np.linalg.norm(truth[tix] - robot[:, None], axis=-1).min() if truth.shape[1] else np.inf
                row = dict(**meta, query=q['query_id'], duration=float(qt[ok][-1]), true_min=float(min(dtv, 5.)),
                           pred_min=float(np.mean(np.minimum(dm, 5.))))
                for rad in (.5, 1.):
                    pp = float(np.mean(dm < rad));yy = float(dtv < rad)
                    row['p%g' % rad] = pp;row['y%g' % rad] = yy
                    rel[('query', rad)].add([pp], [yy])
                query_rows.append(row)
        # ---------------- near-robot 0.5 m occupancy ----------------
        centre = here_xz;bank_f = bank_all
        for H in (1., 2., 3.):
            if T[-1] < r['timestamp'] + H - .03:continue
            k = int(np.argmin(np.abs(ft - H)))
            if abs(ft[k] - H) > .03:continue
            def grid(points, mask):
                g = np.zeros((points.shape[0], 400))
                for w in range(points.shape[0]):
                    pts = points[w][mask[w]] - centre
                    ij = np.floor((pts + 5.) / .5).astype(int);ok = (ij >= 0).all(-1) & (ij < 20).all(-1)
                    g[w, ij[ok, 0] * 20 + ij[ok, 1]] = 1
                return g.mean(0)
            pg = grid(bank_f[:, k], present);yg = grid(human[k][None], np.ones((1, human.shape[1]), bool))
            occ_rows.append(dict(**meta, H=H, fine_brier=float(np.mean((pg - yg) ** 2)) * 100, fine_true=float(yg.sum())))
            nz = (pg > 0) | (yg > 0)
            rel[('occ', H)].add(pg[nz], yg[nz]);rel[('occ', H)].n[0] += float((~nz).sum())
        count += 1
        if limit and count >= limit:break
        if count % 2000 == 0:print(json.dumps(dict(scored=count, seconds=round(time.perf_counter() - started, 1))), flush=True)
    return summarise(name, path_rows, pop_rows, risk_rows, occ_rows, rel, cov_levels, out, count_rows, query_rows)


def ranking(rows, key_p, key_y):
    """Pooled within-Root pairwise ranking accuracy over candidate pairs with different outcomes."""
    groups = collections.defaultdict(list)
    for r in rows:groups[(r['root_id'], r['stream'])].append(r)
    good = 0.;pairs = 0
    for rs in groups.values():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if a[key_y] == b[key_y]:continue
                hi, lo = (a, b) if a[key_y] > b[key_y] else (b, a)
                good += 1. if hi[key_p] > lo[key_p] else .5 if hi[key_p] == lo[key_p] else 0.;pairs += 1
    return dict(accuracy=good / pairs if pairs else None, pairs=pairs)


def spearman_within(rows):
    from scipy.stats import spearmanr
    groups = collections.defaultdict(list);vals = []
    for r in rows:groups[(r['root_id'], r['stream'])].append(r)
    out = []
    for key, rs in groups.items():
        t = [r['true_min'] for r in rs];p = [r['pred_min'] for r in rs]
        if len(rs) >= 3 and np.std(t) > 1e-9 and np.std(p) > 1e-9:
            out.append(dict(root_id=rs[0]['root_id'], scene_id=rs[0]['scene_id'], rho=float(spearmanr(t, p).correlation)))
    return out


def summarise(name, path_rows, pop_rows, risk_rows, occ_rows, rel, cov_levels, out, count_rows=(), query_rows=()):
    s = dict(name=name)
    if query_rows:
        qs = {}
        for rad in (.5, 1.):
            part = [dict(r, b=(r['p%g' % rad] - r['y%g' % rad]) ** 2) for r in query_rows]
            qs['r%g' % rad] = dict(brier=summarize(part, 'b'), base_rate=float(np.mean([r['y%g' % rad] for r in query_rows])),
                                   mean_p=float(np.mean([r['p%g' % rad] for r in query_rows])), pairwise=ranking(query_rows, 'p%g' % rad, 'y%g' % rad),
                                   reliability=rel[('query', rad)].summary())
        rho = spearman_within(query_rows)
        qs['min_distance'] = dict(mae=summarize([dict(r, e=abs(r['pred_min'] - r['true_min'])) for r in query_rows], 'e'),
                                  within_root_spearman=summarize(rho, 'rho'), candidates=len(query_rows))
        s['candidates'] = qs
    cal = {}
    for subset, pred in (('all', lambda r:True), ('h0', lambda r:r['h0'])):
        part = [r for r in count_rows if pred(r)]
        if not part:continue
        pit = np.sort([r['pit'] for r in part])
        cal[subset] = dict(anchor_streams=len(part), pit_ks=float(np.max(np.abs(pit - (np.arange(len(pit)) + .5) / len(pit)))),
                           pit_hist=np.histogram(pit, bins=10, range=(0, 1))[0].tolist(),
                           mean_count=float(np.mean([r['mean_count'] for r in part])), true_mean=float(np.mean([r['true_count'] for r in part])),
                           p0=float(np.mean([r['p0'] for r in part])), y0=float(np.mean([r['y0'] for r in part])),
                           cdf_reliability={k: rel[('count_cdf' if subset == 'all' else 'count_cdf_h0', k)].summary() for k in range(4)},
                           brier_p0=summarize([dict(r, b=(r['p0'] - r['y0']) ** 2) for r in part], 'b'))
    s['count_calibration'] = cal
    groups = {}
    for group, pred in [('all', lambda r:True), ('visible', lambda r:r.get('visible')), ('practical_gap', lambda r:r.get('practical_gap')),
                        ('reappearance', lambda r:r.get('reappearance'))]:
        part = [r for r in path_rows if pred(r)]
        groups[group] = {k: summarize(part, k) for k in ('path_es', 'coverage90', 'joint_es', 'joint_coverage90', 'error', 'radius90')}
    groups['horizons'] = {str(h): {k: summarize([r for r in path_rows if r.get('horizon') == h], k) for k in ('path_es', 'coverage90', 'joint_es')} for h in HORIZONS}
    ages = {'current': lambda a:a < 1e-7, '(0,.75)': lambda a:1e-7 <= a < .75 - 1e-7, '[.75,1.25]': lambda a:.75 - 1e-7 <= a <= 1.25 + 1e-7,
            '(1.25,3]': lambda a:1.25 + 1e-7 < a <= 3, '>3': lambda a:a > 3}
    groups['age'] = {k: {m: summarize([r for r in path_rows if 'age' in r and f(r['age'])], m) for m in ('path_es', 'coverage90')} for k, f in ages.items()}
    s['paths'] = groups
    s['population'] = {g: {k: summarize([r for r in pop_rows if g == 'all' or r.get(g)], k) for k in
                           ('count_nll', 'constant_count_nll', 'count_rps', 'occupancy_brier', 'constant_occupancy_brier', 'joint_set_kernel', 'expected_count', 'true_count')}
                       for g in ('all', 'valid_empty', 'native_occlusion', 'empty_without_in_frustum_human')}
    risk = {}
    for probe in ('executed', 'stationary'):
        for H in RISK_H:
            for rad in RADII:
                part = [r for r in risk_rows if r['probe'] == probe and r['H'] == H and r['radius'] == rad]
                key = '%s_H%g_r%g' % (probe, H, rad)
                risk[key] = dict(brier=summarize(part, 'brier'), logloss=summarize(part, 'logloss'), base_rate=float(np.mean([r['y'] for r in part])) if part else None,
                                 mean_p=float(np.mean([r['p'] for r in part])) if part else None, reliability=rel[(probe, H, rad)].summary())
        part = [r for r in risk_rows if r['probe'] == probe and r['H'] == -1]
        risk[probe + '_step'] = dict(step_brier=summarize(part, 'step_brier'), social_time_error=summarize(part, 'social_time_error'),
                                     reliability=rel[(probe, 'step', 1.)].summary())
    s['risk'] = risk
    s['near_occupancy'] = {str(H): dict(brier_x100=summarize([r for r in occ_rows if r['H'] == H], 'fine_brier'), reliability=rel[('occ', H)].summary()) for H in (1., 2., 3.)}
    s['coverage_levels'] = {'%g_%s_%g' % (h, 'visible' if v else 'stale', lv): dict(mean=float(np.mean(x)), n=len(x)) for (h, v, lv), x in sorted(cov_levels.items())}
    out = Path(out);out.mkdir(parents=True, exist_ok=True)
    (out / (name + '_summary.json')).write_text(json.dumps(s, indent=1))
    np.save(out / (name + '_rows.npy'), np.asarray(dict(path=path_rows, population=pop_rows, risk=risk_rows, occupancy=occ_rows, count=list(count_rows)), dtype=object), allow_pickle=True)
    return s


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('bank');p.add_argument('name');p.add_argument('--out', default=str(Path(__file__).resolve().parents[2] / 'output/stage1_hwm/eval'))
    p.add_argument('--no-set-score', action='store_true');p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    s = score(a.bank, a.out, a.name, not a.no_set_score, a.limit)
    print(json.dumps(dict(path_es=s['paths']['all']['path_es']['mean'], coverage90=s['paths']['all']['coverage90']['mean'],
                          count_nll=s['population']['all']['count_nll']['mean'], occ=s['population']['all']['occupancy_brier']['mean'])))
