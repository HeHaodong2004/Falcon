"""Package measured results and generate reports. Never reruns inference/fitting."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from l25_verify_baseline import OLD, CHECKPOINT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, text):
    with Path(path).open('x') as f:
        f.write(text)


def dump(path, value):
    write(path, json.dumps(value, indent=2, allow_nan=False)+'\n')


def check(artifacts):
    manifest = read(artifacts/'manifest.json')
    assert all(sha(artifacts/p) == h for p, h in manifest['files'].items())
    packet = read(artifacts/'decision_packet.json')
    for outcome in packet['outcomes']:
        assert outcome['p_l4_query_completion'] is None and outcome['p_episode_success'] is None
        if outcome['candidate_legal']:
            assert 0 <= outcome['p_nominal_collision'] <= 1
            assert abs(sum(outcome['collision_first_mass'])-outcome['p_nominal_collision']) < 1e-9
    assert all(not branch['root_executable'] for branch in packet['branches'])
    g = packet['graph']; n = len(g['node_features'])
    assert all(0 <= i < n for row in g['edge_index'] for i in row)
    assert len(g['node_features'][0]) == len(g['feature_names']) == 60
    assert len(g['query_node_indices']) == len(g['query_ids']) == len(packet['outcomes'])
    print(json.dumps(dict(manifest_valid=True, packet_valid=True, artifacts=str(artifacts))))


def performance_details(root):
    import numpy as np
    demo = read(root/'demo_release/run.json')
    first = demo['startup'][0]
    bench = read(root/'benchmark_release/run.json')
    probe = np.quantile([r['elapsed_ms'] for r in bench['cache_probes']], [.5, .95, .99])
    wall = next(line.rsplit(': ', 1)[-1].strip() for line in (root/'startup_process_time.txt').read_text().splitlines() if 'Elapsed (wall clock)' in line)
    text = f'''\n## Startup, cache probes and selected sensitivity cells

New process with an initially empty isolated Numba cache: demo process wall time
{wall} h:mm:ss (includes imports, prefix reconstruction, inference and plotting); model
load after imports {demo['model_load_ms']:.3f} ms; first legal-prefix reconstruction/JIT
{first['prefix_reconstruction_ms']:.3f} ms; first complete inference
{first['first_call']['full_ms']:.3f} ms. The complete `/usr/bin/time -v` ledger is
committed. This startup is substantially slower than loaded-process calls.

Repeated fixed-belief fully cached candidate evaluation, separate from full
calls: P50/P95/P99 {probe[0]:.3f}/{probe[1]:.3f}/{probe[2]:.3f} ms, 83 probes.
These results do not establish the 10 ms incremental gate.

| Root class | Actual queries | Worlds | Requested event expansions | Actual branch counts over 4 calls | Evaluation P95 ms | Incremental P95 ms |
|---|---:|---:|---:|---|---:|---:|
'''
    ablation = read(root/'ablation_release/ablation.json')
    for row in ablation['summary']:
        if row['requested_candidates'] == 8 and row['worlds'] in (16, 64):
            part = [c for c in ablation['calls'] if c['dense']==row['dense'] and c['requested_candidates']==8 and
                    c['worlds']==row['worlds'] and c['branch_queries']==row['branch_queries']]
            counts = dict(Counter(c['branch_count'] for c in part))
            text += f"| {'dense' if row['dense'] else 'representative'} | {row['actual_candidates']} | {row['worlds']} | {row['branch_queries']} | {counts} | {row['p95_evaluation_ms']:.3f} | {row['p95_incremental_ms']:.3f} |\n"
    return text+'\nThese are small sequential sensitivity cells under host variability, not causal branch-cost estimates. Zero actual branches reflects the zero-information guard. They cannot establish a full-call speed pass.\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=BASE/'output/l25_query_outcomes_v1_20261022')
    p.add_argument('--docs', type=Path, default=BASE/'docs/l25/query_outcomes_v1_20261022')
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    root, doc = args.results, args.docs
    art = doc/'artifacts'
    if args.check:
        check(art); return
    art.mkdir(parents=True, exist_ok=False)
    bench = read(root/'benchmark_release/run.json')
    science = read(root/'science_release/run.json')
    demo = read(root/'demo_release/run.json')
    ablation = read(root/'ablation_release/ablation.json')
    assert not bench['errors'] and not science['errors'] and not demo['errors']
    for result in (bench, science, demo):
        assert result['sources_unchanged'] and result['checkpoint_unchanged']
        assert all(sha(BASE/f) == h for f, h in result['source_sha256'].items())
    selected = read(root/'benchmark_release/cohort.json')
    assert len(selected) == 83 and len({r['scene_id'] for r in selected}) == 59
    assert read(root/'baseline_regressions_retry/validation.json')['total_tests'] == 399
    assert 'Ran 47 tests' in (root/'tests_release.log').read_text() and '\nOK\n' in (root/'tests_release.log').read_text()
    snapshot = read(OLD/'snapshot/manifest.json')
    entries = {r['root_id']: r for r in snapshot['rows']}
    inputs = {}
    for meta in selected:
        entry = entries[meta['root_id']]
        assert entry['split'] == 'dev' and entry['source_split'] == 'train'
        inputs[meta['root_id']] = {}
        for name in ('legal_root.json.gz', 'queries.json.gz', 'privileged_future.npz'):
            path = OLD/'snapshot/roots'/meta['root_id']/name
            actual = sha(path)
            assert actual == entry['files'][name]
            inputs[meta['root_id']][name] = actual
    dump(art/'input_manifest.json', dict(snapshot=str(OLD/'snapshot'), snapshot_manifest_sha256=sha(OLD/'snapshot/manifest.json'),
                                        development_cache_sha256=sha(OLD/'dev.npy'), checkpoint=str(CHECKPOINT),
                                        checkpoint_sha256=sha(CHECKPOINT), roots=inputs, all_match_frozen_snapshot=True,
                                        confirmation_payloads_opened=False))
    copies = {
        'baseline_smoke.json': 'baseline_smoke.json', 'baseline_native.json': 'baseline_native.json',
        'baseline_regressions_retry/validation.json': 'baseline_regressions.json',
        'tests_release.log': 'l25_tests.log', 'benchmark_release/run.json': 'benchmark.json',
        'benchmark_release/calls.jsonl': 'benchmark_calls.jsonl', 'benchmark_release/cohort.json': 'cohort.json',
        'benchmark_final/run.json': 'benchmark_before_metadata_hardening.json',
        'science_release/run.json': 'scientific.json', 'ablation_release/ablation.json': 'ablation.json',
        'demo_release/decision_packet.json': 'decision_packet.json', 'demo_release/consumer.json': 'consumer.json',
        'demo_release/context_graph.png': 'context_graph.png', 'demo_release/query_outcomes.png': 'query_outcomes.png',
        'demo_release/run.json': 'demo.json', 'startup_process_time.txt': 'startup_process_time.txt'}
    copies.update({'illustration/decision_packet.json': 'illustrative_decision_packet.json',
                   'illustration/consumer.json': 'illustrative_consumer.json', 'illustration/selection.json': 'illustration_selection.json',
                   'illustration/context_graph.png': 'illustrative_context_graph.png',
                   'illustration/query_outcomes.png': 'illustrative_query_outcomes.png'})
    for source, target in copies.items():
        shutil.copyfile(root/source, art/target)
    for source, target in [('science_release/labels.json', 'scientific_labels.json.gz'),
                           ('science_release/censored.json', 'scientific_censoring.json.gz')]:
        (art/target).write_bytes(gzip.compress((root/source).read_bytes(), mtime=0))
    predictions = [r for path in sorted((root/'science_release').glob('predictions_*.json')) for r in read(path)]
    (art/'scientific_predictions.json.gz').write_bytes(gzip.compress(json.dumps(predictions, allow_nan=False).encode(), mtime=0))
    calls = [json.loads(line) for line in (root/'benchmark_release/calls.jsonl').read_text().splitlines()]
    last = {c['root_id']: c for c in calls}
    caches = {}
    for name in calls[0]['cache_statistics']:
        hits = sum(c['cache_statistics'][name]['hits'] for c in last.values())
        misses = sum(c['cache_statistics'][name]['misses'] for c in last.values())
        caches[name] = dict(hits=hits, misses=misses, hit_rate=hits/(hits+misses) if hits+misses else None)
    counts = dict(calls=len(calls), candidate_evaluations=sum(c['candidates'] for c in calls),
                  unsupported=sum(c['unsupported'] for c in calls), censored=sum(c['censored'] for c in calls),
                  budget_truncated=sum(c['budget_truncated'] for c in calls), branches=dict(Counter(c['branches'] for c in calls)),
                  candidate_counts=dict(Counter(c['candidates'] for c in calls)), caches_after_last_call_including_prime=caches)
    dump(art/'workload_summary.json', counts)
    source_commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', 'eventwm_graph/l25/compiler.py'], cwd=BASE, text=True).strip()
    decisions = dict(BASELINE_RUNNABLE='PASS', API_INTEGRATION='PASS', GRAPH_CORRECTNESS='PASS',
                     QUERY_OUTCOME_CORRECTNESS='PASS', CONDITIONAL_SUCCESSOR='REVISE', CAUSAL_VALIDITY='PASS',
                     GNN_COMPATIBILITY='PASS', INCREMENTAL_LATENCY='REVISE', FULL_CALL_LATENCY='REVISE')
    dump(art/'acceptance.json', dict(decisions=decisions, scope='PASS for declared bounded semantics, not native mesh/PSC/L4 accuracy',
                                    implementation_commit=source_commit, branch='research/l25-query-outcomes-20261022',
                                    unsupported=['official mesh contact', 'official geodesic PSC', 'exact sensor/identity reobservation',
                                                 'L4 completion', 'episode success', 'automatic Bellman targets', 'hard deadlines']))
    perf = '# Measured performance\n\n**INCREMENTAL LATENCY REVISE; FULL CALL LATENCY REVISE.**\n\n'
    perf += 'Release source/checkpoint stayed unchanged throughout measurement. All 581 calls completed across 83 development roots. The target is ≤10 ms incremental and ≤100 ms full-call P95. The release cold and dense workloads fail. No training or GPU work occurred.\n\n'
    perf += '| Cohort/cache | Calls | Incremental P50/P95/P99 ms | Full P50/P95/P99 ms | Candidate/s P50/P95/P99 |\n|---|---:|---|---|---|\n'
    for key, value in bench['quantiles'].items():
        m = value['metrics']
        triple = lambda k: '/'.join(f'{m[k][q]:.3f}' for q in ('p50', 'p95', 'p99'))
        perf += f"| {key} | {value['attempts']} | {triple('incremental_ms')} | {triple('full_ms')} | {triple('candidate_throughput_per_s')} |\n"
    perf += f'''\n## Workload and accounting

Historical metadata-only selection: first 60 development roots plus 12 densest
and 12 longest-history roots, deduplicated to 83; 59 scenes. Five warm and two
dynamically cold repeats per root. Up to eight existing native candidates are
requested; actual counts are {counts['candidate_counts']}. Static maps have
738–3,783 arrived cells, 68–535 tiled nodes; tracked component maxima 0–6.
Forecast: 3 s, 16 worlds, 64 timestamps, hidden population included. All core
risk/social/time/progress/optical outputs and typed graph export are enabled.
Branch selection considers one informative event, up to two finite successors;
branch-count frequencies are {counts['branches']}. Zero branches means no
informative event, not an unreported performance shortcut.

Full call starts before real `observe`, then includes context/readout, bank,
candidate compilation and graph export. It excludes upstream reference
generation, prefix reconstruction, imports/JIT and the untrained score consumer.
Incremental subtracts the measured forecast time and actual update time, and
includes context generation. Route/tile/topology caches are versioned static
work; dynamic cold retains them but clears context, bank, numeric and branch
caches plus inherited route/footprint/kernel caches. Warm bank/numeric hits occur
for repeat zero matching the primed seed (20% of warm calls); dynamic cold banks
are fresh. Actual observation updates are never cached. The fixed-belief cache
probe is separate from these complete calls. Full cumulative cache counters
(including priming) are in `artifacts/workload_summary.json`.

Across {counts['candidate_evaluations']} repeated candidate evaluations,
{counts['unsupported']} ({100*counts['unsupported']/counts['candidate_evaluations']:.2f}%) are static-unsupported and
{counts['censored']} ({100*counts['censored']/counts['candidate_evaluations']:.2f}%) have censored nominal tails. No root is
dropped, and {counts['budget_truncated']} branch requests are budget-truncated in this gate: the
optional-branch soft budget is deliberately 1e6 ms, so speed does not come from
deadline-driven output omission. Unsupported candidate masks remain explicit.

Peak process RSS {bench['peak_rss_bytes']/2**20:.1f} MiB includes Torch/Numba and the frozen
development metadata cache. Model load after imports: {bench['model_load_ms']:.3f} ms. Process
and first-call startup are separate: `artifacts/startup_process_time.txt` records
a new demo process with an initially empty isolated Numba cache, including
imports, legal prefix/JIT, first inference and plotting. Its per-stage timings
are in `artifacts/demo.json`; they are not steady-state latency.

## Replication and failures

The pre-hardening run measured representative full P95 75.228/89.792 ms and
dense 82.017/99.945 ms. After immutable nested metadata was added, the release
run above was slower. Both are retained; no fastest-run selection or causal
claim about the metadata patch is made. Runtime varies under the independent
collector and other host activity. The release figures govern acceptance.
The original L2 report's 87.446/88.275 ms and the new dense-enriched 11-root
gate's 117.358/131.940 ms remain separate workloads. The gate used native
`initialize`; this complete benchmark uses the historical seeded legal-prefix
conversion. These are not matched speedup estimates.

Exploratory profiling found Python static-edge construction, route lookup and
fragmented height regions. Vectorized static export and versioned route caches
reduced overhead. Early runs also repaired a public-goal fixture; they are not
a controlled optimization ablation. The remaining cold swept-support checks,
actual observation update, per-query reductions and coarse successor construction
still cost time. The frozen L2 was not retrained or restructured.

## Bounded fallback and sensitivity

`artifacts/ablation.json` retains 192 calls on two fixed legal roots, four repeats,
1/4/8 requested candidates (actual counts retained), 8/16/32/64 worlds and zero
or one event expansion. Those timings include a fresh bank and all candidate
core features, but **exclude** actual update/context creation, so they are not
full-call passes. Increasing worlds is not hidden behind a constant workload.

The soft-budget-zero fallback returns the same supported nominal risk values,
retains all candidates/core features and puts optional unexpanded observation
mass in explicit residuals. The ablation checks equality. It cannot preempt L2
or guarantee a full-call deadline. Reducing worlds is an explicit caller choice
with reduced finite-bank support, not a default silent dense-scene speed fix.
Dense/unbounded deployment therefore remains REVISE.
'''
    write(doc/'PERFORMANCE.md', perf+performance_details(root))
    scientific = '# Scientific validation\n\n**Forecasting/belief quality REVISE; descriptive development audit only.**\n\n'
    scientific += 'Frozen trained checkpoint, no fitting/tuning/confirmation access. 83 frozen development roots / 59 scenes. Predictions are serialized before offline truth is opened; only legal history/graph/references enter the compiler. All selected payload hashes match the original frozen snapshot.\n\n'
    scientific += '| Target/alignment | Requested H s | Rows / roots / scenes | Positive | Scene/root-balanced Brier [95% CI] | AUROC | Within-root ranking | ECE |\n|---|---:|---|---:|---|---:|---:|---:|\n'
    for key, m in science['science'].items():
        target, alignment, horizon = key.split(':')
        lo, hi = m['brier_95_scene_bootstrap']
        fmt = lambda x: f'{x:.3f}' if x is not None else 'unsupported'
        scientific += f"| {target}/{alignment} | {horizon} | {m['n']}/{m['roots']}/{m['scenes']} | {m['positives']} | {m['brier_scene_root_balanced']:.4f} [{lo:.4f},{hi:.4f}] | {fmt(m['auroc'])} | {fmt(m['within_root_ranking_concordance'])} | {m['ece']:.4f} |\n"
    censoring = Counter(r['status'] for r in read(root/'science_release/censored.json'))
    scientific += f'''
## Targets, time support and inference boundary

Collision target is the audited all-phase rectangle/disk enclosing-body overlap
of the **nominal reference** with the recorded exogenous human center paths.
Social target is horizontal Euclidean center distance <1 m. These targets match
the declared readouts. They are not native articulated mesh contact, installed
geodesic/XYZ PSC, L4 closed-loop completion or episode success. No old follower
completion label is reused as L4 supervision. No RL experiment occurred.

Exact targets require all nominal 48 ms endpoints to occur on the recorded
native tape. Some actual intervals are 56 ms. Censor ledger: {dict(censoring)}.
Exact 1/2/3 s rows coincide because only shorter queries retain complete exact
clock support; these are not independent long-horizon confirmations. The table's
H is the requested forecast cap; actual covered duration is min(query duration,
native horizon), retained per row in the compressed label ledger.

The separately named interpolated auxiliary target uses the existing L2 offline
`truth_at` linear interpolator at the **same absolute nominal times**, with no
extrapolation. It supports 471 legal candidates per requested horizon. It tests
the finite center-path/envelope proxy on time-aligned paths, not exact native
execution/contact. Both cohorts are reported rather than silently rounding
timestamps or discarding clock failures.

## Metrics and uncertainty

Brier intervals use 1,000 fixed-seed scene bootstrap samples: candidates average
within root, then roots within scene. AUROC/ECE/reliability are descriptive row
metrics; within-root ranking compares opposite-label candidates with half-credit
ties. Five predeclared probability bins, including empty-bin counts, are stored
in `artifacts/scientific.json`; the standalone reliability plot uses the auxiliary
interpolated nominal envelope target. Intervals exclude training-seed variation,
geometry error, interpolation error and selection reuse. Multiple correlated
horizons/candidates do not create independent evidence. No statistical claim
is made from the point AUROCs alone.

Sixteen root worlds provide coarse finite-bank support. Zero sampled collisions
does not mean safety. Effective sample support and an explicitly conditional
zero-hit iid reference bound are exposed; the iid coverage assumption is not
established for the inherited stochastic carrier. There is no guarantee against
model error, unknown populations or temporal aliasing. Social region `mass` is
weighted endpoint exposure count (multiply by .048 for duration), not an any-event
probability; collision and first-observation region masses are first-event mass.

## Inherited limitations and decision

The original 4.2% native-gap path ES decrease has paired difference −0.0247 and
95% CI [−0.0905,+0.0479]; improvement is **not statistically established**.
Belief/count support, duplicate/recycled ancestry, missing preferred tracks,
association/reobservation, speed/turning and broad hidden spatial priors remain
unresolved. The selected development scenes were already used by L2; this is
not fresh confirmation or a model-accuracy certification.

Coarse any/none likelihoods conserve their finite optical law and never mutate
actual belief, but do not prove legal exact visibility, native occlusion,
identity reobservation or useful Bellman transitions. Reobservation probability,
static revelation, exact sensor-conditioned belief, official contact/PSC,
L4 completion and full navigation success remain unsupported. Core compilation
works independently of these capabilities. Tests establish implementation
correctness for the declared law; scientific readiness remains REVISE.
'''
    scientific += '\nThe additional illustrative graph/outcome figures use a qualitative root chosen by maximum predicted candidate-risk contrast, without truth labels. See `illustration_selection.json` and `illustrative_decision_packet.json`; this does not change the fixed benchmark/scientific cohorts.\n'
    write(doc/'SCIENTIFIC_VALIDATION.md', scientific)
    repwarm = bench['quantiles']['representative:warm']['metrics']
    repcold = bench['quantiles']['representative:dynamic_cold']['metrics']
    densewarm = bench['quantiles']['dense:warm']['metrics']
    densecold = bench['quantiles']['dense:dynamic_cold']['metrics']
    final = f'''# Final L2.5 research report

**Decision: restricted experimental integration ready; overall REVISE for
unrestricted planning/deployment.** The compiler is runnable and reviewed, but
the incremental/cold/dense latency gates and inherited scientific limitations
remain open. There is no RL improvement or L4 success-probability claim.

Branch `research/l25-query-outcomes-20261022`, based exactly on
`cf1971228c37517202cc9e5787f785a3876a0514`.
Implementation commit `{source_commit}`. The subsequent report/artifact commit
is the branch HEAD (`git rev-parse HEAD`), reported in the delivery message.
Checkpoint SHA256 `{bench['checkpoint_sha256']}`;
model `{bench['model_version']}`. Frozen L2 source/weights were not changed.

| Acceptance category | Decision | Scope |
|---|---|---|
| BASELINE_RUNNABLE | PASS | Trained checkpoint, native history/query calls and 399 unchanged regressions/oracles |
| API_INTEGRATION | PASS | initialize/observe/predict_shared/track readout/graph/optics/finite successors |
| GRAPH_CORRECTNESS | PASS | Arrived connected regions, typed relations, repeated route occurrences, explicit unknown |
| QUERY_OUTCOME_CORRECTNESS | PASS | Declared finite-bank endpoint envelope and horizontal proxy law; official contact/PSC unsupported |
| CONDITIONAL_SUCCESSOR | REVISE | Coarse finite any/none works; exact legal sensor/reobservation/occlusion unsupported |
| CAUSAL_VALIDITY | PASS | Legal input boundary, exact absolute nominal alignment, actual/imagined separation and masks |
| GNN_COMPATIBILITY | PASS | Typed tensors, root candidate IDs/masks, disjoint batching and permutation equivariance |
| INCREMENTAL_LATENCY | REVISE | Representative release P95 {repwarm['incremental_ms']['p95']:.3f} ms warm / {repcold['incremental_ms']['p95']:.3f} ms cold vs 10 ms |
| FULL_CALL_LATENCY | REVISE | Representative release P95 {repwarm['full_ms']['p95']:.3f}/{repcold['full_ms']['p95']:.3f} ms; dense {densewarm['full_ms']['p95']:.3f}/{densecold['full_ms']['p95']:.3f} ms vs 100 ms |

## Delivered

- Two-stage context and candidate APIs, a small native-reference compatibility
  adapter, versioned immutable state summaries, shared future inference and
  bounded caches. No competing L4 compiler or permanent action set.
- Whole-prefix joint risk, temporal/spatial first events, social exposure proxy,
  minimum-distance quantiles, public-goal/observed-topology progress, nominal
  timing/censoring and support/provenance masks. Unsupported values remain null.
- Sparse finite observation successors with conserved represented/residual mass,
  zero-information suppression and no future action enabled as a root candidate.
- GNN-compatible export, an untrained message-passing consumer, real native demo,
  serialized packet, graph/outcome/reliability figures and reproducible drivers.
- 47 new tests plus 399 inherited tests/oracles: **446 tests passed**. Native
  release benchmark: 581 complete calls on 83 roots, no root errors. Scientific
  release audit: 83 development roots, frozen input hashes verified. Source and
  checkpoint stability passed; ownership-hardening labels reproduce exactly.
- 192 sensitivity calls and a checked core-preserving optional-branch fallback.
  It is not a hard-deadline or complete-feature speed pass.

Real L2 callables used: `IntermittentWorldModel.load`, `initialize`, `observe`,
`predict_shared`, `get_previously_observed_tracks`, `predict_observations`
(direct baseline/oracle), `condition_observation_event`, `successor_at`
(through conditioning), `graph.materialize/get`. Optical batching reuses the
same L2 `observation_law.detection` and is tested against the public API. The
baseline gate also exercises `project_future_distribution` and actual-belief
serialization; tests continue held successor tapes. All inference is CPU only.

## Immediate integration conditions

L3 builds the arrived context, proposes current legal references upstream and
consumes one outcome/logit per query ID with masks. It must keep censored tails,
external unknown support, finite-bank approximations and unsupported success
signals visible. L4 retains canonical reference ownership and independent
collision avoidance. It must provide an explicit final compiler/timing adapter
and actual execution/observation feedback. New actual observations update the
separate actual belief, never a hypothetical branch. See
[INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md) for unresolved hash, clock,
executor-profile and status details.

No collector, production/main branch, shared data or L4 worktree was modified;
no checkpoint was rewritten, training started or confirmation data unsealed.

## Deliverable paths

- Worktree: `/research-user/workspace/projects/eventwm-falcon-l25-query-outcomes-20261022`
- Code: `eventwm_graph/l25/`; tests: `tests/l25/`
- Reproduction: `scripts/l25_verify_baseline.py`, `l25_run.py`, `l25_ablate.py`, `l25_package.py`
- Documents: `docs/l25/query_outcomes_v1_20261022/`
- Raw results: `output/l25_query_outcomes_v1_20261022/`
- Committed small results: `docs/l25/query_outcomes_v1_20261022/artifacts/`

Full commands are in [README.md](README.md), semantics in
[ARCHITECTURE.md](ARCHITECTURE.md), audit in
[SCIENTIFIC_VALIDATION.md](SCIENTIFIC_VALIDATION.md), performance in
[PERFORMANCE.md](PERFORMANCE.md), and inspection findings in [REVIEW.md](REVIEW.md).
'''
    write(doc/'FINAL_REPORT.md', final)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), sharex=True, sharey=True)
    for ax, h in zip(axes, (.5, 1., 2., 3.)):
        metric = science['science']['nominal_envelope:interpolated:'+str(h)]
        bins = [b for b in metric['reliability'] if b['n']]
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.plot([b['predicted'] for b in bins], [b['frequency'] for b in bins], 'o-', color='#155e75')
        ax.set(title=f'Requested H={h:g}s', xlabel='Predicted risk', xlim=(0, 1), ylim=(0, 1))
    axes[0].set_ylabel('Nominal envelope target frequency')
    fig.suptitle('Development only · interpolated nominal endpoint proxy · not L4 completion', fontsize=10)
    fig.tight_layout(); fig.savefig(art/'reliability.png', dpi=150); plt.close(fig)
    dump(art/'manifest.json', dict(implementation_commit=source_commit, files={str(f.relative_to(art)): sha(f) for f in sorted(art.iterdir()) if f.is_file() and f.name != 'manifest.json'}))
    check(art)


if __name__ == '__main__':
    main()
