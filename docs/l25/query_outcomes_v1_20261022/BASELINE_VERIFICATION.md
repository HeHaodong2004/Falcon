# Frozen L2 verification — gate PASS

Verified before L2.5 implementation on 2026-10-22. Source and new-worktree HEAD:
`cf1971228c37517202cc9e5787f785a3876a0514`, clean. Dedicated branch:
`research/l25-query-outcomes-20261022`. Original worktree remained clean.

Checkpoint (read only):
`/research-user/workspace/projects/eventwm-falcon-l2-intermittent-500k-20261022/output/l2_intermittent_500k_20261022/training/checkpoint.pt`
SHA256 `215864a013c58c36a670b0aef3c4512a6eb651859f5ad24bb1ee899844189223`.
Runtime model `l2-intermittent-1-fc2c500cda1b243c7526`. 33,309 parameters, width 64,
32-component carrier, seed 221. CPU inference; checkpoint training configuration
records CUDA training, 12,079 train / 666 selection roots and CTMC Stop/Move OU
motion. Full configuration is preserved in `baseline_native.json`.

## Actual execution

Environment `/research-user/miniforge3/envs/eventwm-falcon/bin/python` (Python 3.9).
All commands run from this worktree with:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="$PWD/output/l25_query_outcomes_v1_20261022/cache/numba"
PY=/research-user/miniforge3/envs/eventwm-falcon/bin/python
CK=/research-user/workspace/projects/eventwm-falcon-l2-intermittent-500k-20261022/output/l2_intermittent_500k_20261022/training/checkpoint.pt
nice -n 19 ionice -c 3 "$PY" -m eventwm_graph.l2_intermittent.api_example --checkpoint "$CK"
nice -n 19 ionice -c 3 "$PY" scripts/l25_verify_baseline.py --output output/l25_query_outcomes_v1_20261022/baseline_native.json
```

Artifact commands reject existing destination files. Use a fresh output filename
when reproducing. The complete unchanged regression/oracle driver was invoked as:

```python
from pathlib import Path
import eventwm_graph.l2_intermittent.validate as v
v.OUT = Path('output/l25_query_outcomes_v1_20261022').resolve()
v.run('baseline_regressions_retry')
```

All 399 regressions passed in 62.619 s; inherited and optimized
sources unchanged, numerical oracles passed. An initial test-driver invocation
used a relative output path and failed in its log-relative-to-repository check;
the absolute-path invocation above repaired the invocation, with no source edit.

Native smoke: first 8 frozen development roots plus 3 metadata-selected densest
roots. Only legal histories and queries enter inference. 11/11 roots initialized,
actual last observations assimilated, 22 distinct query evaluations supported,
and shared tapes aligned. Graph materialization consumed 960–3,783 cells.
Tracked future shapes `[16,64,0..6,2]`; hidden futures are separately preserved.
66/66 complete timing calls passed validity/ownership checks. The synthetic public
example also verified a 1 s missing interval, exact new arrived-center assimilation
(error 0), actual-belief JSON roundtrip, two queries on one bank, imagined
successor at 1.296 s and held-tape continuation to 4.296 s.

## Exact callable and semantics map

| Callable | Input/output / semantics |
|---|---|
| `IntermittentWorldModel.load(path)` | Frozen trained checkpoint, CPU; no training |
| `initialize(legal_history, observed_graph)` | Arrived history + arrived support snapshot → persistent actual `Belief`; native history filtering then conversion |
| `observe(actual_belief, observation)` | New arrived sensor row → new actual belief; imagined inputs rejected |
| `predict_shared(b, horizon, num_worlds, seed, times=None)` | Shared `JointFuture`: positions/velocities `[W,T,N,2]`, hidden positions `[W,T,H,2]`, normalized weights `[W]`, slots per world, endpoint carrier |
| `get_previously_observed_tracks(b)` | Internal-slot presence, mean `[x,z,vx,vz]`, covariance `[4,4]`, weighted age/stop/association diagnostics; slots are not physical IDs |
| `predict_observations(model,b,query,future)` | Approximate optical first/any/none events for externally compiled native poses; same belief fingerprint/tape |
| `condition_observation_event(model,future,events,'any'/'none')` | Coarse finite-law imagined posterior; no exact observed center synthesized |
| `successor_at(future,index,masses,robot_pose)` | Finite imagined carrier, not actual belief |
| `project_future_distribution(future,regions,times)` | Joint-world center counts/membership including hidden pool; no absence certificate |
| `belief.graph.materialize()/get()/local_features()` | Arrived static cells, namespace/version, resolution/height; unknown defaults state 0; no dedicated tensor exporter in this L2 release |

World X/Z meters; robot poses `[x,y,z,yaw]` in meters/radians; positive heading
basis `[cos(yaw),-sin(yaw)]`. All `future.times` are absolute seconds. Default
3 s forecast has 64 timestamps: root, every .048 s endpoint through 2.976 s,
then 3 s. Native query poses have `len(actions)+1` rows, .048 s per action;
root optical emission excluded. Exact native endpoint alignment is required.
Queries use `executor_version=known_space_discrete_route_follower_1`,
`compiled.planned_poses`, `compiled.actions`, and graph revision. Static swept
support uses existing `_support`; unsupported routes remain unsupported.

## Measurements and scientific limits

Original report (preserved, not remeasured here): representative warm/cold P95
87.446/88.275 ms; dense 116.866/115.887 ms; 484,615 sealed outcomes. The 4.2%
native-gap path ES decrease has difference −0.0247, 95% CI [−0.0905,+0.0479];
improvement is not statistically established.

New 11-root complete L2-only calls (observe + shared 3 s/16 worlds + longest-query
optics + none successor):

| Cache | P50 ms | P95 ms | P99 ms | Calls |
|---|---:|---:|---:|---:|
| warm | 99.810 | 117.358 | 167.109 | 33 |
| dynamic_cold | 106.659 | 131.940 | 161.261 | 33 |


This small dense-enriched verification cohort is not the historical 83-root
benchmark and does not pass 100 ms P95. Prefix reconstruction/imports/JIT excluded;
load time recorded separately. No GPU benchmark or fitting was needed.

Belief/reobservation, count support, recycled identities, duplicate ancestry,
speed/turning calibration and unlocalized broad spatial prior remain inherited
REVISE limitations. Optics is approximate and does not certify occlusion or
empty space; exact hypothetical sensor refinement is unsupported. There is no
L4 execution model or episode-success law. L2 native event hazards from older
models must not be mistaken for direct mesh contacts from this joint bank.
Official PSC is geodesic center distance with XYZ Euclidean fallback; L2 only
forecasts horizontal centers. Geometry approximations must be labeled separately.

Raw gate evidence: `output/l25_query_outcomes_v1_20261022/baseline_smoke.json`,
`baseline_native.json`, `baseline_regressions_retry/validation.json` and logs.
The frozen checkpoint hash was unchanged after execution.
