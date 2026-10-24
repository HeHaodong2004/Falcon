# Final L2.5 research report

**Decision: restricted experimental integration ready; overall REVISE for
unrestricted planning/deployment.** The compiler is runnable and reviewed, but
the incremental/cold/dense latency gates and inherited scientific limitations
remain open. There is no RL improvement or L4 success-probability claim.

Branch `research/l25-query-outcomes-20261022`, based exactly on
`cf1971228c37517202cc9e5787f785a3876a0514`.
Implementation commit `985b84be429f9ef175f4aa9607bf3426941558a2`. The subsequent report/artifact commit
is the branch HEAD (`git rev-parse HEAD`), reported in the delivery message.
Checkpoint SHA256 `215864a013c58c36a670b0aef3c4512a6eb651859f5ad24bb1ee899844189223`;
model `l2-intermittent-1-fc2c500cda1b243c7526`. Frozen L2 source/weights were not changed.

| Acceptance category | Decision | Scope |
|---|---|---|
| BASELINE_RUNNABLE | PASS | Trained checkpoint, native history/query calls and 399 unchanged regressions/oracles |
| API_INTEGRATION | PASS | initialize/observe/predict_shared/track readout/graph/optics/finite successors |
| GRAPH_CORRECTNESS | PASS | Arrived connected regions, typed relations, repeated route occurrences, explicit unknown |
| QUERY_OUTCOME_CORRECTNESS | PASS | Declared finite-bank endpoint envelope and horizontal proxy law; official contact/PSC unsupported |
| CONDITIONAL_SUCCESSOR | REVISE | Coarse finite any/none works; exact legal sensor/reobservation/occlusion unsupported |
| CAUSAL_VALIDITY | PASS | Legal input boundary, exact absolute nominal alignment, actual/imagined separation and masks |
| GNN_COMPATIBILITY | PASS | Typed tensors, root candidate IDs/masks, disjoint batching and permutation equivariance |
| INCREMENTAL_LATENCY | REVISE | Representative release P95 22.994 ms warm / 45.082 ms cold vs 10 ms |
| FULL_CALL_LATENCY | REVISE | Representative release P95 87.212/119.241 ms; dense 124.883/155.246 ms vs 100 ms |

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
