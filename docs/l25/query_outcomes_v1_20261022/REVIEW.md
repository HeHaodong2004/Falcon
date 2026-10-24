# Review and validation record

This is a primary-engineer code/semantics review backed by numerical oracles
and integration tests. It is **not an independent external scientific review**.
No RL policy or L4 execution result is inferred from it.

## Findings addressed

1. The preserved regression driver requires an absolute output path. The first
   invocation failed in log path accounting; the corrected driver ran all 399
   tests and original oracles successfully without changing inherited code.
2. Frozen L2 exposes coarse optical `any`/`none` conditioning, not exact future
   sensor assimilation. The implementation uses that operation directly and
   marks successors imagined, approximate, and unsupported as Bellman targets.
3. Official collision/PSC require mesh contacts/geodesic distance. Forecast
   centers cannot establish them. The compiler reuses audited enclosing-body
   geometry and labels the separate horizontal social proxy; official fields
   and L4/episode success have false support masks.
4. The native compiler already supplies a route and position/yaw tolerances.
   The adapter now retains them instead of reporting them absent. It rejects
   inconsistent nominal durations and unapproved native compiler versions.
5. The legal root's `pointgoal` is absolute XYZ (`partial.collect` records
   `episode.goals[0].position`). The driver uses that public value. An early
   exploratory timing run used an explicitly labeled ego-goal interface fixture;
   it is not a controlled performance comparator.
6. Exact height equality fragmented static regions because native support
   heights contain small differences. Tiles now tolerate ≤2 cm seed-height
   deviation while retaining actual cell membership and exact native query
   support checks. Region extents are not promoted to traversable polygons.
7. Python static-edge construction and repeated route lookup were expensive.
   Static edges are exported with numeric array operations; route occurrences
   and observed topology are cached by content/graph/goal versions.
8. Identical finite-world observation likelihoods offer no decision information.
   Zero-MI events remain unexpanded with explicit residual mass. No hidden-world
   clustering or centroid measurement creates fictitious observation outcomes.
9. Mutable nested metadata could corrupt cached support statements. Context,
   reference payloads and conditional summaries now use immutable nested storage;
   a regression rejects mutation through returned handles. The release benchmark
   was repeated after this change. Both timing runs are retained; the slower
   release result governs the acceptance decision.
10. Native truth tapes contain occasional 56 ms renewal intervals. Exact native
    labels remain censored when they do not match the 48 ms nominal clock. A
    separate auxiliary target uses the existing offline linear truth interpolator
    without extrapolation. Neither is called native mesh contact or L4 execution.
11. The sensitivity driver's original fallback assertion assumed every scene had
    an informative event. A zero-MI scene correctly expands none. The assertion
    now checks whether an informative event exists; core risk equality is still
    mandatory. The initial failed attempt remains in the isolated output log.

## Verification scope

47 L2.5 tests cover brute-force finite-bank events, audited geometry equality,
multi-human/time union, weighted/no-human cases, exact public optical API
equality, legal context ownership, actual versus imagined separation, absolute
time, horizon censoring, reference integrity, graph connectivity, branch mass
conservation/residuals, observational indistinguishability, cache invalidation,
mutable metadata rejection, graph batching and candidate permutation.

The native gate, complete benchmark, scientific audit and demo exercise trained
weights on actual frozen legal development roots. Tests using generated pose
fixtures are mathematical invariants, not certificates of a new native compiler
or real L4 execution. No production/main merge, training, checkpoint update,
collector action, data rewrite or L4 worktree modification occurred.

Remaining limitations are substantive: frozen L2 belief/count/association errors,
broad hidden spatial prior, forecast calibration, approximate optics, proxy
geometry/distance, unsupported exact reobservation/occlusion, native-clock
integration, non-preemptible core work, and failed incremental/dense latency
targets. Passing regression tests does not remove those limitations.
