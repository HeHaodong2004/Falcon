"""Render bounded decision-revision evidence; no inference, fitting or data reads."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def percentiles(p):return [str(p['n'])]+[f'{p[k]:.2f}' for k in ('p50','p95','p99')]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inputs',required=True);parser.add_argument('--destination',required=True)
    a=parser.parse_args();src=Path(a.inputs);dest=Path(a.destination);dest.mkdir(parents=True,exist_ok=True)
    files={'primary':'benchmark_release_v1.json','batch':'benchmark_batches_v1.json','utility':'development_v1.json',
           'online':'development_v1_online.json','coarse':'coarse_oracle_v1.json','validation':'validation_release_v3.json','schedule':'scheduling_v1.json'}
    data={k:json.loads((src/v).read_text()) for k,v in files.items()}
    for k in ('primary','batch'):
        if data[k]['failure'] or data[k]['source_changed_during_measurement'] or not data[k]['all_same_law_comparisons_passed']:raise ValueError('unsealed benchmark')
    for k in ('utility','validation'):
        if data[k]['failure'] or not data[k]['sources_unchanged']:raise ValueError('unsealed validation/utility')
    primary,batch,utility,online,coarse,validation,schedule=[data[k] for k in files]
    stats=utility['evaluation']['summary'];unc=utility['uncertainty']
    perf_rows=[]
    for method,label in (('structural_bound','Optimized + packet'),('structural_bound_unoptimized','Reference + packet')):
        for mode in ('warm','fully_cold','graph_revalidation'):
            perf_rows.append([label,mode]+percentiles(primary['summary'][f'real_dev/{method}/24/{mode}']['timings_ms']['outer_wall']))
    perf_table=table(['Path','Caches','Calls','P50 ms','P95 ms','P99 ms'],perf_rows)
    comp=[]
    for mode in ('warm','fully_cold'):
        for key in ('observe','forecast','queries','branches','packet'):
            comp.append([key,mode]+percentiles(primary['summary'][f'real_dev/structural_bound/24/{mode}']['timings_ms'][key]))
    component_table=table(['Component','Caches','Calls','P50 ms','P95 ms','P99 ms'],comp)
    batch_rows=[]
    for q in (1,4,8,16,24):
        for mode in ('warm','fully_cold'):
            record=batch['summary'][f'real_dev/structural_bound/{q}/{mode}']['timings_ms']
            batch_rows.append([q,mode]+percentiles(record['outer_wall'])+[f"{record['queries']['p95']:.2f}"])
    batch_table=table(['Rows','Caches','Calls','P50 ms','P95 ms','P99 ms','Eval P95 ms'],batch_rows)
    dense=[]
    for n in (6,12):
        for cert in (False,True):
            for mode in ('warm','fully_cold'):
                dense.append([n,'certificate' if cert else 'positive',mode]+percentiles(primary['synthetic_summary'][f'structural_bound/{n}/{cert}/24/{mode}']['timings_ms']['outer_wall']))
    dense_table=table(['Tracked','Evidence','Caches','Calls','P50 ms','P95 ms','P99 ms'],dense)
    repeats=[]
    for r in primary['repeated_planner']:
        if r['method']=='structural_bound':repeats.append([r['root_id'][:12]]+percentiles(r['summary']['timings_ms']['outer_wall'])+[r['rss_change_kib']])
    repeat_table=table(['Root','Calls','P50 ms','P95 ms','P99 ms','RSS change KiB'],repeats)
    metric=[]
    for method in ('baseline','historical','l2'):
        pair=stats[method+'_pairwise'];regret=stats[method+'_regret'];picked=stats[method+'_selected']
        metric.append([method,f"{pair['root_mean']:.4f}",f"{regret['root_mean']:.4f}",
            f"{regret['scene_root_bootstrap_95'][0]:.4f}–{regret['scene_root_bootstrap_95'][1]:.4f}",
            picked['native_contact'],picked['native_psc'],f"{picked['progress']['mean']:.3f}",f"{picked['duration']['mean']:.3f}"])
    utility_table=table(['Rule','Pairwise','Regret','Regret 95% CI','Contact /60','Any PSC /60','Progress m','Duration s'],metric)
    groups=defaultdict(list)
    for row in utility['budget_sensitivity']:groups[(row['H'],row['W'])].append(row)
    budget_table=table(['H','W','Near-vacuous /12','Mean represented mass','Median evaluation+packet ms'],
        [[h,w,sum(r['collision_width']>=.95 for r in rows),f"{np.mean([r['forecast_represented_mass'] for r in rows]):.6g}",f"{np.median([r['scoring_ms'] for r in rows]):.2f}"] for (h,w),rows in sorted(groups.items())])
    # Descriptive attribution uses only already frozen legal predictions. It is
    # post-hoc engineering analysis, not a newly outcome-selected decision rule.
    attribution=[]
    for row in online['rows']:
        cs=row['candidates'];base=int(np.argmax([c['baseline_score'] for c in cs]));full=int(np.argmax([c['l2_score'] for c in cs]))
        without=int(np.argmax([c['l2_score']+5*c['collision_interval'][1]+c['social_interval'][1] for c in cs]))
        attribution.append({'root_id':row['root_id'],'vacuous':row['forecast_unresolved_mass']>=.95,
            'changed_baseline':base!=full,'changed_direct_risk_terms':without!=full,
            'full_score_difference':cs[full]['l2_score']-cs[without]['l2_score'],
            'contact_upper_span':float(np.ptp([c['collision_interval'][1] for c in cs])),
            'social_upper_span':float(np.ptp([c['social_interval'][1] for c in cs]))})
    changed_vacuous=sum(r['vacuous'] and r['changed_baseline'] for r in attribution)
    risk_changes=sum(r['changed_direct_risk_terms'] for r in attribution)
    meaningful_changes=sum(r['changed_direct_risk_terms'] and r['full_score_difference']>1e-8 for r in attribution)
    completed=[c for r in utility['evaluation']['roots'] for c in r['candidates'] if c['native_query_complete']]
    clock=np.asarray([c['nominal_duration']-c['native_elapsed'] for c in completed])
    analysis={'schema':'l2_decision_analysis_1','attribution':attribution,'risk_term_choice_changes':risk_changes,
        'risk_term_choice_changes_above_1e8_tolerance':meaningful_changes,'changed_baseline_near_vacuous_roots':changed_vacuous,
        'completed_nominal_minus_arrived_s':{'n':len(clock),'mean':float(clock.mean()),'p05':float(np.percentile(clock,5)),
            'p50':float(np.median(clock)),'p95':float(np.percentile(clock,95))},
        'input_hashes':{k:sha(src/v) for k,v in files.items()}}
    (dest/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False))
    def write(name,text):
        path=dest/name
        with path.open('x') as f:f.write(text.strip()+'\n')
    write('FINAL_REPORT.md',f'''# Layer 2 decision-readiness revision — 21 October 2026

**Research decision: DECISION INTERFACE REVISE.** The supported scientific/API
subset works, but uncertainty is frequently unusable and a coarse detection
event discards action-relevant information. Runtime independently needs
**SYSTEMS REVISE**. This is a working bounded release, not L2 DOWNSTREAM-READY.

## Continuity and implemented work

The clean research worktree started at `d66e6691`; branch
`research/l2-decision-ready-20261021` was created without resetting any release.
The pre-code information/evaluation contract was committed at `a443eb47` before
development outcomes were read. Runtime source is in `386ea59a` (same runtime
bytes at `5fa9532c`); hashes in every sealed artifact are authoritative. The
old structural reports, checkpoints and opened/protected splits remain intact.

Added `planning.py`: immutable packet/reference carriers, separate risk
intervals, stopping/time/route/signed-goal distributions, query geometry/yaw,
termination support, physical observation branches, status/cursor/versioning,
paired per-world responses, and an 18-column graph-compatible consumer. The
consumer joins arrived Stage 0 topology explicitly; L2 cspace support alone is
not that topology. Caller-selected branch rows replace the first-four-only
restriction while preserving the default API. The actual continuous belief is
retained separately; known imagined states are tagged and rejected by the
actual online adapter. Real static deltas now validate clock/state/cell/height.

The actual transition, learned dynamics/population/optical/native-event assets,
continuous coordinates, hidden population, shared futures and finite recursive
law are retained. Same-law changes memo-copy shared residual template arrays,
own readonly-view storage safely, and remove unused per-candidate tape copies.
No retained probability, endpoints, worlds or successors were dropped.

Added versioned one-native-step maneuver/OU training loss, finite-gradient and
inference-density tests, native competing-event likelihood bounds and explicit
point-training guards, plus a read-only versioned shard-metadata smoke interface.
This is not a completed 1M-outcome learning pipeline.

## Evidence-based gates

| Gate | Result | Interpretation |
|---|---|---|
| A Legal partial graph | REQUIRED AND PASSED | Audited Stage 0/1 input path; observed support only; future/static malformed deltas refused; current compiler schedules reproduced exactly. |
| B Actual belief reliability | REQUIRED AND PASSED within declared approximation | Exact centers, replay isolation, evidence/termination and persistence tests; no global exact-filter claim. |
| C World-model semantics | REQUIRED AND PASSED on supported finite/analytic law | Shared worlds, conditional coarse recursion and absorption retained; exact imaginary continuous bridge unsupported. |
| D Planning information | REQUIRED executable/API PASS; utility WEAK/INCONCLUSIVE | Packet and consumer execute; no established discrimination improvement or navigation claim. |
| E Usable uncertainty | REQUIRED AND FAILED | 25/60 roots and187/446 candidates have interval width>=.95. |
| F Full-path runtime | REQUIRED AND FAILED | Warm/cold P95 137.11/153.16 ms;100 and65 ms gates fail. |
| G Scaling handoff | REQUIRED bounded compatibility PASSED | Kernel/loss and metadata exercised; streaming/full event training migration still required. |
| H Regression/reproducibility | REQUIRED AND PASSED | {validation['total_tests']} cases,124 same-law comparisons, fixed assets/seeds/provenance and separate reviews. |

## Decision usefulness

Original development only:60 independent roots,51 scenes,446 unique legal
queries. Every query recompiles to the saved actions/poses and has complete
scoring support (399 native query completions,47 contact terminations). Utility
uses one6 s/16-world future per root, covering complete plans; the6 s statistical
head extrapolation is distinct from the matched3 s systems workload. Outcome
summaries were opened only after online scores were published. No hidden future
people or future graph payload was opened.

{utility_table}

The paired baseline-minus-L2 regret difference is .0685 with95% interval
[-.1485,.3063]; historical-minus-L2 is .0451[-.1675,.2750]. Neither establishes
improvement. L2 changes17 baseline selections;{changed_vacuous}/17 changes occur
on near-vacuous roots, whose contact upper endpoints cannot distinguish
candidates. Lower contact counts accompany lower progress/duration and cannot
be attributed entirely to forecasting quality. Contact AUC has only6 mixed
roots; social AUC has14. No SR/SPL, predictive SOTA or equal-sensor claim.

## Uncertainty and observation limits

Represented forecast mass ranges6.01e-164–.99999947. Mean contact/social width is
.4207. The same5/12 difficult roots remain near-vacuous for H6/12/24 and
W8/16/32. One actual-history trace goes from pruning8.57e-9 at.048 s to
unresolved .9855 at.096 s under the global residual density envelope, then
effectively1. This is a structural inference-usefulness blocker; more worlds
or a new motion fit alone is not the repair. No omitted mass was renormalized.

The finite observation oracle executes legal next-query continuations: any/no
detection changes the preferred route. Yet two exact center observations in
the same detection category disagree with its action preference with conditional
probability20%, losing1.2736 reference utility units (.6368 unconditional).
These are declared geometric proxy-law results, not native contact/PSC evidence.
A small spatial/time detection partition is justified for a next bounded
experiment; a universal continuous bridge is not justified by this result.

## Systems and disposition

{perf_table}

Prior frozen structural warm/cold P95 was115.52/135.19 ms without this packet.
Current complete calls are slower; the packet alone has warm P95 13.08 ms, and
the internal runtime P95 is124.42 ms. Current optimized/reference packet P95 is
137.11/159.24 ms, a same-law comparison, not an overall promotion. Cold and dense
failures remain visible. Exactly 124 comparisons pass, max absolute
numeric error1.465e-11. All3×100 repeated optimized loops preserve beliefs and
show sampled RSS change0. See PERFORMANCE for density and all batch sizes.

**SUPPORTED APPROXIMATION:** Gaussian/GH filtering and finite conditional paths;
local omitted-mass intervals exclude projection/quadrature/native-calibration
error. Exact continuous imagined refinement remains explicitly unsupported.

**DEFERRED TO DATA SCALING:** statistical motion/population/optical/event fitting
after compatible objectives and the residual problem are resolved. Data volume
does not resolve probability semantics or the runtime gate.

**OPTIONAL FUTURE RESEARCH:** decision-region observation partitions and external
compute allocation, with bounded oracles before any broad model work. Full
L2.5, L3 RL, social L4, map completion and large training remain outside scope.

Three independent team reviews found and verified concrete corrections; they
do not replace the negative gates. Production remained running with28 workers;
only status/source reads were performed in acceleration-v2. No service, schema,
database, worker count, collection policy or training job was changed.
''')
    write('SCIENTIFIC_VALIDATION.md',f'''# Scientific validation and precise applicability

The supported continuous actual-filter and finite imagined-successor interface
passes its implementation gates; **Gate E fails decision usefulness**. The old
universal continuous-successor-closure requirement is not reintroduced. Neither
local residual bookkeeping nor passing tests establishes a globally exact Bayes
filter, calibrated native safety, or arbitrary continuous hypothetical closure.

## Executed regression evidence

`validation_release_v3.json`: {validation['total_tests']} passing cases, unchanged
runtime source hashes, no fitting/truth payloads. The preserved322 comprise91
L2 tests,179 structural tests and52 CLI self-tests. Added39 cover planning/API,
ownership, deadlines, causal deltas and training compatibility. All original
analytic/finite probability and500-arrival persistence oracles were rerun.
Reviewer A independently reran68 overlapping cases, not68 extra independent
scientific samples. Both new API examples run with empty stderr.

124 optimized/reference comparisons pass (64 primary/stress,60 matched batch),
max absolute discrepancy1.465e-11. They include beliefs, sampled paths, event
mass, branches and successors. A new test also compares paired per-world
contact/social responses, which the archived comparator did not inspect.

| Property | Tested scope |
|---|---|
| Actual exact positions | Anonymous injective updates preserve retained exact XYZ centers and zero measurement position covariance; tracker ID changes do not supply physical identity. |
| Persistence/population | Continuous tracked/hidden state, count support, reappearance and500-arrival session; never-localized humans remain broad, not graph-constrained. |
| No evidence injection | Replay receipts separate sensor/execution/static updates; unavailable sensor is not empty; propagation adds no artificial observation. |
| Actual versus imagined | `actual_continuous`, `imagined_analytic`, `imagined_finite` ancestry; actual adapter rejects known imagined priors. Raw manually constructed states remain a caller trust boundary. |
| Temporal recursion | Direct3 s vs2×1.5/3×1 analytic composition at off-grid starts agrees to2.842e-14 in preserved oracle. |
| Coarse recursion | Retained finite path law direct/segmented probabilities and joint weights agree; prior reconstruction tested. No independent redraw at a coarse endpoint. |
| Counterfactuals | Candidate order preserves per-world responses on one shared human tape; no independent query redraw. Native event-noise cross-query coupling is not defined. |
| Terminal/nonexecution | Absorbing termination; horizon0/invalid route expose no new risk.120-action limit permits another high-level query, unlike unfinished horizon censoring. |
| Query provenance | Pose/action/waypoint/yaw/executor digest detects changed query metadata; STOP/unknown primitives and action/pose cardinality rejected. Full compiler provenance is a trusted Stage0/1 boundary. |
| Static causality | Future/stale delta clocks, UNKNOWN/state3, noninteger cells, nonfinite heights and missing sources refused without mutating prior. Unversioned synthetic delta clocks inherit arrived time explicitly. |
| Ownership | Graph cells/tapes immutable; residual snapshots own even readonly-view backing storage; shared buffers memo-copied once. Small result arrays owned, graph/world arrays borrowed. |
| Atomic deadlines | Earliest nested deadline and lock wait retained; early/late timeout publishes no packet/posterior, actual input unchanged. Nonpreemptive kernels can overrun. |

## Probability interpretation

For retained event contribution a and unresolved mass u, event interval is
[a,min(1,a+u)] under the retained surrogate. It does not bound projection,
quadrature or learned calibration error. A scalar risk upper endpoint is never
called a calibrated probability. Physical branch lower masses are not
renormalized. Their upper endpoints share one residual allocation simplex.
Unresolved epistemic mass has no physical observation or selectable successor.

Only contact and the one supported survivor cause receive residual termination
mass. Complete/censored/action-limit/invalid/absorbed causes remain distinct;
structurally impossible causes stay[0,0]. The first-event and stopping
distributions are lower submeasures with unresolved ranges, not complete point
laws. Terminal stopping time/pose remains represented-only when u>0.

Exact continuous observations may update the separately retained continuous
belief. Finite coarse successors refuse arbitrary exact detections/certificates
and unrecorded interior-time bridges. They continue their own joint paths/RNG
at supported endpoints. No Gaussian smoothing around particles was added.

Optional execution likelihoods reweight incoming joint hypotheses under a
component-constant likelihood contract. They are not automatically sampled-world
risks or an exact continuous path-dependent native survival posterior. Mere
terminal/time metadata does not imply measured survival conditioning.

## Real information boundary

Acceleration-v2 revision0fb8d983 was inspected read-only, including live
GeometryObserver, PartialGraph, query compiler, fixed follower, metrics and
storage. Sensor version is ideal_frustum_observed_cspace_dynamic_occlusion_1:
live articulated camera, static/current human occlusion, exact visible centers.
It is idealized structured sensing, not original pixel/depth equivalence.
It supplies no general human-visible-empty certificate. Native articulated
contact and geodesic-center PSC (XYZ fallback) differ from planar proxy features.
Safe/intrusion proposal provenance is not observed outcome safety.

Observed configuration-space support, UNKNOWN geometry, dynamic people,
predicted optical events and unsupported future static deltas remain separate.
The utility reader opens only legal history/query files online; after score
publication it opens frozen development outcome summaries for scoring. Neither
hidden future arrays nor saved future graph deltas enter online prediction.

## Negative evidence and review disposition

25/60 actual-history roots,187/446 candidates are nearly vacuous. H/W growth
does not cure the same five affected roots in the12-root sensitivity subset.
An observed8.57e-9 pruning remainder becomes.9855 in one positive update via
the residual supremum, not a new physical-population measurement. Projection
or dynamics calibration may also be deficient, but those unmeasured effects
are not the demonstrated source of this bound amplification.

The finite coarse sufficiency oracle reports20% conditional ranking loss
probability and1.2736 conditional utility loss. Its next routes compile on the
same observed graph; its exact events are supported finite center atoms. It
does not claim a continuous bridge or native metric validation. A refined
observation partition should be tested before a deeper planner relies on any/no.

Reviews: [scientific](reviews/SCIENTIFIC.md), [planning](reviews/PLANNING.md),
[systems](reviews/SYSTEMS.md). Earlier adversarial failures and the final fixes
are preserved. Supported B/C passes coexist with E/F failures.
''')
    write('DECISION_UTILITY.md',f'''# Decision utility: executable consumer, weak discrimination

The interface executes successfully; **decision improvement is inconclusive**.
The main actionable failure is usable uncertainty, not absence of a packet.

## Fixed protocol and support

`EXPERIMENT_PROTOCOL.md` was committed at a443eb47 before outcome access.
The original60-root/51-scene development timing cohort supplies446 distinct
queries (3–19/root). Every proposal recompiles against its actually arrived
Stage0 graph to identical native actions/poses. No rows are duplicated in this
utility experiment. All446 nominal plans fit the120-action cap (max109 actions).
One6 s/16-world/seed27 human future is shared per root; this covers complete
queries, rather than scoring a3 s prefix risk against a full-query label.
The frozen head's6 s extrapolation remains a statistical limitation.

The18 candidate features preserve geometry/yaw, duration, route and signed
Euclidean PointGoal progress, separate contact/social intervals, represented and
unresolved mass, completion/support and branch availability. Arrived sparse
topology and frontier/proposal provenance are joined separately by version;
UNKNOWN never becomes a new edge. This is a fixed consumer, not L2.5/L3 RL.

Utility = actual signed goal-distance reduction −5*native human mesh contact
−any official native PSC −.1*arrived elapsed seconds. Graph baseline uses
planned progress/time; history baseline adds declared planar CV proximity
proxies. L2 uses progress lower, contact/social upper and duration upper
endpoints. This separable pessimistic rule is conservative and can be looser
than a jointly optimized residual allocation; it is not calibrated expectation.

Online scores were saved to development_v1_online.json before any outcomes were
opened. Only hash-verified original development outcomes.json.gz summaries were
then read.399 query completions and47 contact terminations give complete
candidate-set support on all60 roots. No future observations, people truth or
graph-delta payload is read. The hindsight oracle is restricted to these scored
siblings. It is not an online navigation oracle.

## Results

{utility_table}

Pairwise summaries use58 roots/50 scenes with non-tied realized utilities.
Regret uses60/51.95% intervals resample scenes then roots,1000 draws,seed271;
siblings are not independent human futures. Baseline−L2 regret=.06849
[-.14847,.30635]; history−L2=.04510[-.16751,.27496]. Lower mean regret therefore
does not establish improvement. L2 lowers selected progress and duration too.
L2 pairwise ranking is below both baselines. Native PSC here is the official
geodesic-center/XYZ-fallback outcome, not the historical CV proxy.

There are47 native-contact and100 any-PSC outcomes across446 siblings. Mixed
contact support is6 roots/6 scenes: within-root AUC=.5556[.4438,.7500]. Mixed
PSC support is14 roots/13 scenes: AUC=.7310[.6000,.8641]. These are weak/small
development diagnostics, not calibrated-probability or SOTA results.

## Attribution and usable uncertainty

Forecast represented mass: mean.58071, median.99999937, minimum6.01e-164.
Unresolved mass: mean.41929. Contact/social interval width: mean.42069,
median6.34e-7,P95=1. Near-vacuity(width>=.95):25/60 roots,187/446 candidates.
Unsupported/time-out candidate counts are0/0 under this deliberately unbounded
diagnostic; this is not the frequency under a100 ms deadline. Pairwise score
ties(tolerance1e-8) occur1.735% of candidate pairs.

![Measured mass and actual-history amplification](figures/uncertainty.png)

L2 changes17 baseline and13 historical choices. Of the17 baseline changes,
{changed_vacuous} occur among the25 nearly vacuous roots. All25 have contact
upper-risk span<=1e-8;24/25 also have social span<=1e-8. Thus changed selections
mostly reflect conservative unresolved progress/time treatment, not useful
human-risk discrimination. Descriptive post-hoc legal-prediction attribution
removing only direct risk penalties (holding stopping distributions fixed)
changes{risk_changes} choices, of which{meaningful_changes} exceed1e-8 score
advantage. This is not a newly selected outcome-scored policy.

Actual-history traces identify envelope amplification: root7415a33faee8 has
u=0 at0 s, pruning u=8.57e-9 at.048 s, then u=.98553 at.096 s and effectively1
by.192 s. The residual count/density supremum is far larger than ordinary
represented exact-position evidence. No residual point law is fabricated.

{budget_table}

All H12/H24 choices match H6 at each matched(root,W) in this12-root subset;
five roots remain near-vacuous for every budget. More worlds improve retained
forecast support for the already useful roots but do not repair collapsed
filtering mass. These are diagnostics, not a selected larger default or a fit.

## Coarse observation sufficiency

The executable finite oracle declares two stationary possible human centers,
equal prior, persistent endpoint detection probabilities.2/.8 and geometric
conflict/.45 m + social/1 m proxy events. A completed one-action observation
query yields legal next-query decision points on unchanged known geometry.
Runtime any/no branches reverse the next route ranking, so coarse evidence
can matter. Both branches have probability.5.

Within survived_detection, exact center[1.5,0,.8] has conditional probability.2
and prefers the opposite route from the coarse action; the other center has
probability.8 and agrees. Conditional disagreement=.2; conditional decision
loss=1.27362; unconditional loss=.63681. Exact events are finite supported atoms,
not arbitrary continuous refinements of a Gaussian particle approximation.
These are not native articulated collision or official PSC measurements.

This falsifies universal sufficiency of any/no detection. A bounded next
experiment should partition detected centers by a query-relevant spatial
region (and possibly first-detection time), under the same endpoint optical
law and persistent joint tape, keeping residual allocation explicit. It does
not justify arbitrary pedestrian behavior labels, universal continuous
smoothing, or fabricated observations. No refinement was rushed into this
release merely to claim a positive result.

## Time and computation limitations

Primary planning-only cost here includes forecast/evaluation/packet plus the
optional Stage0 topology join; prefix reconstruction and recompilation are
recorded separately. Latency acceptance uses the separate3 s matched benchmark.
The all-outcome nominal-minus-arrived mean.09234 s includes collision stopping,
so it is not clock error. Among{len(clock)} actual query completions the mean
is{clock.mean():.5f} s (P05/P50/P95={np.percentile(clock,5):.4f}/{np.median(clock):.4f}/{np.percentile(clock,95):.4f}).
Native times remain authoritative. No navigation SR/SPL or final social L4
benefit is established. Verdict: DECISION INTERFACE REVISE.
''')
    write('PERFORMANCE.md',f'''# Complete online planning-packet performance

**Gate F FAIL:** warm P95 137.11 ms; cold153.16 ms. Neither100 ms integration
nor65 ms preferred target passes. Retain this result; no reduced-world or
omitted-successor pass is substituted.

## Matched workload and timing boundary

Same original60 roots/51 scenes,3 s,16 worlds,24 candidate rows,first four
refined rows,every native endpoint and successor. Repeated rows preserve the
prior workload where fewer than24 unique routes exist. Utility uses distinct
routes separately. Models receive the same audited preserved-repair incoming
legal prefix, as in the old benchmark; this operator benchmark is distinct
from the utility experiment's full structural actual-history initialization.

The timed path is public online_call plus packet_from_batch, including actual
update, shared forecast, candidate validation/support, all requested branches,
immutable result construction and graph/world references. The new actual
online_planning_call wrapper's lock/deadline behavior is separately tested.
Optional bind_stage0_context support scanning/topology copying is excluded from
this repeated-call benchmark; L2.5 can retain it per graph version. Its cost is
included in the utility timing. No graph topology is secretly regenerated.

Cold clears all dynamic support/footprint/template/Sobol caches in an initialized
model/JIT process. Checkpoint/JIT/prefix preparation and result verification
are separate and recorded. CPU only, one backend thread/job,nice19/idle I/O,
at most two research jobs. Empirical repetitions are correlated; these are
workload quantiles, not real-time guarantees or independent-scene confidence.

{perf_table}

Prior release P50/P95/P99 warm59.46/115.52/127.39,cold68.05/135.19/155.31 ms.
Current packet path is slower, including new provenance/typed-output work.
Current internal online path before packet has warm P95 124.42 ms; packet-only
P95 is13.08 ms. Optimized versus same-law reference full warm P95 is137.11
versus159.24 ms. A kernel speedup does not turn the complete-call gate into PASS.

![Measured complete-call latency CDF](figures/latency.png)

## A/B/D/E component evidence

{component_table}

Branches are nested inside queries; quantiles do not add. Observation P95
remains53.93 ms. Joint association, negative-evidence quadrature, validation,
posterior/template ownership, full belief/query fingerprints and serialization
remain material. Packet validation/metadata is not free.

## C matched candidate scaling

The preserved12-root difficult cohort is identical at each Q. Eval P95 is the
query evaluation component, excluding actual update/forecast/packet; complete
columns include all those operations.

{batch_table}

No independently regenerated human futures per candidate. The cohort differs
from the60-root primary aggregate and should not be compared as if identical.

## F dense tracked/hidden cases

Preserved3,721-cell fixtures,6/12 tracked people,hidden count support0/2/4,
24 queries,16 worlds,3 s,four refined. Warm20/case;cold2/case is descriptive.
Certificates are explicit synthetic stress evidence, not current-sensor defaults.

{dense_table}

All posterior support and worlds retained; residual mass remains explicit.
Dense positive6 improves relative to old measured142.92 ms P95;12 positive
improves versus192.93. Certificate6/12 now117.04/142.63 versus old106.52/133.43;
packet/provenance overhead and shared-host variation remain visible. These are
separate historical observations, not a causal micro-optimization ablation.

## G repeated calls and memory

{repeat_table}

Each loop varies seed while keeping arrived evidence/time fixed; all beliefs
unchanged. This is not300 physical observation steps. Primary final RSS
{primary['metadata_after']['rss_kib']/1024:.1f} MiB; peak
{primary['metadata_after']['rss_max_kib']/1024:.1f} MiB includes models/JIT and
the allocator, not a per-belief requirement or global memory proof. Existing
bounded cache limits are preserved (256 support/footprint entries,64 endpoint
templates/8 MiB,16 Sobol entries/8 MiB). Small packet arrays are owned; graph and
human tape arrays are referenced without full copies.

## Planning-oriented scheduling and attribution

On the same12 roots,60 calls each: Q24 base-only evaluation+packet P50/P95/P99
42.27/64.85/77.49 ms; four externally selected rows77.06/101.41/152.82 ms.
Both passes reuse the exact same forecast and give identical per-world risks.
The selected pass currently recomputes base hazards; this is reusable forecast
support, not an optimized incremental branch-only operation or a compute policy.
Forecast timing is separate in scheduling_v1.json. Base-only cost is not used
to pass the full gate.

Current dense cProfile retains all4 fixtures. For12-person certificate:
total instrumented320 ms, observe120 ms,queries127 ms,packet33 ms; nested
fingerprints46 ms,certificate conditioning58 ms,residual snapshot23 ms.
Old comparable profile attributed40 ms to eager residual freezing. The new
memoized snapshot helps allocation, while the complete budget still fails.
cProfile times are attribution only, not latency quantiles. Full query identity
checks and packet metadata remain a concrete next systems target; any reuse
must preserve mutation detection and immutable ownership.

124 same-law comparisons pass,max absolute discrepancy1.465e-11. Atomic early,
late,nested and lock-wait deadlines are tested. Cooperative kernel overrun is
still possible; no hard preemption claim. Collector primary snapshots remain
RUNNING,28 workers,{primary['collector_before']['status']['outcomes']}→
{primary['collector_after']['status']['outcomes']} outcomes, recent rates
{primary['collector_before']['status']['validated_outcomes_per_second_recent']:.2f}→
{primary['collector_after']['status']['validated_outcomes_per_second_recent']:.2f}/s.
This is observational coexistence, not a causal zero-impact claim.
''')
    write('SCALING_HANDOFF.md','''# Bounded training and inference handoff

No100K/300K/1M training was launched. The online architecture remains bounded
by local graph/population/hypotheses/worlds/queries/branches, not dataset size.
The following compatibility pieces execute; a streaming production trainer
and statistically validated new checkpoints are not delivered.

## Frozen and learnable quantities

Freeze legal partial-map/sensor boundary, anonymous association measure,
actual/imagined distinction, absolute clock, executor identity, residual policy,
first-event/censoring labels, sibling grouping and packet schema. Checkpoints
and historical sources are unchanged. Dynamics, population/spatial priors,
optical parameters and native heads may later be learned within those laws.
No fit may secretly supply full-map geometry, physical IDs or future inputs.

Current human exogeneity is versioned for the audited Falcon campaign. It is
not assumed for an arbitrary social L4 or another human controller. A new L4
needs duration/route/observation/event validation even if human motion transfers.

## Motion objective actually implemented

Version native-jump-ou-one-step-position-nll-1 in training_contract.py:
at an absolute.048 s grid start, a Bernoulli jump with p=1−exp(−rate*.048)
acts on velocity (retention1 by default,independent jump std2), then integrated
OU relaxes toward a persistent latent target with diffusion. Positive relaxation
also adds.0025*dt independent position variance, matching inference. The loss
analytically marginalizes the jump/no-jump Gaussian position density and one
latent intention mixture for that teacher-forced interval.

Tests compare density against inference transition_components, rate0 OU parity,
finite one-batch gradients and root balancing. Off-grid starts are refused;
they require retained interval jump state, not a redraw. This is a statistically
defined one-step conditional position objective, not a complete multi-step path
likelihood. It does not rederive the initial velocity/ancestry supervision.

Legacy train.distribution/objective uses integrated OU displacement without
these jumps. It is incompatible with claiming optimized compound inference
likelihood. A future longer-horizon objective must preserve one intention mode
across targets and native-grid jump indicators across API partitions, with its
moment projection/integration approximation declared. The compact network may
be reused; its old objective is not relabeled by loading a larger dataset.

## Native event targets and scoring

Version native-competing-first-event-residual-bounds-1 exercises finite
likelihood intervals for normalized competing native termination hazards:
joint survival through observed exposure, first cause at its endpoint, no
post-contact exposure, full planned horizon retained, explicit right censoring.
Complete binaries require full noncontact followup; an observed positive can
be scored at first event. Administrative incomplete negatives remain censored.
PSC is a separate nonterminal first-event channel (terminal endpoint included),
not a competing cause that terminates execution. Native PSC means geodesic
center distance with specified XYZ fallback, not planar proxy features.

For represented likelihood L and residual u, return[L,min(1,L+u)] and score
interval[−log(L+u),−log(L)] (appropriate endpoint clipping). These are diagnostic
bounds, not proper point objectives; minimizing either bound does not certify
calibration. Conservative contact/PSC upper endpoints must never become point
probability targets. A point fit requires a declared fully specified normalized
train-time latent law, or a separately justified complete-support event model.
Root-balanced marginal composite scores are permissible if named correctly;
a joint sibling likelihood would instead integrate the shared root future once.

require_point_training_law rejects bound/legacy residual semantics in the new
adapter. Archived repair_events fitting/calibration still uses its old.005
residual hazard and is retained for historical reproduction. It is NOT an
allowed structural trainer, and the new guard is not a wrapper around every
archived CLI. Future event fitting, PSC exposure handling, calibration caches
and scoring readers must migrate before100K training. This work does not claim
those migrations complete.

## Data interface actually exercised

Stage1 outcomes are sealed per-root shards under schema
observation_limited_query_conditioned_root_1, with versions, file checksums,
query IDs and one independent human future per root. Frozen snapshot rows use
versions; direct root manifests use version. The read-only metadata iterator
accepts both conventions and refuses explicit unsealed/multiple-future records.
Real frozen snapshot metadata was smoke-tested; no full corpus preprocessing.
No fresh production snapshot or database migration was performed.

Later readers must stream these shards, retain offline ancestry/support and
visibility/censoring metadata and group scenes→roots→siblings. Current data.build
and train.run still materialize whole object arrays and use legacy visible-MAP
training ancestry. Streaming/full-population supervision remains required.
One million query outcomes are not one million independent human futures.

## Required migration before later campaigns

1. Repair or retain structure in the residual evidence envelope; H/W growth
   alone did not repair observed mass collapse. This is not deferred to learning.
2. Implement and validate the chosen persistent multi-step training law or
   explicitly train the tested one-step conditional contract with legal inputs.
3. Migrate native first-event/PSC/censoring supervision and point-vs-interval
   objectives; fit only on train, tune only declared development diagnostics.
4. Stream sealed versioned shards with fixed scene/root grouping and preserved
   protected exclusions; verify source law/executor/checkpoint compatibility.
5. Freeze metrics/rules before a genuinely new scene-disjoint confirmation pool
   is opened once. The historical holdout and15-root fresh cohort are already
   opened and cannot become new confirmation under a renamed report.
6. Re-measure complete packet latency and decision usefulness. Do not advance
   from100K to300K/1M merely because more samples are available.

No larger architecture is prescribed. Gaussian/GH projection remains a visible
approximation; finite hypothetical exact refinement stays unsupported. A small
decision-region observation oracle is appropriate future work; universal map
completion/POMDP closure is not a prerequisite for the supported bounded API.
''')
    write('API.md','''# Layer 2 planning API

Run from this research worktree with the existing eventwm-falcon Python env:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
 nice -n19 ionice -c3 /research-user/miniforge3/envs/eventwm-falcon/bin/python \\
 scripts/l2_decision_api_example.py
# Repeat with --frozen-checkpoints for the unchanged trained assets.
```

Both modes initialize/observe a continuous actual belief, evaluate multiple
queries with one future, build the packet/18-column consumer, continue a finite
coarse branch using a newly compiled known-space route, reject unsupported exact
imagined refinement, update the retained actual belief, round-trip JSON and
demonstrate atomic timeout. Fixtures are synthetic, budgets generous; these
examples are correctness evidence, not native or latency experiments.

```python
from scripts.l2_partial_structural_api_example import (
    build_model, legal_fixture, observation, queries,
)
from eventwm_graph.l2_partial.planning import (
    packet_from_batch, graph_decision_features, pessimistic_scores,
    online_planning_call,
)

model, _ = build_model()
history, observed = legal_fixture()
actual = model.initialize(history, observed)
actual = model.observe(actual, observation(.048, 2.01))
plans = queries(actual.graph.version)

shared = model.predict_shared(actual, .288, num_worlds=16, seed=7)
batch = model.evaluate_queries(
    actual, plans, shared, budget_ms=5000., refined_query_indices=(1,))
packet = packet_from_batch(actual, plans, batch, pointgoal=[3., 0., 0.])
features = graph_decision_features(packet)  # candidate rows + graph reference
scores = pessimistic_scores(packet)        # declared interval rule
assert packet.shared_future.positions is shared.positions
assert packet.graph_reference is actual.graph
```

The caller may evaluate with refined_queries=0 first, then request up to four
arbitrary row indices against the SAME shared future. Default refined_queries=4
still refines the first four; existing callers remain compatible. Deferred
branch work means NOT_REQUESTED, not zero probability/no detection. A second
evaluation currently recomputes candidate hazards; there is no hidden compute
allocation policy. All queries must start at the belief/successor pose and use
the Stage1 compiler artifact. Primitive labels/cardinality and full query
identity are checked, while full kinematic/geometry compilation is upstream.

```python
# Atomic actual action/execution/arrived-observation cycle. Real integrations
# supply actually arrived rows, not synthetic predictions from packet branches.
result = online_planning_call(
    model, actual, observation(.096, 2.02), plans,
    pointgoal=[3., 0., 0.], horizon=.288, num_worlds=16, seed=7,
    branch_indices=(0,), budget_ms=5000.)
if result.status == 'COMPLETE':
    actual = result.actual_belief
    packet = result.packet
else:
    assert result.packet is None and result.pending_observation
    # Keep the old actual belief and pending arrived row for explicit retry.
```

Optional actual_execution_trace uses the existing component-indexed likelihood
contract, support_fingerprint and evidence_id. It does not automatically convert
sampled-world risks/native path contact into an exact Gaussian survival update.
The adapter holds the model lock over update/forecast/branches/packet, preserves
the earliest outer deadline, and publishes atomically. Kernels are cooperative,
not hard-preempted. Known imagined finite/analytic states cannot be actual priors.

Supported successors are packet.candidates[i].observation_branches[j].successor.
Use handle.resolve() as a borrowed read-only state. At a completed nonterminal
query or executor limit another legal query can be selected; a horizon-censored
prefix supports reference continuation, without an invented Stage1 interruption
contract. At terminal contact neither operation is allowed. The unresolved
bucket is epistemic and has no physical successor handle.

For a NEW query compile on retained observed geometry at the successor pose
(see executable example); continuing an unfinished reference preserves its
cursor. predict_shared/continue_rollout preserve the finite carrier's world
count and stored path/RNG law. Exact continuous detections, certificates and
unrecorded intermediate times can raise UnsupportedConditioning. After real
execution, update the separately retained actual continuous prior instead.

compile_observed_queries(snapshot, proposals, robot_pose) uses the existing
Stage1 compiler. packet_from_batch(..., stage0_snapshot=snapshot) additionally
binds arrived topology after verifying timestamp and support equality. This
optional scan/copy need not repeat when L2.5 retains a context per graph version.
The packet's L2 cspace graph is not the complete Stage0 topological structure.

Packet ownership: graph cells and shared tapes are immutable references; small
route/distribution/metadata buffers are owned and readonly. Legacy successor
belief objects are borrowed under a no-mutation contract, not deep-frozen by
the packet. Caller mutation/concurrent edits of input beliefs/queries are not
supported. Durable actual/carrier state uses existing schema2 serialization;
packets are in-process references, not a new durable checkpoint schema.

Distribution.lower_weights sum to represented mass, with separate unresolved
range. mean_interval combines those without normalizing. Goal progress is
signed Euclidean PointGoal distance reduction, route progress is arclength,
duration is the evaluated physical stopping/completion interval. These are not
navigation SR/SPL, arrival success, calibrated event point probabilities or
predicted future static revelation. Complete planned yaw/poses and evaluated
survival endpoint/yaw remain separate. See L2_TO_L25_CONTRACT for all fields.
''')
    write('REPRODUCE.md',f'''# Reproduce the decision-readiness revision

Start from the separate research branch; no production merge/deployment or
training. Baseline d66e6691,prior frozen runtime ec2ad5b1da4437b2a0d9d747d7659323af07450a,
pre-outcome protocol a443eb47,runtime386ea59a (same bytes5fa9532c). Later commits
add tests/reports/artifacts. Source SHA-256 mappings in sealed JSON are exact;
report/review commits need not share the runtime freeze commit.

Python 3.9.19, NumPy 1.23.5, SciPy 1.13.1, PyTorch 1.12.1, Numba 0.59.1,
matplotlib 3.3.2; CPU Linux x86_64 (kernel 6.8.0-138-generic).
Use the existing env; do not change production packages. One numerical thread
per job, nice19/idle I/O, maximum two research jobs. Collector status reads only.

```bash
cd /research-user/workspace/projects/eventwm-falcon-l2-partial-20261021
L2_PY=/research-user/miniforge3/envs/eventwm-falcon/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
L2_RUN=output/l2_decision_ready_20261021/reproduce_001
mkdir "$L2_RUN"
nice -n19 ionice -c3 "$L2_PY" scripts/l2_decision_validate.py --output "$L2_RUN/validation_release_v3.json"
nice -n19 ionice -c3 "$L2_PY" scripts/l2_decision_api_example.py
nice -n19 ionice -c3 "$L2_PY" scripts/l2_decision_api_example.py --frozen-checkpoints
nice -n19 ionice -c3 "$L2_PY" scripts/l2_decision_coarse_oracle.py --output "$L2_RUN/coarse_oracle_v1.json"
nice -n19 ionice -c3 "$L2_PY" scripts/l2_decision_utility.py --output "$L2_RUN/development_v1.json"
nice -n19 ionice -c3 "$L2_PY" scripts/l2_decision_benchmark.py --roots60
```

For the actual full benchmark, use the expanded command (the abbreviated last
line above must use `--roots 60`, with a space):

```bash
nice -n19 ionice -c3 "$L2_PY" scripts/l2_decision_benchmark.py \\
 --roots 60 --batches 24 --warm-repeats 10 --cold-repeats 2 \\
 --synthetic-warm-repeats 20 --rl-repeats 100 \\
 --methods structural_bound structural_bound_unoptimized \\
 --output "$L2_RUN/benchmark_release_v1.json"
nice -n19 ionice -c3 "$L2_PY" scripts/l2_decision_benchmark.py \\
 --roots 12 --cohort docs/l2/partial_graph_wm_structural_20261021/artifacts/raw/batch_cohort.json \\
 --batches 1 4 8 16 24 --warm-repeats 5 --cold-repeats 2 \\
 --rl-repeats 0 --skip-synthetic --methods structural_bound structural_bound_unoptimized \\
 --output "$L2_RUN/benchmark_batches_v1.json"
nice -n19 ionice -c3 "$L2_PY" scripts/l2_decision_scheduling.py --output "$L2_RUN/scheduling_v1.json"
nice -n19 ionice -c3 "$L2_PY" scripts/l2_decision_report.py \\
 --inputs "$L2_RUN" --destination "$L2_RUN/rendered"
```

Every output path must be new; do not overwrite preserved runs. The report
renderer consumes sealed results only and writes a NEW destination. The final
release also appends implementation details to the pre-code contract and seals
artifacts via l2_decision_seal.py. Replaying bundled artifacts recreates reports,
not new scientific observations. Source-changed/failure flags must be false/null.

Fixed checkpoint hashes/paths are inherited unchanged from the prior structural
bundle at docs/l2/partial_graph_wm_structural_20261021/artifacts/checkpoints:

| Asset | SHA-256 |
|---|---|
| graph_dynamics.pt | 57c91156101320e2be6a02aee23d214a0405959e823b73be3786b0287bd25c05 |
| population_prior.json | b7df625ac62e0eae09656fa32532b72e28039af140fc7b2803e7866b8f1f539c |
| observation_fit.json | ace29a5160b4a8510ac96d3c7bfb3cbad0936fc809c2dd35bf302aaa874e8793 |
| native_events_v5_scene.json | feec91f6918963f618c8a44694fb210bcd92549c9a92e52fc394f2c0b6aa736b |

Snapshot manifest SHA-256: 92b80427e258db7f1ddef94a70e0d932dbe6d2ba834809ac322715f16599210c.
All checks are enforced by existing loaders/audits. Restore missing assets from
the prior committed structural bundle at the original relative paths, checking
hashes; no retraining or new live snapshot is required.

Real-scene reproduction needs the frozen legal/query files for selected original
development roots. Utility scoring additionally needs their frozen outcome
summaries; no privileged_future.npz/future observation or graph files are read.
Without licensed local snapshot payloads, synthetic API/oracles/tests still run.
The utility code publishes legal scores before opening outcomes, verifies each
checksum and uses fixed seed27; bootstrap seed271. Every sibling remains grouped
by root and scene. The15-root opened cohort and protected confirmation are not
new evaluation inputs. No full-corpus preprocessing or large training occurs.

Final validation: {validation['total_tests']} tests,124 same-law comparisons.
Primary1872 complete calls across two implementations plus600 repeated loops;
batch1080 calls. Dense profiles and scheduling add separate diagnostic calls.
Earlier smoke/validation artifacts, initial profiles, failed path-guard attempt
and reviewer counterexamples remain archived with their own hashes, not relabeled
as final evidence. The old structural runner rejects a decision-output path;
l2_decision_validate.py intentionally uses its functions with the new output root.

`artifacts/evidence_manifest.json` maps outputs to deterministic gzip/binary
copies and source checksums. `artifacts/release_files.sha256` covers current
source/tests/scripts, reports/reviews/figures and artifacts, excluding itself.
Verify from repository root with sha256sum -c. No collector/database write,
service restart, worker change or merge command is part of reproduction.
'''.replace('nice -n19 ionice -c3 "$L2_PY" scripts/l2_decision_benchmark.py --roots60\n','').replace('For the actual full benchmark, use the expanded command (the abbreviated last\nline above must use `--roots 60`, with a space):','For the full benchmarks, use:'))
    # Standard standalone scientific plots, no synthetic data added to figures.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs=dest/'figures';figs.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    masses=sorted(r['forecast_represented_mass'] for r in online['rows'])
    axs[0].semilogy(np.arange(1,len(masses)+1),masses,'o',ms=3);axs[0].set(xlabel='Development roots, sorted',ylabel='Represented forecast mass',ylim=(1e-170,2))
    axs[0].set_yticks([1.,1e-40,1e-80,1e-120,1e-160])
    witness=next(r for r in online['rows'] if r['root_id'].startswith('7415a33faee8'))
    axs[1].plot([r['timestamp'] for r in witness['history_mass_trace']],[r['unresolved_mass'] for r in witness['history_mass_trace']],'o-')
    axs[1].set(xlabel='Actual arrived time (s)',ylabel='Unresolved mass',ylim=(-.03,1.03));fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(figs/('uncertainty.'+ext),dpi=180,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else None)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.6))
    for mode,label in (('warm','Warm packet'),('fully_cold','Cold packet')):
        values=sorted(r['timings_ms']['outer_wall'] for r in primary['records'] if r['kind']=='real_dev' and r['method']=='structural_bound' and r['cache_mode']==mode)
        ax.plot(values,np.arange(1,len(values)+1)/len(values),label=label)
    ax.axvline(100,color='black',ls='--',label='100 ms gate');ax.set(xlabel='Complete call (ms)',ylabel='Empirical CDF',ylim=(0,1.01));ax.legend();fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(figs/('latency.'+ext),dpi=180,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else None)
    plt.close(fig)
    (figs/'README.md').write_text('Standalone measured figures; input hashes in ../analysis.json. uncertainty: original development actual-history filtering; latency: matched3s packet workload. No new calibration or confirmation.\n')
    print(json.dumps({'destination':str(dest),'validation_cases':validation['total_tests'],'parity':len(primary['same_law_parity'])+len(batch['same_law_parity'])}),flush=True)


if __name__=='__main__':main()
