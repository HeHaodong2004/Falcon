"""Render the final scientific reports from preserved measured artifacts."""
import collections
import gzip
import json
import math
import re
from pathlib import Path
import numpy as np
from .common import OUT, OLD, DOC, BASE, AGE_BINS, dump, sha


def readj(path):return json.loads(Path(path).read_text())


def ci(value):
    if not value or value.get('mean') is None:return 'unsupported (0 roots)'
    if value['scenes']==1:
        return '%.4f [bootstrap degenerate: 1 scene] (%d/1)'%(value['mean'],value['roots'])
    return '%.4f [%.4f, %.4f] (%d/%d)'%(value['mean'],value['lo'],value['hi'],value['roots'],value['scenes'])


def table(head,rows):
    return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)


def readable_prose(text):
    """Space rendered numeric prose while preserving code, paths and hashes."""
    lines=[];fenced=False
    for line in text.splitlines():
        if line.startswith('```'):
            fenced=not fenced;lines.append(line);continue
        if fenced:
            lines.append(line);continue
        parts=re.split(r'(`[^`]*`|(?<=\])\([^)]*\))',line)
        for i in range(0,len(parts),2):
            value=re.sub(r'(?<=[a-z])(?<!\de)(?=[+-]?(?:\d|\.\d))',' ',parts[i])
            value=re.sub(r'(?<=\d)(?=(?!e[+-]?\d)[a-z])',' ',value)
            value=re.sub(r'(?<=[;:])(?=\d|[+-]?\.\d)',' ',value)
            value=re.sub(r'(?<=%)(?=[A-Za-z])',' ',value)
            value=re.sub(r'\bReviewer(?=[ABC]\b)','Reviewer ',value)
            parts[i]=value
        for i in range(1,len(parts),2):
            if not parts[i].startswith('`'):continue
            if parts[i-1] and parts[i-1][-1].isalnum():parts[i-1]+=' '
            if i+1<len(parts) and parts[i+1] and parts[i+1][0].isalnum():parts[i+1]=' '+parts[i+1]
        lines.append(''.join(parts))
    return '\n'.join(lines)


def write(name,text):
    (DOC/name).write_text(readable_prose(text).strip()+'\n')


def run():
    train=readj(OUT/'training/training.json');learning=readj(OUT/'training/learning.json');summary=readj(OUT/'evaluation/summary.json')
    gaps=readj(OUT/'data_and_gaps_strict.json');context=readj(OUT/'gap_context.json')
    physics=readj(OUT/'physical.json');oldphysics=readj(OUT/'old_physical_reproduction.json')
    optics=readj(OUT/'optical_validation.json');speed=readj(OUT/'final_speed_manifest.json')
    systems=readj(OUT/speed['benchmark']/'benchmark.json')
    startup=readj(OUT/(speed['benchmark']+'_source_capture')/'first_use.json')
    def timing_group(category,cache):
        return systems[category][repr((cache,16,'primary'))]['metrics']['combined_ms']
    warm=timing_group('representative_first60_quantiles','warm')
    cold=timing_group('representative_first60_quantiles','dynamic_cold')
    dense_warm=timing_group('dense_quantiles','warm');dense_cold=timing_group('dense_quantiles','dynamic_cold')
    systems_pass=max(warm['p95_ms'],cold['p95_ms'])<100
    preferred_pass=max(warm['p95_ms'],cold['p95_ms'])<65
    gate_text='passes' if systems_pass else 'fails'
    preferred_text='passes' if preferred_pass else 'fails'
    b=readj(DOC/'reviews/reviewer_b_final_score_checks.json')
    event_check=readj(DOC/'reviews/reviewer_b_final_event_key_checks.json')
    if not event_check['reviewer_source_stable'] or event_check['original_score_check_sha256']!=sha(DOC/'reviews/reviewer_b_final_score_checks.json'):
        raise ValueError('event normalization/unique belief support supplement is not bound')
    if not event_check['event_key_normalization']['normalized_all_reappearance_events_match']:
        raise ValueError('canonical reappearance event keys differ')
    belief_union=next(r for r in event_check['beliefs'] if
        (r['model'],r['scope'])==('D_StopMove','all_retained_actual_states'))
    final_validation=readj(OUT/'final_validation.json')
    validation=readj(OUT/final_validation['directory']/'validation.json')
    selected=min((r for r in learning if r['epoch']),key=lambda r:r['dev_es'])
    def contrast(base,cohort,scope):return next(r for r in summary['contrasts'] if (r['baseline'],r['cohort'],r['scope'])==(base,cohort,scope))
    primary=[contrast(base,'native','one_second') for base in ('B_MemoryCV','B_HistoricalCV')]
    masked_primary=contrast('B_MemoryCV','masked','one_second')
    secondary_scope='root_anchor_supported_correspondence_unambiguous_one_second'
    def reviewed_contrast(base,metric,scope=secondary_scope):
        return next((r for r in b['contrasts'] if
            (r['baseline'],r['cohort'],r['scope'],r['metric'],r['streams'])==
            (base,'native',scope,metric,'both')),None)
    secondary_comparison=table(['baseline','secondary native D minus baseline path ES',
        'secondary native D minus baseline endpoint ES'],
        [(base,ci(reviewed_contrast(base,'path_es')),ci(reviewed_contrast(base,'tracked_es')))
         for base in ('A_CV','B_MemoryCV','B_HistoricalCV','C_Previous')])
    masked_calibration=[r for r in b.get('scoped_calibration',[]) if
        (r['model'],r['cohort'],r['scope'])==('D_StopMove','masked','one_second')]
    native_calibration=[r for r in b.get('scoped_calibration',[]) if
        (r['model'],r['cohort'],r['scope'])==('D_StopMove','native','one_second')]
    native_calibration_table=table(['native age[.75,1.25] future H s','coverage50',
        'coverage90','radius90 m','original status support'],
        [(r['H'],*(ci(r['metrics'][metric]) for metric in
            ('coverage50','coverage90','radius90')),str(r['statuses']))
         for r in native_calibration])
    native_one_second_coverage=ci(next(r for r in native_calibration if r['H']==1.)['metrics']['coverage90'])
    practical_recovery=[r for r in b['reobservation'] if r['model']=='D_StopMove' and r['gap_bin']=='one_second']
    recovery_half=next(r for r in practical_recovery if r['H']==.5)
    masked_calibration_table=table(['masked age[.75,1.25] future H s','coverage50',
        'coverage90','radius90 m','original status support'],
        [(r['H'],*(ci(r['metrics'][metric]) for metric in
            ('coverage50','coverage90','radius90')),str(r['statuses']))
         for r in masked_calibration])
    masked_two_second=next((r for r in masked_calibration if r['H']==2.),None)
    masked_two_second_coverage=ci(masked_two_second['metrics']['coverage90']) if masked_two_second else 'unsupported'
    monte_carlo=readj(OUT/'primary_agent_notes/native_monte_carlo.json')
    if not monte_carlo['sources_stable']:raise ValueError('unstable Monte Carlo sensitivity readout')
    mc_rows=[r for r in monte_carlo['results'] if r['baseline'] in ('B_MemoryCV','B_HistoricalCV') and r['scope']!='root_visible']
    mc_table=table(['scope','baseline','common target cells','stream41 path contrast',
        'stream107 path contrast','107 minus41 contrast'],
        [(r['scope'],r['baseline'],r['common_cells'],ci(r['common_streams']['41']['path_es']),
          ci(r['common_streams']['107']['path_es']),ci(r['stream107_minus41_on_same_cells']['path_es'])) for r in mc_rows])
    native_numeric_pass=all(r['path_es']['roots']>=30 and r['path_es']['scenes']>=10 and r['path_es']['hi']<0 for r in primary)
    native_support_pass=all(r['path_es']['roots']>=30 and r['path_es']['scenes']>=10 for r in primary)
    native_conclusion=('The native path-ES interval/support threshold clears on supported annotation-valid development targets; other readiness requirements still fail.' if native_numeric_pass else
        'Native aggregate support exceeds the30-root/10-scene threshold, but both Memory-CV path-ES contrast intervals include zero. Native path superiority is inconclusive despite sufficient aggregate support.' if native_support_pass else
        'Native aggregate support does not clear30roots/10scenes; native one-second superiority is inconclusive.')
    # Measured coverage/physical defects require revision independently of ES.
    labels=['FORECASTING REVISE','BELIEF / REOBSERVATION REVISE']
    if not systems_pass:labels.append('SYSTEMS REVISE')
    general='All scores are development evidence. Values are equal-scene root means [two-sided95% scene-bootstrap interval] (unique roots/scenes). People, horizons, streams, masks and siblings never become independent physical futures. The intervals do not account for checkpoint selection on this development population.'
    comparison=table(['cohort','baseline','scope','D minus baseline path ES','D minus baseline endpoint ES'],
        [(r['cohort'],r['baseline'],r['scope'],ci(r['path_es']),ci(r['tracked_es'])) for r in summary['contrasts'] if r['scope'] in ('visible','one_second')])
    def d_physics(metric):return next(r for r in physics['summary'] if
        (r['model'],r['metric'])==('D_StopMove',metric))
    d_stop=d_physics('stop_frequency')
    d_speed=d_physics('speed_p95');d_turn=d_physics('turn_p95_rad')
    d_restart=d_physics('restart_frequency');d_duration=d_physics('stop_duration_p95')
    visible_scope='root_anchor_supported_correspondence_unambiguous_visible'
    scorecard=table(['scientific capability','evidence and limitation'],[
        ('A. Visible forecasting','Secondary supported-root/unambiguous D minus Memory-CV path ES '+
            ci(reviewed_contrast('B_MemoryCV','path_es',visible_scope))+'; full matrices in FORECASTING.'),
        ('B. Intermittent forecasting','Artificial near1s D minus Memory-CV path ES '+ci(masked_primary['path_es'])+
            '; native path superiority is inconclusive, native H1 nominal90% coverage '+native_one_second_coverage+
            '; separate masked H2 coverage '+masked_two_second_coverage+'.'),
        ('C. Reobservation','Same-target gains/support are reported, but dense anonymous ambiguity, center ordering, slot reuse and unavailable-initialization witnesses fail.'),
        ('D. Physical behavior','New native stop frequency '+ci(d_stop['predicted'])+
            ' versus matched truth '+ci(d_stop['actual'])+'; restart/speed/turn/duration and trajectory tradeoffs remain measured.'),
        ('E. Recursive model','Finite-law direct/segmented joint-state, higher-moment and characteristic-function discrepancies are at numerical precision; retained-tape RNG continuation passes.'),
        ('F. Observation model','Finite event normalization/conditioning are coherent; measured optical-probability bias and exposure censoring limit calibration.'),
        ('G. Joint prediction','Shared temporal worlds and cross-query reuse are implemented; marginal fitting and differing supported target sets prevent joint-superiority claims.'),
        ('H. Systems',f"Complete-call P95 {warm['p95_ms']:.2f}ms warm /{cold['p95_ms']:.2f}ms dynamically cold {gate_text}100ms; dense stress and startup costs remain separate.")])
    write('FINAL_REPORT.md',f'''# Intermittent-observation Layer 2 — scientific decision

**{' / '.join(labels)}.** This is a trained, evaluated research release, not an
unqualified integration promotion. No Layer2.5/L3/L4, query selector, navigation
policy, SR/SPL or navigation-regret experiment was implemented.

**Speed priority: the declared representative100ms P95 target {gate_text}.**
The complete call measures{warm['p95_ms']:.2f}ms warm and
{cold['p95_ms']:.2f}ms dynamically cold, versus247.23/241.47ms before optimization
on the SAME cohort/checkpoint. All32belief components,16worlds/3s,64timestamps,
actual updating, optical events and supported successor work remain included.
Dense P95 remains{dense_warm['p95_ms']:.2f}/{dense_cold['p95_ms']:.2f}ms;
preferred65ms and fresh-process100ms operation are not achieved. Startup/JIT is
separate. See [PERFORMANCE](PERFORMANCE.md) for full P50/P95/P99 and limits.

The33,309-parameter Stop/Move graph model completed{train['completed_epochs']}
epochs and selected epoch{selected['epoch']}. Component-conditional development
path ES changed from{learning[0]['dev_es']:.6f} at initialization to
{selected['dev_es']:.6f}. Checkpoint SHA256`{train['checkpoint_sha256']}`.
The frozen17-epoch baseline and128,990-outcome snapshot remain unchanged.
Only3,212 independent training roots directly supervise learned trajectories;
128,990 outcomes are not128,990 independent neural futures.

The implementation preserves anonymous persistent actual tracks, exact arrived
position assimilation, continuous multimodal targets, small-noise stopped
motion, restart dynamics, shared temporal worlds, conditional finite successors
and generic region/time projections. Evidence-free recursion keeps latent clocks
and Gaussian components rather than discarding higher-order mixture information.
Actual observations update a separately retained continuous belief.

Inherited target ambiguity also limits the fit:40training roots/33scenes and
6development roots/6scenes contain nonunique native correspondence;30training
and3development roots duplicate selected physical targets. The new target
expansion preserves the original prefix/features. Independent hash checks find
no new feature leakage. These cached fitting ambiguities were not retrospectively
removed or used to refit. Final scored people require separate unique legal
correspondence. DATA_AND_GAPS and TRAINING state the exact support and limits.

{general}

The corrected native ledger's practical stratum is last-observed age
[.75,1.25]s, forecasting the next.5/1/2s. Its paired
new-minus-MemoryCV path ES is
**{ci(primary[0]['path_es'])}**; versus historical-diffusion CV it is
**{ci(primary[1]['path_es'])}**. Negative favors the new model.
Native support and all individual horizons are in INTERMITTENT_OBSERVATIONS.
On the same native target cells, the two streams give Memory-CV path contrasts
-.115653 and-.046710; their difference is+.068943 [.007911,.131034]. This
material finite-carrier/filter/forecast sensitivity is retained, not resolved
by choosing the favorable stream. Primary results average both within root.
Offline accounting checks birth identity and common first-association provenance
in every same-origin branch. A pre-result annotation rule censors post-root
identity metrics on30 roots/24scenes with ambiguous exact-center correspondence;
raw model failures, unsupported ancestry and original targets remain counted.
These changes repair score validity without changing the predictor. The weaker
research partials remain superseded evidence. EVALUATION_ACCOUNTING documents
the correction, executed sources, conservative support loss and chronology.
ReviewerB also reports a separate, secondary native scope restricted to people
with supported ROOT anchors and unambiguous correspondence:

{secondary_comparison}

The separate controlled one-second artificial-mask path contrast is
**{ci(masked_primary['path_es'])}** against Memory-CV. It tests unavailable
measurements under the preserved actual robot/static history.
Synthetic-mask results remain separate and cannot establish native occlusion
or camera-turn superiority. {native_conclusion}

Calibration remains a material forecasting limitation: after the approximately
one-second NATIVE gap, nominal90% coverage at a further1s is
**{native_one_second_coverage}**. After the separate one-second artificial mask,
the nominal90% region at a further2s has coverage
**{masked_two_second_coverage}**. The primary contrast averages available
.5/1/2s targets within each root; its root count must not be read as2s support.
The finite32-world radial regions are not exact or conformal regions.

Strong finite evidence supports persistent memory, source-expiry independence,
same-time real observation after propagation, same-law training/inference, and
recursive mixture consistency. The motion-noise representation can now express
stopping. On666 matched observed-person roots/98scenes, predicted stop frequency
is{d_stop['predicted']['mean']:.3f}, versus{d_stop['actual']['mean']:.3f} observed
and.005 in the previous trained model on that SAME cohort. This materially
reduces the demonstrated mismatch, but does not clear physical calibration:
restart frequency remains{d_restart['predicted']['mean']:.3f} versus
{d_restart['actual']['mean']:.3f}; root/scene-averaged speed P95 is
{d_speed['predicted']['mean']:.3f} versus{d_speed['actual']['mean']:.3f}m/s and
turn-angle P95 is{d_turn['predicted']['mean']:.3f} versus
{d_turn['actual']['mean']:.3f}rad. Completed stop-duration tails are also too
long. PHYSICAL_VALIDATION reports paired intervals and censoring. Reobservation gains are evaluated
at identical absolute targets with fixed pre-arrival ancestry and explicit
unsupported denominators.

Native approximately-one-second reappearance has only22roots/16scenes for
immediate focal-slot diagnostics: correct preferred-antecedent mass is
{ci(recovery_half['metrics']['association_accuracy'])}; focal newborn/duplicate
mass is{ci(recovery_half['metrics']['duplicate'])}. The subsequent.5s future-only
path gain is{ci(recovery_half['metrics']['future_path_es_gain'])}. Its18roots/
13scenes fall below the30-root gate;1s/2s gains have only9/1roots. Thus native
one-second recovery is both poorly associated and insufficiently supported for
a predictive-improvement claim. Broader short-gap updates show useful gain
without establishing this primary reappearance requirement.

A belief requirement independently fails. A symmetric four-person/three-center
association witness loses16 of24 valid assignments in the bounded beam and
makes one ancestor falsely certain. The enumerated-retention diagnostic does
not bound earlier pruning. Permuting those same anonymous centers changes one
posterior mean by .075 m despite an unchanged evidence receipt. These are
concrete belief/reobservation defects,
and actual retained-state diagnostics are also material:23351zero-true-count-
support states occur across609roots/94scenes;28700states have positive duplicate-
ancestry mass across750roots/97scenes, out of125282retained D states. These are
dependent frame/stream records, not independent trials, and do not prove that a
generic person was deleted. They limit coherent count/association support for
observed humans even though never-observed spatial superiority is secondary.
Exact coordinates of accepted measured slots remain correct; easier successful
examples do not clear these failed requirements. After the user's explicit speed
priority, the separately versioned implementation reduces representative
complete-call P95 to{warm['p95_ms']:.2f}ms warm and{cold['p95_ms']:.2f}ms
dynamically cold; dense P95 is{dense_warm['p95_ms']:.2f}/{dense_cold['p95_ms']:.2f}ms.
The100ms integration target {gate_text}; the65ms preferred target {preferred_text}.
Full event/successor work,64timestamps and32actual components remain included.
The unchanged-cohort pre-speed reference was247.23/241.47ms. Startup/JIT costs
are reported separately. Scientific evaluation remains attributed to frozen
`344/a527/a278` execution; source-bound full-state/trajectory/RNG/conditioning
comparisons establish the optimized implementation's numerical equivalence.
Hard region boundaries can amplify floating-point differences for samples
within numerical precision of a boundary; universal bitwise projection equality
is not claimed. PERFORMANCE and reviewerA document the finite witness.

A separate setup-boundary witness also fails: initializing on a final unavailable
frame wrongly installs an evidence receipt and blocks a genuine same-time
center. The retained-belief observe-unavailable/observe-genuine sequence succeeds.
Inherited unavailable replay itself is correct. Aged bootstrap last-position
metadata is a predicted mean, not a recoverable old measurement; timestamps
remain genuine. These limitations are explicit in ARCHITECTURE/API and reviewerA's
initialization supplement. No arbitrary-initialization readiness is claimed.

Available misses also renew descendant future clocks. An independent32-carrier
witness holds the GH-updated continuous posterior fixed while preserving versus
renewing those clocks: at+.5s Stop mass changes .125 to .28125, the position mean
differs .08320m and a joint characteristic function differs .31016. This is a
material finite conditional approximation, distinct from the successful
evidence-free semigroup and from systematic native calibration evidence.
Saved association scalars are overlapping focal-slot diagnostic masses, not
exhaustive false-merge rates. Those evaluation limitations reinforce the belief
revision decision.

Never-observed spatial superiority is not an acceptance gate. The broad retained
unlocalized channel remains uncalibrated; zero true-count support and failures
that can corrupt observed-track associations are exposed rather than hidden.
No count floor or invented recovery detection was introduced.

{comparison}

{scorecard}

Reports cover data/censoring, fitting, visible/intermittent forecasts,
reappearance, full-distribution recursion, physical motion, optics, joint paths
and systems. Three focused independent agent reviews provide finite
counterexamples and checks; they are independent perspectives within this
research team, not external certification. Passing{validation['total_tests']}
regressions does not override the failed scientific gates. Reproduction and
source/checkpoint manifests accompany the committed research release. Production
collection remained independent; there is no production merge.
''')
    age_rows=[];reob_rows=[]
    for split in ('train','dev'):
        for name,_,_ in AGE_BINS:
            item=gaps['counts'].get(split+':native_age:'+name,{})
            age_rows.append((split,name,item.get('events',0),item.get('roots',0),item.get('scenes',0)))
        for threshold in (.25,.5,1.,1.5,2.):
            item=gaps['counts'].get(split+':reobservations_ge'+str(threshold),{})
            reob_rows.append((split,threshold,item.get('events',0),item.get('roots',0),item.get('scenes',0)))
    proxy=readj(DOC/'reviews/reviewer_b_initial_source_proxy.json')
    strict_check=readj(DOC/'reviews/reviewer_b_final_strict_gap_checks.json')
    if not strict_check['all_checks_passed'] or strict_check['strict_audit_sha256']!=sha(OUT/'data_and_gaps_strict.json'):
        raise ValueError('strict gap review does not bind the adopted audit')
    multitimestamp_rows=[]
    for correction in strict_check['reporting_corrections']:
        raw=correction['original'];positive=correction['positive_occurrence_accounting']
        multitimestamp_rows.append((correction['key'],raw['events'],raw['roots'],raw['scenes'],
                                    positive['roots'],positive['scenes']))
    source_rows=[]
    for split in ('train','dev'):
        for name in ('source_ids_ever_positive','source_ids_multitimestamp','root_previously_seen_sources_absent_from_current_source_tracker','source_reappearance_gap_ge:1.0'):
            q=proxy['counts'].get(split+':'+name,{})
            source_rows.append((split,name,q.get('events',0),q.get('roots',0),q.get('scenes',0)))
    gap_features=[]
    for split in ('train','dev'):
        for name in ('nodes_without_native_truth','unmatched_positive_centers','ambiguous_exact_physical_matching','physical_chains_censored_for_matching_ambiguity','close_human_association_cases','reobservation_yaw_change_over15deg','reobservation_yaw_change_under15deg','gap_contains_stop','gap_stop_move_change','gap_contains_turn','reobservation_has_previous_positive'):
            q=gaps['counts'].get(split+':'+name,{})
            gap_features.append((split,name,q.get('events',0),q.get('roots',0),q.get('scenes',0)))
    quantiles=[(name,item['n'],*('%.3f'%v for v in item['quantiles']))
               for name,item in sorted(gaps['distributions'].items())]
    correspondence=b.get('canonical_correspondence_audit') or {}
    correspondence_rows=[(name,item['events'],item['roots'],item['scenes'])
        for name,item in sorted(correspondence.get('counts',{}).items())]
    annotation_review=readj(DOC/'reviews/reviewer_b_final_owned_annotation_checks.json')
    if not annotation_review['all_checks_passed']:raise ValueError('annotation preservation review failed')
    write('DATA_AND_GAPS.md',f'''# Frozen data and full legal-sequence gap audit

Source cut:128,990 outcomes,17,912 roots,864 scenes. Manifest SHA256
`{gaps['snapshot_sha256']}`. Eligible payload audit:12,045 roots and87,085
outcomes (train9,935/72,539/467scenes; dev2,110/14,546/108scenes).
Protected source development/confirmation, research holdout and previously
opened fresh scenes remain excluded. "Full sequence" means every legal DAG
node in eligible sealed payloads, not opening protected payloads. Sibling node
events are dependent; unique roots/scenes govern scientific support.

The inherited supervised cache is not wholly uniquely identified. An independent
audit of all3,878native supervised archives verifies every manifest hash, the
old/new target prefix, declared nearest-native extension, unchanged nontruth
features/weights/selection and no new online future leakage. Training has
40roots/33scenes with nonunique physical correspondence (520component-target
instances;414nearest ties), and30roots/27scenes/180components with duplicate
selected physical targets. Development has6roots/6scenes/66nonunique instances
(36ties) and3roots/3scenes/18components with duplicates. These dependent
alternatives do not become extra independent futures. Their direct expected
root-balanced ambiguous-loss weight is .889892%train/.510511%dev; duplicate
excess is .366853%/.137638%. Those small fractions do not bound fitting effects:
ambiguous candidates can separate by3.239mtrain/2.557mdev. No refit or post-hoc
target repair was made. All6ambiguous-supervision dev roots are in the separate
30-root canonical censor; all666supervised dev roots still have at least one
independently unique legal evaluation target. See reviewerB's training-target
integrity ledger and `reviewer_b_training_target_legality.md`.

The old root-state P95 age.488s and maximum1.6s did not establish a lack of
longer history. The audit retains every genuine positive in the full pre-root
history and follows offline post-root position correspondence through
the legal observation DAG. The PRIMARY inventory below uses strict unique-center
physical chains. An ambiguous positive censors each affected person's chain;
the next unique positive starts a new anchor, not a gap spanning that ambiguity.
An earlier argmin-following audit remains preserved as historical evidence, but
its apparent physical continuity does not supply these strict gap counts.
Current and intervening ambiguous measurements are observations, never false
empty or unavailable evidence. Physical identity labels never enter inference.
Pre-root physical truth is unavailable, so pre-root source-ID continuity is a
separately named proxy. It may split/reuse IDs and cannot establish physical
association accuracy. Previously observed sources missing from the current
source tracker are retained by this proxy audit.

The separate canonical correspondence audit covers all2,110 development roots
and bounds ages when a genuine ambiguous or unmatched center intervenes after
the last unique match. Such a center is an observation, not unavailable sensing.
Its analysis-only exclusions are shared across models, horizons and streams;
they never change online inputs or the original primary raw rows. Root-supported
initial ancestry additionally excludes later-born lifetimes from the secondary
native analysis. Neither restriction establishes an optical occlusion cause.

{table(['canonical correspondence category','dependent events','roots','scenes'],correspondence_rows)}

Native target-key agreement with the final eligibility ledger:
`{correspondence.get('all_native_target_decisions_match')}`. The original raw
reappearance flag is`{correspondence.get('all_reappearance_events_match')}` because
the canonical index is an integer while emitted decision keys are strings.
The separately preserved type-normalization supplement verifies exact equality
of all117physical event keys and all3744model/horizon/stream rows:
`{event_check['event_key_normalization']['normalized_all_reappearance_events_match']}`.
Only the index representation is normalized; no scientific row or raw flag is
changed. See `reviewer_b_final_event_key_checks.json`.

For final native scores, the separately declared annotation rule conservatively
censors the entire post-root identity ledger on30 roots/24scenes, even when
only one person's shorter interval was ambiguous. The strict all-DAG inventory
uses chain-level censoring instead. Their scopes differ intentionally and must
not be merged into a single denominator. Birth-qualified fixed ancestry and
all-component support remain additional scoring requirements.

The independent raw-to-accounted ledger below preserves original operational
statuses even inside annotation-censored roots. Those rows move to uncertain
age/gap categories rather than being silently declared successful or erased.
Only joint-vector rows without reliable person annotations remain archive-only;
all physical target, reappearance and update-gain rows remain present.

{table(['ledger','accounting category','dependent rows'],[(r['kind'],r['category'],r['rows']) for r in annotation_review['row_accounting']])}

{table(['split','native age','dependent person-node events','roots','scenes'],age_rows)}

{table(['split','native reappearance gap >=s','dependent events','roots','scenes'],reob_rows)}

A reappearance requires an intervening legal frame without that person's
positive measurement; ordinary native sampling intervals are not mislabeled
occlusion. Full quantiles of genuine intermeasurement intervals, history age,
native age, reappearance gaps and duration are in`artifacts/raw/data_and_gaps_strict.json.gz`.

{table(['split','pre-root SOURCE-ID proxy','events','roots','scenes'],source_rows)}

The full strict audit's multi-timestamp source-track support has a producer
accounting defect: zero-occurrence calls added every eligible root/scene to two
support fields. ReviewerB independently reconstructed all172 count keys. Every
event total and all native age/gap counts agree. The following derived view
corrects only those two positive-occurrence root/scene counts; the original
strict JSON and its hash remain unchanged. Source IDs are continuity proxies.

{table(['strict source-track key','tracks across timestamps','raw roots','raw scenes','positive roots','positive scenes'],multitimestamp_rows)}

{table(['split','offline sequence diagnostic','events','roots','scenes'],gap_features)}

Yaw counts compare last-positive and reappearance robot yaw. They are observed
orientation changes, not proof that rotation caused the missed detection; a
turn-away-and-back can have small net yaw. The graph is not an optical map, and
we do not invent native occlusion causes. Close centers are an association
difficulty proxy, distinct from exact ambiguous physical matching. Stop/turn
counts use native truth only as an offline diagnostic.
Full-DAG stop/turn windows use native searchsorted boundaries and can omit a
partial starting interval or include an interval beyond reappearance. The
canonical context audit below instead interpolates exact observation endpoints.
`physical_chains_censored_for_matching_ambiguity` counts candidate-ID incidences
at ambiguous nodes, including candidates never anchored, not distinct deleted
persistent tracks.

Strictly matched post-root ancestry establishes a lower bound on previously observed
physical people: pre-root-only expired people cannot all be physically labeled.
Native forecasting includes ever-positive people even if they never reappear.
Reappearance gains use first-person/gap-bin events and report their separate
eligibility and censoring. Native future truth ends are never stationary target
padding. Full age/horizon supported/censored counts and independent roots/scenes
are in the raw gap audit and reviewerB score/support ledger.

{table(['descriptive distribution / source-ID proxy','dependent n','min s','P25','P50','P75','P95','P99','max s'],quantiles)}

## Strict canonical pre/post gap context

The following is a separate audit of the final annotation-verified,
model-independent reappearance eligibility ledger, deduplicated across the
two streams. It records genuine positive
timestamps immediately before/after each gap, every intervening sensor's
availability/validity/raw-positive count, and unique offline focal matching.
Pre-root physical correspondence is not invented. Robot-yaw excursion also
catches turn-away-and-back patterns; neither net yaw nor excursion establishes
an optical cause. Motion changes use native truth only as an offline diagnostic.

{table(['gap group s','context feature','dependent events','roots','scenes','future truth .5/1/2/3s: events/roots/scenes'],[(r['gap_group'],r['feature'],r['events'],r['roots'],r['scenes'],str(r['future_truth'])) for r in context['summary']])}

Exact context and censoring are preserved in `artifacts/raw/gap_context.json.gz`.
Strict matching still means verified correspondence rather than a full optical
cause label; ambiguous relevant centers must not be called certified occlusion.

Artificial masks are a controlled, separate experiment. All human centers are
removed and`human_sensor_available=False`; actual timestamps, arrived robot pose
and chronological legal static updates remain. Root-visible people are advanced
to actual endpoints near.5/1/2s within.06s. Withheld positions are offline targets
only. Available endpoint and missing-endpoint support is:

{table(['requested age','eligible root-positive roots/scenes','endpoint roots/scenes','no endpoint roots/scenes'],[(r['requested_age'],str(r['root_positive_eligible']),str(r['legal_endpoint_available']),str(r['no_legal_endpoint_near_requested_age'])) for r in b['synthetic_endpoint_eligibility']])}

Masks are not available empty frames, new physical futures, native camera
rotation or native occlusion. They never multiply root support. Exact masked
timestamps are preserved in the mask manifest. The preliminary source-proxy and
legal-cache audit preceded fitting; the expensive all-DAG audit overlapped
training and completed afterwards. This chronology is a disclosed deviation
from completing every audit first, not a claim that its final native counts
selected the model. No new collection campaign was launched.
''')
    peak=max(r['rss_kb'] for r in learning)/1024
    gpu=max(r['gpu_peak_bytes'] for r in learning)/(1024**2)
    write('TRAINING.md',f'''# Actual fitting and selection

One33,309-parameter width64 revised graph candidate was trained on3,212
independent root-visible trajectory-supervised roots. Selection uses666
development motion roots. The same immutable train/dev scenes and cached legal
features as the previous release are retained; native truth extends targets
where available. Other eligible training roots contribute to the train-only
scalar motion-law fit (8,571 native-motion roots), not invented neural examples.
The previous checkpoint is frozen; its17epochs were not assumed converged.

The revised network starts from seed221 initialization, rather than the previous
trained weights. Architecture, stochastic law, objective and training budget
differ, so its comparison to the old checkpoint is not a causal Stop/Move-only
ablation. The independent inherited-cache audit finds40training roots/33scenes
with ambiguous physical target correspondence and30roots/27scenes with duplicate
selected targets; development has6/6and3/3respectively. Exact counts, direct loss
weights and candidate separations are in DATA_AND_GAPS and reviewerB's source-
bound audit. New expansion preserves old target prefixes and nontruth features;
there is no evidence of new future leakage. These inherited ambiguities limit
fit/selection evidence and were retained without a post-evaluation refit.

Root-balanced Monte Carlo path Energy Score analytically weights three
continuous target modes and initial Stop/Move alternatives. Four independent
paths per alternative train; eight fixed-seed paths select. Each root draws one
old legal bootstrap association/maneuver component. Thus this is a proper
component-conditional marginal path objective, not a full-posterior joint-human
likelihood. Per-person scoring does not identify a joint-human copula.

Half the draws request longer gap-shifted target vectors from ages.5/1/1.5/2s;
the others use four ordinary native points up to3s. Short available futures
fall back or censor according to the recorded selection logs. These targets
extend evidence-free prediction from root history. They are not filtering
training on artificially available empty frames, and do not train a new
anonymous association network. No withheld position becomes an input.

The same continuous-time exponential-switch/integrated-OU/reset law is sampled
in training and inference. Train-only hazards fit finite-difference speed-state
transition proxies; the exact fitted configuration is`artifacts/motion.json`.
The independent moment-ODE and training/inference path comparisons are in
reviewerA's evidence. Gaussian full-path NLL is not substituted for the law.

AdamW LR.001, weight decay.001, batch48roots, gradient norm5, seed221,
cosine LR floor.00005, minimum30/max80epochs. Significant-best patience12
compares cumulative improvement>=.001 against the last significant best.
The absolute best trained checkpoint is selected independently of that
patience reference. Completed{train['completed_epochs']}epochs; selected
epoch{selected['epoch']}; last significant epoch{train['last_significant_epoch']}.
Stop reason`{train['stop_reason']}`. Initial devES{learning[0]['dev_es']:.9f};
selected{selected['dev_es']:.9f}. This is a conventional completed stopping-rule
experiment, not proof of global optimization convergence.

Initial, selected and final checkpoints retain optimizer, scheduler, NumPy,
Python, Torch and CUDA RNG states. Selected SHA256`{train['checkpoint_sha256']}`.
Training wall{train['seconds']:.2f}s; CPU{train['cpu_seconds']:.2f}s;
peak recorded RSS{peak:.1f}MiB; peak allocated tensor GPU memory{gpu:.2f}MiB.
GPU tensor memory excludes context/reservation. One numerical CPU thread, loader
workers0, nice19/idle I/O and a20% CUDA allocation cap preserved collector room.
NVML utilization was unavailable; it was not repaired or fabricated.

Epoch0 is a frozen untrained control under the same scalar law. The selected
checkpoint's no-gap-objective diagnostic has a different target distribution;
its absolute score is not a causal benefit estimate for gap weighting. There
is no separately selected architecture or broad baseline search. Dynamics law,
objective, epoch budget and input-state distributions differ from the prior
release, so a causal improvement cannot be assigned to one factor alone.

Hash-matched executed fitting sources and dependencies are preserved in
`training_sources/`. The training-time runtime model file hash is recorded but
its full bytes could not be reconstructed; fit.py did not import that module.
Post-fit runtime precision/ownership/receipt/tape and same-law optimization
changes are separately source-bound and numerically validated. The original
PROTOCOL and its hash remain unchanged; EVALUATION_FREEZE records pre-result
evaluation corrections. No protected confirmation scenes were opened.

![Learning curves](figures/03_learning.png)
''')
    visible=[r for r in summary['cells'] if r['cohort']=='native' and r['age_bin']=='visible']
    write('FORECASTING.md',f'''# Forecasting immediately after legal observations

{general}

An observed target has a genuine current anonymous center. Scores retain fixed
physical ancestry for the entire future path; no independent per-time identity
rematching occurs. The main endpoint Energy Score and RMS-scaled finite-vector
path Energy Score use iid-world off-diagonal U-statistics. The path coordinates
are0,.25,.5,.75,1,1.5,2,3s truncated atH. Strict propriety concerns that declared
finite path vector, not every possible continuous trajectory functional.
Future-only path scores omit the current measured coordinate.

{table(['model','H s','endpoint ES m','path ES m','coverage50','coverage90','radius90 m'],[(r['model'],r['H'],ci(r['metrics']['tracked_es']),ci(r['metrics']['path_es']),ci(r['metrics']['coverage50']),ci(r['metrics']['coverage90']),ci(r['metrics']['radius90'])) for r in visible])}

A_CV uses zero-acceleration future forecasts on common Memory-CV legal-history
actual estimates. B_MemoryCV uses acceleration diffusion.16; B_HistoricalCV
uses the frozen historical.64 comparator. These are persistent legal-history
kinematic controls, not a baseline search. C_Previous preserves the43187f24
checkpoint and scalar motion law on common B filtering states. It is a matched
dynamics comparison, not byte-identical replay of the old entire filtering
release. D_StopMove is the selected trained model; E_Initial is epoch0.
All use the same root/scene targets and declared scored object.
The.16/.64 settings are the strongest TESTED Memory-CV laws in this bounded
predeclared set. There is no separately fitted sticky-kinematic control or
equal-budget old-network training control. E_Initial tests learning within the
revised law, not those missing causal controls. No claim of beating every
possible kinematic baseline or isolating the effect of gap weighting/StopMove
alone follows from these comparisons.

{comparison}

Across all currently-visible native decisions (root and scheduled updates),
D minus Memory-CV path ES is{ci(contrast('B_MemoryCV','native','visible')['path_es'])}.
D minus the previous checkpoint is
{ci(contrast('C_Previous','native','visible')['path_es'])}; its interval includes
zero. The auxiliary root-only visible population is a different narrower scope.
A favorable root-only comparison does not establish superiority over the previous
checkpoint at every currently-visible decision.

Visible summaries combine root and predeclared later canonical decisions within
root. E_Initial has only root/masked evaluations, so its paired common support
is narrower. Native future truths are linearly interpolated OFFLINE to physical
H=.5/1/2/3s; missing durations are censored. All cells' original denominators,
unsupported ancestry, duplicate lineages and failed models are in reviewerB's
independent ledger. Finite-only score superiority cannot excuse lost support.
The corrected native ledger fixes an ancestor's slot AND birth timestamp at its
first genuine common explanation across every positive-weight same-origin fork.
Recycled slots cannot rescue a missing preferred ancestor. A conservative
pre-result annotation rule censors post-root identity metrics on30roots/24scenes
with ambiguous physical correspondence; original eligible targets and model
failures remain counted. EVALUATION_ACCOUNTING identifies exact executed sources.
ReviewerB's secondary supported-root-anchor/unambiguous native scope is narrower
development evidence, reported separately rather than substituted for the
corrected primary. Root-only and masked physical ancestry never acquires later
labels from withheld measurements.

50/90% coverage uses empirical radial balls about the ensemble mean; radius
is reported alongside it. These32-world regions are finite-sample diagnostics,
not conformal or exact posterior regions. The empirical90% quantile contains
28/32=.875 of the same generic continuous sample, and estimating the center
introduces additional finite-sample effects; not every difference from.900 is
a bias-corrected law-level rejection. The practical H2 masked coverage failure
is substantial even against.875. Calibration error is coverage minus
nominal; either direction is a defect. Velocity/direction/stop diagnostics use
matched.048-second displacement intervals in prediction and interpolated truth.
Their finite-world Brier estimates contain sampling error/bias; no exact
continuous-law classification calibration is claimed. Native-interval physical
validation is separately authoritative for physical frequency statements.

{table(['model','H s','endpoint mean error m','velocity mean error m/s','velocity coverage90','direction Brier','physical stop Brier'],[(r['model'],r['H'],ci(r['metrics']['endpoint_error']),ci(r['metrics']['velocity_error']),ci(r['metrics']['velocity_coverage90']),ci(r['metrics']['direction_brier']),ci(r['metrics']['stop_brier'])) for r in visible])}

Joint rows preserve one whole vector of people under each world. Raw summaries
stratify by cohort,H and number of people. Marginal training does not learn all
cross-human dependence, and incomplete or differing target sets prevent a joint
superiority claim. Temporal correlation is retained in every scored path.
Never-observed full-population spatial SOTA is not an objective of this release.

{table(['model','cohort','H s','people in joint scored vector','descriptive joint path ES'],[(r['model'],r['cohort'],r['H'],r['people'],ci(r['joint_path_es'])) for r in summary['joint']])}

These joint rows are descriptive: the same number of people does not guarantee
the same supported physical target set across models. No joint-superiority
claim is made without matched target vectors. Every row still uses one shared
world index across its people and times.
''')
    matrices=[]
    for cohort in ('native','masked'):
        cells=[r for r in summary['cells'] if r['model']=='D_StopMove' and r['cohort']==cohort]
        matrices.append('## '+cohort+' decisions\n\n'+table(['age bin s','H s','endpoint ES','path ES','coverage50','coverage90','diversity','variance trace'],
            [(r['age_bin'],r['H'],ci(r['metrics']['tracked_es']),ci(r['metrics']['path_es']),ci(r['metrics']['coverage50']),ci(r['metrics']['coverage90']),ci(r['metrics']['diversity']),ci(r['metrics']['variance_trace'])) for r in cells]))
    secondary_cells=[r for r in b['cells'] if r['model']=='D_StopMove' and
        r['cohort']=='native_root_anchor_supported_correspondence_unambiguous']
    secondary_matrix=table(['secondary native age bin s','H s','endpoint ES','path ES',
        'coverage50','coverage90','radius90 m'],
        [(r['age_bin'],r['H'],*(ci(r['metrics'][metric]) for metric in
            ('tracked_es','path_es','coverage50','coverage90','radius90')))
         for r in secondary_cells])
    def support(value):
        return '%d rows / %d roots / %d scenes'%(value.get('rows',0),value.get('roots',0),value.get('scenes',0))
    support_rows=[]
    for row in b['cells']:
        if row['model']!='D_StopMove':continue
        support_rows.append((row['cohort'],row['age_bin'],row['H'],
            str(row['eligible_including_truth_censored']),
            *(support(row['statuses'].get(status,{})) for status in
              ('supported','truth_censored','ancestry_unsupported','identity_annotation_censored','model_failed'))))
    write('INTERMITTENT_OBSERVATIONS.md',f'''# Last-observed age and future horizon

{general}

Age is current timestamp minus last genuine legal position-measurement time;
future H starts at the CURRENT timestamp. Frozen bins: visible<=1e-7;
(1e-7,.25],(.25,.5],(.5,1],(1,1.5],(1.5,2],>2s, with1e-8 numerical edge tolerance.
The practical one-second stratum is[.75,1.25]s. Native and masked populations
never share an estimate. Each matrix cell states its own unique support.
Age labels require uniquely verified offline correspondence. The corrected
primary censors post-root identity metrics on30roots/24scenes with ambiguous
genuine centers and fixes slot/birth provenance across all same-origin branches.
No ambiguous positive becomes negative or unavailable evidence. The independent
canonical audit finds no ambiguous-intervening match in the practical native
one-second stratum. Separate supported-root/unambiguous secondary results remain
explicitly restricted; annotation-valid finite scores never erase model or
ancestry failures. Weaker superseded partials are archived rather than pooled.

{''.join(matrices)}

## Secondary supported-root native scope

This analysis requires genuine root-positive people with supported initial
all-component ancestry and no intervening ambiguous correspondence. Its
selection was declared from the source invariant and legality audit before
opening final native contrasts. It is development evidence on a restricted
population, not replacement of the original all-ever-seen primary.

{secondary_matrix}

{secondary_comparison}

## Original denominators and unsupported targets

{table(['cohort','age bin','H s','eligible unique support','supported','truth censored','ancestry unsupported','annotation censored','model failed'],support_rows)}

Raw rows include repeated people/streams/decisions and are never independent
sample counts. Root sets may overlap across statuses when different people or
decisions have different support. Every comparator has the equivalent ledger in
reviewer B's final score artifact. Finite-score contrasts use paired common
support; they do not assign a favorable finite score to a failed distribution.

## Practical approximately-one-second contrasts

{table(['cohort','baseline','D minus baseline path ES','D minus baseline endpoint ES'],[(r['cohort'],r['baseline'],ci(r['path_es']),ci(r['tracked_es'])) for r in summary['contrasts'] if r['scope']=='one_second'])}

The primary averages supported.5/1/2s future horizons within root. It does not
require a person to reappear later. Less than30 roots/10 scenes makes a native
one-second conclusion explicitly inconclusive. A root with only.5s remaining
truth does not establish2s performance. Full paired age/H contrasts against both
Memory-CV laws are in`summary.cell_contrasts`; failure and censoring denominators
are in reviewerB's cell ledger. Secondary cells are descriptive development
analyses, not multiplicity-adjusted confirmation.

## Exact practical-age calibration and truth support

These rows explicitly require actual mask age[.75,1.25], rather than summing
ordinary bins that can overlap roots or admit jittered half-second masks.
Coverage uncertainty is scene clustered. A nominal90% label is not justified
as a calibrated probability when its observed coverage is materially lower.
The status ledger retains truth-censored roots alongside supported roots.

{masked_calibration_table}

Native near-one-second calibration is separately evaluated at the same ages.
Its H1coverage failure has70roots/34scenes; H2has only24roots/17scenes and is
below the predeclared30-root threshold. These cells are not pooled with masks.

{native_calibration_table}

## Paired Monte Carlo stream sensitivity

{mc_table}

These compare the SAME supported target cells. Stream variation includes
bootstrap, finite actual filtering/carrier randomness and future-world scoring,
not only endpoint sampling. The native stream difference is material; two
streams do not establish convergence. Neither stream was selected after seeing
its result. Root/mask rows repeated in the auxiliary execution are exact
duplicates and never increase independent support.

Evidence-free propagation neither deletes tracks nor resets positions to the
last center. Uncertainty includes initial Gaussian state, persistent targets,
switch histories and temporally correlated acceleration. Mode probabilities
can evolve without new evidence. A Move target can be near zero, so latent
Stop-state mass and physical speed<.1 probabilities are separately measured.
Variance is not required to grow monotonically at every marginal/time; the
physical conditional mixture law, not an artificial monotonicity constraint,
determines its evolution. Diversity and region sizes accompany proper scores.

{table(['cohort','age bin','H s','latent Stop mass','predicted physical stop frequency','actual physical stop frequency','physical stop Brier','velocity coverage90'],[(r['cohort'],r['age_bin'],r['H'],ci(r['metrics']['mode_stop_probability']),ci(r['metrics']['stop_probability']),ci(r['metrics']['observed_stop']),ci(r['metrics']['stop_brier']),ci(r['metrics']['velocity_coverage90'])) for r in summary['cells'] if r['model']=='D_StopMove'])}

Artificial thinning makes human measurements unavailable while preserving legal
pose/time/static arrivals. Native available misses can additionally apply GH
conditioning and finite future-clock renewal. ReviewerA's clock-neutral-likelihood
witness shows the latter can change a finite future carrier materially; it is
separate from the evidence-free transition law. This further limits transferring
synthetic-mask gains to camera-away or native-occlusion claims.
The mask selection does not depend on future visibility. Exact masked timestamps
are retained. The six-panel real-history example is explicitly synthetic masking,
with truth only a separate offline overlay.

![Real legal history with artificial masking](figures/01_masked_trajectory_sequence.png)
![Age/horizon scores and unique support](figures/04_age_horizon_support.png)
''')
    recovery=[r for r in summary['reobservation'] if r['model']=='D_StopMove']
    secondary_recovery=[r for r in b['reobservation'] if r['model']=='D_StopMove'
        and r['gap_bin'].startswith('root_anchor_supported_correspondence_unambiguous_')]
    update=[r for r in summary['update_gains'] if r['model']=='D_StopMove']
    belief_rows=[]
    for row in b['beliefs']:
        if row['model']!='D_StopMove':continue
        belief_rows.append((row['event'],support(row['counts'].get('all',{})),
            *(support(row['counts'].get(key,{})) for key in
              ('count_unsupported','missing_ancestry_mass','duplicate_ancestry_mass')),
            row['worst']['max_exact_coordinate_error']))
    write('REOBSERVATION.md',f'''# Genuine reobservation and same-target prediction

{general}

Each native event first observes a person, has at least one intervening frame
without that person's positive center, then receives a genuine anonymous center.
The first person/gap-bin event per root is retained. Before/after arms have the
same physical target at reappearance timestamp+H. The prior receives arrived
robot/static information with only human measurement unavailable. The posterior
receives the actual frame. Offline identities label metrics only.

Gain=PriorScore-PosteriorScore; positive favors the new observation. The original
path includes the just-measured time0, so future-only path and endpoint gains
are reported separately. Coverage change is posterior minus prior and is not
a lower-is-better gain. Reappearance eligibility/failure rows remain even after
a model has failed an earlier update.

{table(['gap bin','H s','endpoint gain','path gain','future-only path gain','correct antecedent mass','focal split mass','focal merge mass','focal duplicate mass'],[(r['gap_bin'],r['H'],ci(r['metrics']['tracked_es_gain']),ci(r['metrics']['path_es_gain']),ci(r['metrics']['future_path_es_gain']),ci(r['metrics']['association_accuracy']),ci(r['metrics']['false_split']),ci(r['metrics']['false_merge']),ci(r['metrics']['duplicate'])) for r in recovery])}

The exact practical reappearance interval is[.75,1.25]s, not a sum of ordinary
bins. Its unique root/scene support and prior/posterior status pairs are:

{table(['practical native gap','H s','endpoint gain','future-only path gain','missing preferred antecedent mass','original status support'],[(r['gap_bin'],r['H'],ci(r['metrics']['tracked_es_gain']),ci(r['metrics']['future_path_es_gain']),ci(r['metrics']['missing_antecedent_mass']),str(r['statuses'])) for r in practical_recovery])}

Any practical gain with fewer than30roots/10scenes is inconclusive. Finite
positive gain on fewer targets cannot establish native one-second recovery.
Post-minus-prior coverage changes and association entropy are descriptive:
positive coverage change alone does not prove better calibration.

{table(['gap bin','H s','coverage50 change','coverage90 change','association entropy','missing preferred mass'],[(r['gap_bin'],r['H'],*(ci(r['metrics'][metric]) for metric in ('coverage50_change','coverage90_change','association_entropy','missing_antecedent_mass'))) for r in recovery])}

These are **focal-slot diagnostic masses**, not exhaustive association error
rates. `association_accuracy` is soft posterior mass whose fixed preferred
antecedent is genuinely observed at the focal center, not top-one accuracy.
`false_split` is newborn-at-center mass; `duplicate` additionally retains the old
preferred ancestor. `false_merge` is mass where that preferred ancestor was
updated at a DIFFERENT center. Missing antecedent means that preferred ancestor
is absent. These events can overlap and do not sum to a partition.
In a finite witness, another person's track receives the focal person's center
while the focal ancestor remains missed: correct mass, split, merge, duplicate
and missing mass are all0. Another witness has split=merge=duplicate=1. Thus low
saved merge mass does not establish a low population false-merge rate. Full
posterior assignments were not archived in the scalar ledger and cannot be
reconstructed by renaming it. This is an unmet association-evaluation requirement;
see `reviewer_b_reobservation_scalar_definitions.md` and its finite checks.
Fixed ancestry is never rematched to improve future errors.
Unsupported posterior mass cannot enter a proper tracked score as though it
vanished. Full event support and status pairs are in reviewerB's ledger; these
scores concern natural reappearance, not source-ID proxy continuity.
The corrected event table fixes pre-arrival ancestry by slot and birth, retains
missing preferred ancestry separately from any duplicate, and applies the same
pre-result annotation censor as forecasting. The prior and posterior use the
SAME absolute future and pre-arrival physical ancestor; a new label cannot rescue
an unsupported prior. Near-one-second native recovery remains support limited
where unique roots/scenes are few. ReviewerB's supported-root/unambiguous events
provide a separately restricted secondary comparison. No secondary result or
synthetic mask is promoted into native reappearance superiority.

{table(['secondary supported-root gap group','H s','endpoint gain','path gain',
    'future-only path gain','association correct mass','false split mass',
    'focal merge mass','focal duplicate mass','status support'],
    [(r['gap_bin'],r['H'],*(ci(r['metrics'][metric]) for metric in
        ('tracked_es_gain','path_es_gain','future_path_es_gain','association_accuracy',
         'false_split','false_merge','duplicate')),str(r['statuses']))
     for r in secondary_recovery])}

Each verified tag records the first anchor and its actual slot birth timestamp.
A newly observed person's first assignment must agree across EVERY positive
same-origin branch; later anchors cannot overwrite it. A missing preferred slot
cannot be rescued by a duplicate, and simultaneous different first anchors are
ambiguous/unsupported. Earlier birth-minimum and weaker lifetime-selector partial
runs are superseded, with their original evidence and stop records preserved.
The unchanged-future ES10-to-0 counterexample and repaired checks are preserved.

{table(['sensor category','H s','endpoint gain','future-only path gain','coverage90 change'],[(r['category'],r['H'],ci(r['metrics']['tracked_es_gain']),ci(r['metrics']['future_path_es_gain']),ci(r['metrics']['coverage90_change'])) for r in update])}

The secondary update experiment uses the already scheduled.5/1/2s decisions and
targets observed before that arrival. Both arms use ancestry frozen BEFORE the
arrival; a newly acquired label cannot retroactively rescue its prior score.
`positive` is focal-person positive. `available_no_relevant_detection` also
records `frame_has_positive` directly from genuine raw legal center emissions,
independently of offline correspondence. It separates another person's positive
or an unmatched/ambiguous positive from an actual empty frame. The executed
raw belief ledger retains its original ID-based event label. ReviewerB's final
readout reconstructs raw-center-positive versus actually empty categories from
legal timestamped emissions without changing inference or the archived rows.
Reappearance and focal positive require unique offline matching. An empty frame remains no global
spatial emptiness certificate. Native unavailable support
may be absent; exact synthetic neutrality is not invented native evidence.

**Unresolved dense ambiguity:** four known humans with equal current position
laws and three distinct anonymous centers have24 equiprobable injective
assignments. Exact probability each ancestor is observed is.75. The beam retains
eight assignments, yielding[1,.75,.625,.625], entropylog8 rather thanlog24 and
true retention1/3. Its enumerated-retention readout is1. This concrete failure
requires BELIEF / REOBSERVATION REVISE even if native finite scores improve.
Reversing the same center set preserves the evidence receipt but shifts an
ancestor's posterior mean by .075 m. Anonymous source-ID values are ignored;
bounded sequential pruning is nevertheless not permutation invariant.
The two-person analytic ambiguity check retains alternatives within finite
resampling error, but does not prove arbitrary association calibration.

Receipt sequencing, exact-coordinate updates and persistence pass finite
regressions on retained actual beliefs. Initialization ending with unavailable
sensing still wrongly blocks a same-time genuine center; the ordinary persistent
sequence succeeds. This is another concrete failed belief-interface requirement.
Available misses use GH moment approximation; finite association
pruning/resampling and learned-context refresh prevent a universal exact-Bayes
claim. Count-support losses and model update failures remain visible in the
belief ledger rather than adding artificial people or probability floors.

{table(['actual event','all retained states','zero true-count support','missing ancestry mass','duplicate ancestry mass','max exact-center error m'],belief_rows)}

Unique all-state unions are computed separately rather than adding overlapping
event root sets. For D across{belief_union['counts']['all']['rows']}states/
2110roots/108scenes, zero true-count support occurs in23351states across
609roots/94scenes, and positive duplicate-ancestry mass in28700states across
750roots/97scenes. Missing verified-ancestry mass occurs in1377states across
72roots/48scenes. Worst expected duplicate count is12; accepted measured-slot
coordinate error remains0. These diagnostics concern represented ancestry and
offline true-count support; they do not by themselves prove a person was deleted.
They do show that coherent belief support remains a primary failure, beyond
secondary never-observed-population spatial accuracy. The original raw states,
type-normalization supplement and scene/root unions are preserved.

These are offline diagnostics on every retained actual state, including roots
without a scored target. Count/identity support defects are retained even though
never-observed spatial accuracy is not a primary objective. Missing first-anchor
or simultaneous-anchor ambiguity can additionally make a tracked score unsupported.

Native illustration selection/status is recorded in the figure manifest. A
model failure on the selected example remains recorded rather than being
replaced by an easier example.

![First metadata-eligible native reappearance](figures/02_native_reappearance.png)

The selected native example illustrates a failed recovery: after a genuine
positive, its fixed preferred lineage still has age.920s. The measured center
can be accepted by another/new slot while the old ancestry remains uncorrected.
This is not a successful reobservation claim. The separately labeled artificial
mask example shows posterior correction; neither illustration changes scores.
''')
    write('RECURSION.md','''# Non-Gaussian recursive consistency

The authoritative independent numerical evidence is reviewerA's final report,
`reviewer_a_final_checks_final_runtime.json` and the receipt-bound release
supplement. Thirty-two metadata-ordered localized development roots/29 scenes
compare direct3s,2x1.5s,3x1s and eight short/off-grid segments summing to3s,
without new human evidence and at the same absolute target.

After float64 affine-target/state arithmetic, maximum mean discrepancy is
1.066e-14, covariance4.263e-14, coupled full-state W2 upper bound2.374e-14,
selected joint characteristic-function difference1.223e-15 and relative fourth
moment difference1.114e-15. Latent clocks, counters, modes, continuous targets,
weights and ancestry agree. The coupling compares corresponding joint Gaussian
components and retained switching histories, not merely two marginal moments.
The represented non-Gaussian mixture is preserved. This is measured finite-law
consistency, not an exact unbounded-carrier inference theorem.

The old projected jump-law cumulants direct5.94868 versus.71601/.17567 after
intermediate projection remain frozen baseline evidence. The new conditional
carrier does not project them away at API partitions. Earlier submicrometre
target-rounding discrepancies and the failed tighter check are retained as
superseded evidence rather than concealed.

Independent scalar/vector conditional Gaussian references cover Stop/Move,
event-at-endpoint cases, padding and short actual intervals. Training and
inference samples also agree with an independent42-dimensional reset-moment
ODE within their recorded Monte Carlo errors. Small-interval covariance series
are checked down to1e-8s; numerical eigendecomposition can still show negligible
roundoff at such extreme scales. No arbitrary floating-point exactness claim.

Held paths, nested truncated views, endpoint states and parent RNG continuation
replay exactly on the same stored physical grid. Continuing a finite carrier
does not reseed it with a newly supplied caller seed. Retained sampled hidden
velocities also repair zero-duration-tape extension. Fresh actual-state forecasts
on different innovation grids need not be bitwise identical; their transition
law is tested separately. Unsupported imagined interior bridges and changing
world counts are explicit errors. Terminal beliefs absorb.

Arrived human evidence is a distinct operation: GH miss moment projection,
bounded associations/resampling and observed-target context refresh are declared
approximations. Those are not evidence-free API segmentation and are not excused
by passing recursion. Repeated receipts and propagate-then-observe now compose:
an earlier receipt clears on time advance, unavailable sensing creates no human
receipt, duplicate actual emissions are idempotent and conflicting ones reject.

The future-clock renewal at an AVAILABLE empty arrival is a material conditional
finite-carrier approximation. ReviewerA's genuine root-receipt witness compares
the exact same GH-updated continuous posterior with old clocks retained versus
the implemented renewed clocks. At32carriers and arrival+.5s, Stop mass changes
.125 to .28125; mean differs .08320m and a selected joint characteristic function
differs .31016. At+1s the mean difference is .11926m. Unavailable arrivals equal
pure propagation exactly. Infinite exponential memorylessness may preserve the
marginalized CTMC law; it does not make these two realized finite conditional
carriers equal. This witness neither invalidates the evidence-free semigroup nor
proves systematic native error. It qualifies conditional inference applicability.
See `reviewer_a_empty_clock_review_local_34425709_20261022.md` and source-bound
runner/results. The manually constructed no-receipt witness is secondary.

Per-depth computation and source-bound scalar/vector timings remain in reviewerA
artifacts; component/complete invocation quantiles are in PERFORMANCE. No
RECURSION REVISE blocker is demonstrated for the supported finite law within
these measured numerical limits. This is not a forecasting or systems pass.
''')
    metrics=('stop_frequency','restart_frequency','stop_duration_p50','stop_duration_p95','speed_p50','speed_p95','speed_p99','turn_p95_rad','curvature_p95','displacement_p95')
    prows=[r for r in physics['summary'] if r['metric'] in metrics]
    oldstop=next(r for r in oldphysics['summary'] if r['metric']=='stop_frequency')
    write('PHYSICAL_VALIDATION.md',f'''# Native matched-interval motion physics

The old physical mismatch is reproduced without modifying the43187f24 model:
predicted stop frequency{ci(oldstop['predicted'])}, observed
{ci(oldstop['actual'])}, on matched nonempty native support. The independently
preserved exact replay also checks the original records and matched summaries.
Those all-population baseline diagnostics are distinct from the root-visible
physical-target comparison below.

The frozen previous-release measurements and the analytic position-noise
ceiling motivated the transition extension before fitting. Independent exact
replay of the old diagnostic completed after the new fit. This is a chronology
deviation from reproducing the diagnostic first; the reproduced result did not
select or tune the already frozen candidate.

For old independent position diffusion rateq=.0025, even a stationary centered
mean has maximum probability`1-exp(-.1**2*.048/(2*.0025))=.09153598` of native
finite-difference speed<.1m/s at.048s. A noncentral shift cannot increase centered
Gaussian-ball mass, so independent OU/jump motion cannot remove this ceiling.
The old observed frequency about.3811 cannot be fitted by its neural target
alone. The new integrated-acceleration/Stop law removes that position-noise
floor and retains small, nonzero stopped velocity uncertainty.

Main comparison: all supported root-positive development people; fixed physical
ancestry, identical ACTUAL native timestamps through available<=3s,32 worlds,
two independently root-keyed paired streams. No per-time rematching or nominal
clock substitution. People/worlds/streams average within root, then scenes.
Native actual dt can differ from.048s; the fixed-.048 proof is illustrative,
while measured diagnostics use actual dt.
Speed units are m/s, displacement is m, turn angle is rad, curvature is rad/m,
and duration is s. Frequencies are interval or interval-pair proportions.
The reported P50/P95/P99 values are equal-scene averages of within-root
quantiles, not pooled-population quantiles.

{table(['model','native metric','prediction','actual matched target','prediction minus actual'],[(r['model'],r['metric'],ci(r['predicted']),ci(r['actual']),ci(r['difference'])) for r in prows])}

The stop-frequency deficit is now{ci(d_stop['difference'])}; its interval
includes zero, but that does not establish general physical calibration.
Restart remains low by{ci(d_restart['difference'])}. Speed P95 is high by
{ci(d_speed['difference'])}m/s and turn-angle P95 by
{ci(d_turn['difference'])}rad. Completed stop-duration P95 is high by
{ci(d_duration['difference'])}s on its412root/85scene matched nonempty-run
subset. The latter subset has different support from the all-root frequency
comparison. These failures prevent a physical-readiness claim despite the
material stop-frequency improvement. The model was not refitted to these
development diagnostics.

`restart_frequency` is the fraction of adjacent native interval pairs with
previous speed<.1 and next speed>.2, not a conditional restart probability or
hazard. Stop-entry frequency, speed/turn CDFs, displacement and curvature remain
in the raw artifact. Completed stop-duration quantiles exclude left/right-
censored runs and are equal-root/scene averages of run quantiles, not a pooled
survival estimate. Censored-run counts are preserved. Short native prefixes
therefore limit duration calibration.

Latent Stop/Move probability differs from physical speed-threshold probability:
a Move target can be near zero. Neither a matching stop frequency nor a finite
mode label establishes correct trajectories. FORECASTING and the unseen-age
matrices jointly judge that tradeoff. No speed clipping, deterministic stopped
position or "everyone stops" replacement was applied. Turning is continuous
trajectory geometry, not a route-intention head.

Failures:{len(physics['failures'])} raw model/stream/root diagnostics;
original independent supports and failure causes are retained. These results
are development calibration evidence and do not establish full-population SOTA.
''')
    performance(speed,systems,startup)
    write('OBSERVATION_MODEL.md',f'''# Supported observation law and joint conditional consistency

The inherited persistent-regime smooth range/sector law is an optical
approximation. It has no full map, native mesh visibility or universal empty
region certificate. Human paths are exogenous to the supplied fixed query.
One held32-world future includes both nominal planned and actual timestamps;
the latter/recorded future poses are OFFLINE optical diagnostics only.
Two independent root streams are averaged within root/scene.

{table(['evaluation','Brier','finite NLL','predicted frequency','observed frequency','first-detection finite NLL','zero-support rows/roots'],[(r['kind'],ci(r['metrics']['brier']),ci(r['metrics']['nll']),ci(r['metrics']['predicted']),ci(r['metrics']['observed']),ci(r['metrics']['first_detection_nll']),str((r['zero_support_rows'],r['zero_support_roots']))) for r in optics['summary']])}

Any genuine zero-probability outcome makes unconditional NLL infinite; finite
summaries above are conditional where zero support is reported. No probability
floor is introduced. Complete planned prefixes require all evaluated sensor
endpoints and at most62nominal steps (2.976s). Unavailable/censored followup is
not an empty measurement. Actual-minus-nominal clock quantiles are
`{optics['native_minus_nominal_clock_quantiles']}`; planned scores are qualified
by those clock/follower deviations and informative exposure censoring.

Normalization does not establish calibrated native event probabilities.
ReviewerB independently reproduces the15optical metrics and finds complete-
prefix predicted-minus-observed frequency+.073873 [.050294,.095234] on1495roots/
108scenes. The planned endpoint probability[.4,.5)bin has gap+.317457
[.284476,.344800] on591roots/93scenes. The1230row failure ledger comprises
1222prefix-censored/unavailable records (611roots) and8unsupported known-space
query records (4roots), with no exceptions or realized zero-support outcomes in
this cohort. Those missing exposures remain failures/censors, not invented
empty observations or a calibrated full optical model.

First-detection masses plus none normalize with maximum residual
{optics['max_normalization_error']:.3g}; none-successor weights match their held
likelihood reweighting within{optics['max_conditional_weight_error']:.3g}.
Independent finite optical tower tests also reconstruct the original joint
measure. This proves consistency of the supported finite event law, not native
optical calibration. Region selection is externally supplied and never chosen
by navigation utility. Exact hypothetical centers/interior refinement remain
explicitly unsupported; real observations update actual continuous state.
''')
    print(json.dumps(dict(reports='rendered',decision=labels,selected_epoch=selected['epoch'])),flush=True)


def performance(speed=None,systems=None,startup=None):
    speed=readj(OUT/'final_speed_manifest.json') if speed is None else speed
    systems=readj(OUT/speed['benchmark']/'benchmark.json') if systems is None else systems
    startup=readj(OUT/(speed['benchmark']+'_source_capture')/'first_use.json') if startup is None else startup
    def group(category,cache):return systems[category][repr((cache,16,'primary'))]['metrics']['combined_ms']
    warm=group('representative_first60_quantiles','warm');cold=group('representative_first60_quantiles','dynamic_cold')
    dense_warm=group('dense_quantiles','warm');dense_cold=group('dense_quantiles','dynamic_cold')
    gate_text='passes' if max(warm['p95_ms'],cold['p95_ms'])<=100 else 'fails'
    preferred_text='passes' if max(warm['p95_ms'],cold['p95_ms'])<=65 else 'fails'
    reassociation_path=OUT/'reviewer_c_reassociation_speed008_v1/result.json'
    reassociation=readj(reassociation_path) if reassociation_path.exists() else None
    reassociation_rows=[]
    if reassociation:
        if not reassociation['source_stable'] or reassociation['model_version']!=systems['model_version']:raise ValueError('unbound reassociation runtime')
        for record in reassociation['records']:
            for cache,values in record['quantiles'].items():
                reassociation_rows.append((record['people'],cache,values['complete'],
                    *('%.2f'%values[k] for k in ('p50_ms','p95_ms','p99_ms'))))
    timing=[]
    for category,key in (('representative first60','representative_first60_quantiles'),('dense stress','dense_quantiles'),('world sensitivity','world_sensitivity_quantiles')):
        for setting,group in systems[key].items():
            for component,values in group['metrics'].items():
                timing.append((category,setting,component,values['n'],
                    '%.3f'%values['p50_ms'],'%.3f'%values['p95_ms'],'%.3f'%values['p99_ms']))
    write('PERFORMANCE.md',f'''# Measured Layer 2 computation

**The representative100ms gate {gate_text}; the preferred65ms gate {preferred_text}.**
Complete-call P50/P95/P99 is
{warm['p50_ms']:.2f}/{warm['p95_ms']:.2f}/{warm['p99_ms']:.2f}ms warm and
{cold['p50_ms']:.2f}/{cold['p95_ms']:.2f}/{cold['p99_ms']:.2f}ms dynamically cold.
Dense12-root P95 is{dense_warm['p95_ms']:.2f}/{dense_cold['p95_ms']:.2f}ms.
The dense stress group exceeds100ms; the representative result is not a
universal per-call guarantee. Representative P99 also exceeds100ms.
The original frozen new-model implementation on this SAME cohort measured
247.23/241.47ms representative P95 and533.65/541.75ms dense P95.
The optimization was requested after scientific evaluation was frozen; it
changes execution, with separately guarded implementation ownership and
backward-compatible stochastic RNG identity, rather than refitting weights.
The old release measured105.6/115.2ms P95 on its declared cohort; the new
carrier/workload/cohort differs, so this is a gate comparison, not a causal
same-cohort speed ratio.

![Fixed-cohort inference latency](figures/05_latency.png)

Authoritative optimized benchmark`{speed['benchmark']}`: sources/checkpoint/cache
unchanged,83 attempted roots,82 supported overall; first60 metadata roots have
59 supported. Stress roots are deliberately oversampled and are reported
separately, not silently pooled as the representative workload. Across702
complete timed calls there are seven explicit query-certificate refusals on
the same UNKNOWN-touching root, not state/update/successor failures.

Combined timer encloses actual final-root observation after legal-prefix replay,
one shared3s/16world forecast with64stored timestamps, one externally supplied
canonical fixed query optical forecast and its supported none successor.
Actual32-component support, exact input checks and full conditional work are
retained. Prefix replay/import/load are separately timed. Dynamic cold clears
route/footprint/kernel caches in an initialized process; it is not fresh process
startup. Component quantiles do not add to invocation P95.

Unprimed synthetic complete first use is{startup['combined_ms']:.2f}ms, including
{startup['observe_ms']:.2f}ms observation work and{startup['forecast_ms']:.2f}ms
forecast work, with first-use Numba compilation where needed. Framework/package
imports add{startup['framework_import_ms']:.2f}/{startup['model_package_import_ms']:.2f}ms,
and checkpoint loading adds{startup['model_load_ms']:.2f}ms. Synthetic history
bootstrap adds{startup['bootstrap_ms']:.2f}ms, including first seed-kernel JIT.
Bootstrap through successor totals{startup['bootstrap_through_successor_ms']:.2f}ms;
framework import through successor totals{startup['import_through_successor_ms']:.2f}ms.
These startup costs
are explicitly outside the initialized-process latency gate. A deployment must
complete kernel initialization before treating the loaded-process measurements
as applicable; fresh-process100ms operation is not demonstrated. Numba runs one
thread with fastmath disabled and cache metadata/signatures are archived.

{table(['cohort','cache/world setting','operation','calls','P50ms','P95ms','P99ms'],timing)}

The optimized path batches exact Gaussian conditioning and continuous-state
propagation, materializes only selected assignment states, reuses identical
within-call GH and observed-cell inputs, and compiles the stable existing beam
and full stochastic path loops. Keyed PCG64 construction, categorical CDF and
the retained continuation RNG are checked against NumPy's reference streams.
Fingerprint reuse applies only to fully validated immutable byte-backed state;
replacement invalidates it. Ownership and invalid-domain checks remain active.
Actual updates match the frozen source bitwise on source-bound legal-history
comparisons; coupled sampled paths and conditional probabilities agree within
the declared floating tolerance with exact latent clocks and RNG continuation.
ReviewerA also preserves a hard-boundary projection counterexample: a sample
within1.24e-14m of a boundary can flip finite membership under roundoff. No
universal bitwise projection-equivalence claim is made. Source-changing
preliminary timing is preserved but does not establish final timing. No
difficult people, worlds, timestamps, updates, likelihood support or conditional
work were dropped. Shared futures remain reusable without regenerating dynamics.

The pre-speed synthetic one-second reassociation update-only P95 was35.4ms for
one person and227.9ms for six (10 repeats/case); it is historical reference,
not a final optimized measurement. The final008 artifact is
`reviewer_c_reassociation_speed008_v1/result.json`. Its SAME fixed fixtures
have a genuine0.1s positive, unavailable-human-sensor propagation to1.1s, then
genuine anonymous centers at1.1s. All40updates retain32components and every
one/six-person track, exact positions/last-seen times and normalized mass.
Source/manifest/old-v2 identities remain stable; invalid timestamp/covariance/
slot inputs are still rejected. Gap propagation, bootstrap and JIT are recorded
separately. This is update-only synthetic latency, not native association
accuracy, independent physical-root support or complete-call latency.

{table(['people','cache','completed updates','P50ms','P95ms','P99ms'],reassociation_rows)}

Track readout, pure Stop/Move propagation,
projection, warm/cold dense cases,8/16/32/64world sensitivity and300-arrival
synthetic persistence traces are included in the raw system artifact/review.
Weak future registries release; immutable ownership, JSON dtype/shape restore,
receipt sequencing, stale-source and invalid-domain checks pass. Empirical RSS
and finite repeated-call traces do not prove unlimited-session memory bounds.

One numerical CPU thread/job, nice19 and idle I/O were used; final benchmark
scheduler/affinity are recorded verbatim. Fitting/evaluation jobs also yield CPU
through their recorded idle scheduler controls. Read-only resource
logs preserve collector worker counts, throughput, status fluctuations, restart
and configuration identities. Production was not stopped, restarted,
reconfigured, schema-changed or merged into. No causal no-slowdown claim follows
from shared-host throughput. Full sources and hashes are in reviewerC's manifest.
''')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--performance-only',action='store_true')
    a=p.parse_args();performance() if a.performance_only else run()
