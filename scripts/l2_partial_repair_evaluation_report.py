"""Render complete saved population/conditioning evidence, without fitting or payload access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eventwm_graph.l2_partial.repair_evaluation import scene_ci


def ci(value):
    if value.get('status')=='UNBOUNDED_LOG_SCORE_PRESENT':
        return 'Unbounded ('+str(value['unbounded_roots'])+' roots)'
    if value.get('mean') is None:return 'Unavailable'
    return f"{value['mean']:.4f} [{value['lo']:.4f}, {value['hi']:.4f}]"


def run(directory,output):
    directory=Path(directory);output=Path(output)
    evaluation=json.loads((directory/'population_evaluation.json').read_text())
    comparisons=json.loads((directory/'paired_comparisons.json').read_text())
    if evaluation['roots']!=632 or evaluation['scenes']!=106 or evaluation['smoke_test_only']:
        raise ValueError('Fixed full development cohort required')
    text=['# Full development population, forecasting and conditioning',
        '', 'All632 independent roots/106 scenes are evaluated with all six frozen variants. '
        'Every sibling query stays with its root. Tables show equal-scene means and1,000-replicate scene-bootstrap95% intervals, seed73. '
        'These are development descriptions, not untouched confirmation. The primary model was frozen before fresh payload access.',
        '', '## Population accounting and uncertainty', '',
        'Count error is conditional on represented mass; residual mass has no point count law. '
        'Count NLL is bounded by assigning the unspecified mass to or away from the true count. '
        'Any zero-probability true count makes the corresponding global upper/conditional log score unbounded; finite-only means are never substituted.', '',
        '| Variant | Count MAE | Count NLL lower | Count NLL upper | Truth excluded roots | Unresolved mass |',
        '|---|---|---|---|---:|---|']
    for name,summary in evaluation['summary'].items():
        p=summary['population']
        text.append('|'+name+'|'+ci(p['count_absolute_error'])+'|'+ci(p['count_nll_lower'])+'|'+ci(p['count_nll_upper'])+'|'+str(p['truth_excluded_roots'])+'|'+ci(p['unresolved_mass'])+'|')
    text+=['', 'Localized GOSPA uses p=2, cutoff2 m, alpha2 against ALL true people. Unlocalized people contribute misses; source identity is not used online. '
        'The sampled-population score additionally includes private hidden positions, and is comparable as a complete population score only where the complete-mass column supports it. '
        'Legacy has no spatial law for remaining hidden counts. GOSPA is a set-error metric, not a proper probability score.', '',
        '| Variant | Localized GOSPA m | False localized count | Missed localized count | Sampled-population GOSPA m | Complete sampled-population mass |',
        '|---|---|---|---|---|---|']
    for name,summary in evaluation['summary'].items():
        p=summary['population'];text.append('|'+name+'|'+ '|'.join(ci(p[k]) for k in ['gospa','false_localized','missed_localized','sampled_population_gospa','sampled_population_complete_mass'])+'|')
    text+=['', '| Variant | False-track fraction | Missed-human fraction | Undefined false denominator roots | Undefined missed denominator roots |',
           '|---|---|---|---:|---:|']
    for name,rates in comparisons['localized_rates'].items():
        text.append('|'+name+'|'+ci(rates['false_track_fraction'])+'|'+ci(rates['missed_human_fraction'])+'|'+str(rates['zero_predicted_denominator_roots'])+'|'+str(rates['zero_truth_denominator_roots'])+'|')
    text+=['', 'Rates average per-root expected false/localized and missed/true ratios. Zero denominators are undefined rather than scored as zero; all those roots remain in count/GOSPA tables.',
           '', '## Fixed-ancestry forecasts over physical horizon', '',
           'Exact root-visible physical identities are aligned once on unsampled means; aged root alignment is a separately labeled offline oracle. '
           'No forecast world or future target is rematched. Energy Score and90% sampled-region coverage condition on represented root-target ancestry. '
           'They do not score the entire random-population law. Full-population losses and represented mass remain adjacent. '
           'Longer horizons may naturally have larger error; score support changes with recorded future availability.', '',
           '| Variant | Horizon s | Energy Score m | Scored roots/scenes | Coverage90 | Represented target mass | All-target GOSPA m |',
           '|---|---:|---|---|---|---|---|']
    for name,summary in evaluation['summary'].items():
        for horizon,f in summary['forecast'].items():
            score=f['ancestry_energy_score']
            scored=[r for r in evaluation['forecast_rows'] if r['variant']==name and r['horizon']==float(horizon)
                    and r.get('ancestry_energy_score') is not None]
            represented=scene_ci(scored,'ancestry_scored_mass_mean')
            text.append('|'+name+'|'+horizon+'|'+ci(score)+'|'+str(score['roots'])+'/'+str(score['scenes'])+'|'+ci(f['ancestry_coverage90'])+'|'+ci(represented)+'|'+ci(f['gospa'])+'|')
    text+=['', 'Paired comparisons use the intersection of roots with both scores and report excluded roots explicitly. '
        '`full_kinematic` retains the same repaired maneuver/observation machinery, so its comparison isolates learned targets within that machinery, '
        'rather than establishing superiority over all strong CV/memory baselines. The old release\'s future-rematched metrics are not directly comparable.', '',
        '| Full minus reference | Horizon s | Energy Score difference m | Common scored roots/scenes | Excluded roots |',
        '|---|---:|---|---|---:|']
    for row in comparisons['forecast_pairs']:
        if row['metric']!='ancestry_energy_score' or row['left']!='full':continue
        q=row['left_minus_right'];text.append('|'+row['right']+'|'+str(row['horizon'])+'|'+ci(q)+'|'+str(q['roots'])+'/'+str(q['scenes'])+'|'+str(row['excluded_roots_missing_either_score'])+'|')
    text+=['', '## Same-target observation conditioning', '',
        'Both arms propagate the same root history to the tenth actual native endpoint (about0.48 s), receive the same execution/survival evidence, '
        'and predict the identical absolute root+2 s primary or root+1 s secondary target. Intermediate sensor reports are withheld in both arms; '
        'the posterior alone receives the arrived tenth observation. Survival uses the same declared analytic-component Gaussian-moment projection, not exact truncation. '
        'Negative delta means the posterior improves Energy Score. The+0.020 m non-inferiority margin, one-sided95% upper bound and minimum100 roots/30 scenes were declared before these results. '
        'The secondary target cannot replace the primary or be pooled with it.', '',
        '| Full model category | Absolute target s | Posterior−prior ES m | Scored roots/scenes | Eligible/unsupported pairs | NI status |',
        '|---|---:|---|---|---|---|']
    categories=['overall','positive','any_detection','reappearance_physical','reappearance_proxy','aged_tracks','dense','association_ambiguous','occluded_or_ambiguous']
    for horizon in ['2.0','1.0']:
        for category in categories:
            q=evaluation['summary']['full']['conditioning_by_horizon'][horizon][category]
            text.append('|'+category+'|'+horizon+'|'+ci(q)+'|'+str(q['roots'])+'/'+str(q['scenes'])+'|'+str(q['eligible_pairs'])+'/'+str(q['unsupported_pairs'])+'|'+q['status']+'|')
    text+=['', '| Variant | Target s | Overall posterior−prior ES m | Scored roots/scenes | Sampled ALL-population GOSPA delta m |',
           '|---|---:|---|---|---|']
    for name,summary in evaluation['summary'].items():
        for horizon in ['2.0','1.0']:
            q=summary['conditioning_by_horizon'][horizon]['overall']
            text.append('|'+name+'|'+horizon+'|'+ci(q)+'|'+str(q['roots'])+'/'+str(q['scenes'])+'|'+ci(q['sampled_population_gospa_delta'])+'|')
    text+=['', 'Physical reappearance uses offline exact identity matches over the root and arrived prefix; it does not equate source track IDs with physical identity. '
        'Pre-root invisible sightings cannot be classified by this diagnostic. A positive category means a root target is actually detected, and dense means true population>=4 for offline stratification only. '
        'Occluded/ambiguous here means no positive detections; it is not a certified empty region. All295 unsupported1s and131 unsupported2s full-model pairs have zero root-visible targets, rather than lost target ancestry. '
        'Forecast represented mass in the table is averaged only over scored roots; the raw unfiltered summary also contains zero-target roots. '
        'There are no real human-empty certificates, so empirical certified-negative benefit is unavailable.', '',
        'Mechanism attribution is limited. `association_only` changes maneuver-aware association. Its comparison to `full` jointly changes set emissions, persistent spatial hidden state, first-localization velocity and history transition; '
        'it is not an optics-only causal experiment. The frozen no-maneuver, kinematic and unfitted-optics ablations separate named mechanisms, but do not form a complete factorial identification. '
        'No event, uncertainty or optical parameter is selected after fresh access.', '',
        '## Population strata retained as regressions', '',
        '| Full model stratum | Roots/scenes | Count MAE | Localized GOSPA m | Unresolved mass |',
        '|---|---|---|---|---|']
    for label,s in evaluation['summary']['full']['population_strata'].items():
        q=s['count_absolute_error'];text.append('|'+label+'|'+str(q['roots'])+'/'+str(q['scenes'])+'|'+ci(q)+'|'+ci(s['gospa'])+'|'+ci(s['unresolved_mass'])+'|')
    text+=['', 'See ROLLOUT_AND_PROBABILITY.md for the post-freeze conditional joint-time trajectory supplement and physical-speed failures; '
        'RISK_AND_OBSERVATIONS.md for native event, joint observation calibration and fixed-rule decision evidence; '
        'FRESH_EVALUATION.md for the genuinely disjoint but underpowered confirmation. '
        'Bootstrap intervals condition on the fixed forecast/survival/conditioning seeds and do not separately quantify Monte Carlo seed uncertainty.', '',
        'Report source SHA256: `'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+'`. '
        'Input population SHA256: `'+hashlib.sha256((directory/'population_evaluation.json').read_bytes()).hexdigest()+'`.']
    with output.open('x') as stream:stream.write('\n'.join(text)+'\n')
    print(str(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory');parser.add_argument('output')
    args=parser.parse_args();run(args.directory,args.output)
