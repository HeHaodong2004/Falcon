"""Summarize the completed frozen panel; never fit, select a seed or evaluate again."""
import argparse
import json
from pathlib import Path

from eventbeliefwm.data import read, sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.performance_v2.compare_completed import compare, paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign.resolve()
    output = root / 'confirmation_analysis'
    if output.exists():
        raise FileExistsError('Completed summaries are immutable; use their receipts')
    panel_path = root / 'confirmation_panel/result.json'
    panel = read(panel_path)
    summary = dict(status='frozen_confirmation_analysis_complete', confirmation=True,
        panel_sha256=sha256(panel_path), source_sha256=sha256(__file__), banks={})
    for bank in ('natural_reference', 'unseen_evaluation'):
        receipts = {name: read(value[bank]['path']) for name, value in panel['models'].items()}
        rows = {}
        for name, receipt in receipts.items():
            if sha256(panel['models'][name][bank]['path']) != panel['models'][name][bank]['sha256']:
                raise ValueError('Panel receipt changed')
            rows[name] = dict(checkpoint_sha256=receipt['checkpoint_sha256'],
                horizons={h: dict(trajectory_roots=scores['trajectory_roots'],
                    trajectory_scenes=len({r['scene'] for r in scores['records'] if 'energy' in r}),
                    trajectory=scores['metrics'], occupancy=receipt['extras'][h]['occupancy'],
                    occupancy_support=receipt['extras'][h]['occupancy_metric_roots'])
                    for h, scores in receipt['scores'].items()},
                joint_uncertainty={k:v for k,v in receipt['joint_uncertainty'].items() if k!='records'})
        comparisons = {}
        for reference in ('incumbent_primary', 'incumbent_replicate', 'cv', 'cv_geometry'):
            names = (['candidate_replicate'] if reference == 'incumbent_replicate' else
                ['candidate_primary', 'candidate_replicate', 'stopped_alternative', 'horizon_alternative',
                 'full_self_primary', 'full_self_replicate'])
            result = compare(panel['models'][reference][bank]['path'],
                {name:panel['models'][name][bank]['path'] for name in names})
            result.update(confirmation=True, panel_sha256=sha256(panel_path), bank=bank)
            for name in names:
                left=receipts[name]['joint_uncertainty']; right=receipts[reference]['joint_uncertainty']
                if left['draws']!=right['draws'] or left['nominal_finite_rank_coverage']!=right['nominal_finite_rank_coverage']:
                    raise ValueError('Joint rank protocols differ')
                result['comparisons'][name]['joint_coverage90'] = paired(left['records'],right['records'],'joint_coverage90')
            comparisons[reference] = result
        summary['banks'][bank] = dict(models=rows, comparisons=comparisons)
    output.mkdir()
    save_json(output/'result.json', summary)
    lines=['# Frozen new confirmation panel', '',
        'All rows use the same frozen support and sample counts. Lower ES, trajectory NLL, Brier and occupancy log score are better; higher AP is better. Five seconds is extrapolation.', '']
    for bank,value in summary['banks'].items():
        lines += ['## '+bank, '', '| Model | ES1 | ES2 | ES3 | NLL3 | ADE3 | FDE3 | AP3 | Brier3 | Occupancy log3 | Joint coverage3 | ES5 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name,row in value['models'].items():
            h=row['horizons']; t=h['3.0']['trajectory']; o=h['3.0']['occupancy']
            numbers=[h['1.0']['trajectory']['energy'],h['2.0']['trajectory']['energy'],t['energy'],t['nll'],t['mean_ade'],t['mean_fde'],o['average_precision'],o['brier'],o['nll'],row['joint_uncertainty']['metrics']['joint_coverage90'],h['5.0']['trajectory']['energy']]
            lines += ['| '+name+' | '+' | '.join(f'{n:.6f}' for n in numbers)+' |']
        lines += ['', '| Matched seed | Metric | Candidate minus v1 | 95% scene interval | Roots / scenes |', '|---|---|---:|---|---:|']
        for candidate,reference in [('candidate_primary','incumbent_primary'),('candidate_replicate','incumbent_replicate')]:
            result=value['comparisons'][reference]['comparisons'][candidate]
            for label,metric in [('ES1',result['1.0']['trajectory']['energy']),('ES2',result['2.0']['trajectory']['energy']),('ES3',result['3.0']['trajectory']['energy']),('NLL3',result['3.0']['trajectory']['nll']),('AP3',result['3.0']['occupancy']['average_precision']),('Brier3',result['3.0']['occupancy']['brier']),('Occupancy log3',result['3.0']['occupancy']['nll']),('Joint coverage3',result['joint_coverage90'])]:
                d=metric['delta'];lines += [f"| {candidate} | {label} | {d['mean']:+.6f} | [{d['low']:+.6f}, {d['high']:+.6f}] | {d['roots']} / {d['scenes']} |"]
        lines += ['']
    (output/'TABLES.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(output),panel_sha256=summary['panel_sha256'])))


if __name__ == '__main__':
    main()
