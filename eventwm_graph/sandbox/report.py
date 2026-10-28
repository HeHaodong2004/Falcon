"""Markdown tables for docs/sandbox/REPORT.md from the benchmark / validation summaries (no hand-copied numbers).

    python scripts/sandbox_run.py eventwm_graph.sandbox.report output/bench output/validate
"""
import json
from pathlib import Path
import sys


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def fmt(v, digits=0):
    if v is None:
        return '-'
    return ('%.'+str(digits)+'f') % v


def latency_table(bench, names):
    rows = ['| run | decisions | sandbox P50 / mean / P95 (ms) | Stage 1 P50 | imagination P50 / P95 | Stage 2 P50 / P95 | '
            'risk P50 | first-decision sandbox P50 | end-to-end decision P50 / mean / P95 (ms) |',
            '|---|---|---|---|---|---|---|---|---|']
    for name in names:
        s = _load(Path(bench)/name/'summary.json')
        if s is None:
            continue
        w, f, e = s['warm'], s['first'], s['end_to_end_decision_ms']
        rows.append('| %s | %d | %s / %s / %s | %s | %s / %s | %s / %s | %s | %s | %s / %s / %s |' % (
            name, s['decisions'], fmt(w['sandbox_ms']['p50']), fmt(w['sandbox_ms']['mean']), fmt(w['sandbox_ms']['p95']),
            fmt(w['stage1_ms']['p50'], 1), fmt(w['imagine_ms']['p50']), fmt(w['imagine_ms']['p95']),
            fmt(w['stage2_ms']['p50'], 1), fmt(w['stage2_ms']['p95'], 1), fmt(w['interaction_ms']['p50'], 1),
            fmt(f['sandbox_ms']['p50']) if f.get('sandbox_ms') else '-', fmt(e['p50']), fmt(e['mean']), fmt(e['p95'])))
    return '\n'.join(rows)


def components_table(bench, names):
    keys = [('env_step_ms', 'simulator'), ('sensor_ms', 'legal sensor'), ('range_ms', 'range frames'),
            ('stream_ms', 'stream update'), ('decision_state_ms', 'decision state'), ('generator_ms', 'generator'),
            ('sandbox_ms', 'sandbox')]
    rows = ['| run | '+' | '.join(k[1] for k in keys)+' | sandbox share |', '|---|'+'---|'*len(keys)+'---|']
    for name in names:
        s = _load(Path(bench)/name/'summary.json')
        if s is None:
            continue
        w = s['warm']
        tot = sum(w[k]['mean'] for k, _ in keys if w.get(k))
        rows.append('| %s | ' % name+' | '.join(fmt(w[k]['mean']) for k, _ in keys)+' | %.0f%% |' % (100*w['sandbox_ms']['mean']/tot))
    return '\n'.join(rows)+'\n\n(mean ms per decision interval of 16 steps; warm decisions)'


def frames_table(bench, name):
    s = _load(Path(bench)/name/'summary.json')
    if s is None:
        return ''
    rows = ['| frames in history | decisions | Stage 2 P50 (ms) | decision state P50 (ms) | sandbox P50 (ms) |', '|---|---|---|---|---|']
    for b, v in s['by_frames'].items():
        rows.append('| %s | %d | %s | %s | %s |' % (b, v['n'], fmt(v['stage2_p50'], 1), fmt(v['state_p50'], 1), fmt(v['sandbox_p50'])))
    return '\n'.join(rows)


def throughput_table(bench):
    rows = ['| run | workers | work | decisions/s | per worker (decisions) | window (s) | load |', '|---|---|---|---|---|---|---|']
    for p in sorted(Path(bench).glob('tp_*/summary.json')):
        s = _load(p)
        rows.append('| %s | %d | %s | %.2f | %s | %.0f | %.0f |' % (p.parent.name, s['workers'], s['work'], s['decisions_per_s'],
                                                               s['per_worker'], s['window_s'], s['load'][0]))
    return '\n'.join(rows)


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else 'output/bench'
    val = sys.argv[2] if len(sys.argv) > 2 else 'output/validate'
    names = ['k10_branch', 'k6_branch', 'k10_branch_dwa_inertia', 'k10_real']
    print('### Latency\n'); print(latency_table(bench, names)); print()
    print('### Components\n'); print(components_table(bench, names)); print()
    print('### History length (k10_branch)\n'); print(frames_table(bench, 'k10_branch')); print()
    print('### Throughput\n'); print(throughput_table(bench)); print()
    r = _load(Path(val)/'ranges/summary.json')
    if r:
        print('### Fast range sensor vs native\n'); print('```'); print(json.dumps({k: v for k, v in r.items() if k != 'per_episode'}, indent=1)); print('```')
    f = _load(Path(val)/'fidelity/summary.json')
    if f:
        print('### Fidelity\n'); print('```'); print(json.dumps(f, indent=1)); print('```')


if __name__ == '__main__':
    main()
