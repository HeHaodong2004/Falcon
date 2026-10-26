"""Markdown tables from evaluation summaries and benchmark receipts."""
import json
import sys


def latency(path):
    d = json.load(open(path))
    rows = {(s['cohort'], s['cache'], s['k']): s for s in d['summary']}
    out = ['| cohort / cache | K=1 P50 / P95 | K=4 P50 / P95 | K=6 P50 / P95 |', '| --- | ---: | ---: | ---: |']
    for cohort in ('dense', 'representative'):
        for cache in ('cold', 'map_warm_new_queries', 'warm'):
            cells = []
            for k in (1, 4, 6):
                s = rows.get((cohort, cache, k))
                cells.append('%.3f / %.3f' % (s['p50_ms'], s['p95_ms']) if s else '—')
            out.append('| %s, %s | %s |' % (cohort, cache, ' | '.join(cells)))
    out.append('')
    out.append('startup: %s; load average before/after: %s / %s; device %s (%s)' % (
        json.dumps(d['startup']), d.get('load_average_before'), d.get('load_average_after'), d['device'], d['hardware']))
    return '\n'.join(out)


def accuracy(path):
    d = json.load(open(path))
    h = d['hierarchical']; p = d['pooled']
    lines = ['| metric | previous baseline | this model |', '| --- | ---: | ---: |']
    for name in ('label_aware_delta_iou', 'revelation_iou', 'revelation_precision', 'revelation_recall'):
        lines.append('| hierarchical %s | %.6f | %.6f |' % (name, h['baseline'][name], h['candidate'][name]))
    for name in ('free', 'blocked'):
        lines.append('| hierarchical %s precision / recall | %.4f / %.4f | %.4f / %.4f |' % (
            name, h['baseline'][name]['precision'], h['baseline'][name]['recall'], h['candidate'][name]['precision'], h['candidate'][name]['recall']))
    lines.append('| pooled label_aware_delta_iou | %.6f | %.6f |' % (p['baseline']['label_aware_delta_iou'], p['candidate']['label_aware_delta_iou']))
    lines.append('')
    lines.append('Roots %d, scenes %d, windows %d; zero-new windows %s; out-of-domain positives %d; bootstrap 95%% candidate %s, paired delta vs baseline %s' % (
        d['roots'], d['scenes'], d['windows'], json.dumps(d['zero_new']), d['out_of_domain_positives'],
        [round(v, 4) for v in d['bootstrap_95']['candidate']], [round(v, 4) for v in d['bootstrap_95']['paired_delta']]))
    lines.append('')
    lines.append('Pooled confusion (rows true U/F/B, cols predicted): %s' % json.dumps(d['pooled_confusion'][1]))
    lines.append('')
    lines.append('| stratum | windows | label-aware IoU (pooled) |')
    lines.append('| --- | ---: | ---: |')
    for k, v in d['pooled_strata'].items():
        lines.append('| %s | %d | %.4f |' % (k, v['windows'], v['candidate']['label_aware_delta_iou']))
    return '\n'.join(lines)


if __name__ == '__main__':
    kind, path = sys.argv[1], sys.argv[2]
    print(latency(path) if kind == 'latency' else accuracy(path))
