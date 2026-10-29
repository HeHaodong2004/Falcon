"""Active-perception statistics of a run: how often the policy looks (scan / look queries) and what looking is worth
according to the on-policy counterfactual labels.

    python scripts/l3_run.py eventwm_graph.l3.scanstats --run output/runs/R8 [--since-version 0]
"""
import argparse
import json
from pathlib import Path

import numpy as np

LOOK = ('scan_', 'look_', 'glance_')


def is_look(label):
    return str(label).startswith(LOOK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--spool', default='spool')
    ap.add_argument('--since-version', type=int, default=0)
    a = ap.parse_args()
    from .actor import read_episode
    run = Path(a.run)
    by = {}
    for p in sorted(Path(run, a.spool).glob('*.ep')):
        try:
            d = read_episode(p)
        except Exception:
            continue
        if int(d.get('version', 0) or 0) < a.since_version:
            continue
        rows = d['rows']
        c = by.setdefault(d['config'], dict(episodes=0, success=0, decisions=0, offered=0, chosen=0, ep_look=0,
                                            ep_look_success=0))
        c['episodes'] += 1
        ok = d['metrics']['outcome'] == 'success'
        c['success'] += int(ok)
        n_look = 0
        for labels, act in zip(rows['labels'], rows['actions']):
            c['decisions'] += 1
            c['offered'] += int(any(is_look(x) for x in labels))
            if is_look(labels[act]):
                c['chosen'] += 1
                n_look += 1
        if n_look:
            c['ep_look'] += 1
            c['ep_look_success'] += int(ok)
    out = {}
    for k, c in sorted(by.items()):
        out[k] = dict(episodes=c['episodes'], sr=round(c['success']/max(c['episodes'], 1), 3),
                      look_share=round(c['chosen']/max(c['decisions'], 1), 3),
                      offered_share=round(c['offered']/max(c['decisions'], 1), 3),
                      episodes_with_look=c['ep_look'],
                      sr_with_look=round(c['ep_look_success']/max(c['ep_look'], 1), 3) if c['ep_look'] else None)
    # counterfactual value of looking
    cf = dict(states=0, look_cands=0, look_succ=0, other_cands=0, other_succ=0, critical=0, crit_look_fix=0,
              crit_other_fix=0, crit_look_cands=0, crit_other_cands=0)
    for f in Path(run, 'cf_online').glob('*.json'):
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        for st in rec.get('states', {}).values():
            cands = st.get('candidates')
            if not cands:
                continue
            labels = st['labels']
            dflt = labels.index('continue') if 'continue' in labels else (labels.index('goal') if 'goal' in labels else 0)
            done = [c for c in cands if not c.get('alive') and not c.get('skipped')]
            if not done:
                continue
            cf['states'] += 1
            for lab, c in zip(labels, cands):
                if c.get('alive') or c.get('skipped'):
                    continue
                s = c['outcome'] == 'success'
                if is_look(lab):
                    cf['look_cands'] += 1; cf['look_succ'] += int(s)
                else:
                    cf['other_cands'] += 1; cf['other_succ'] += int(s)
            d = cands[dflt]
            if d.get('alive') or d.get('skipped') or d['outcome'] == 'success':
                continue
            cf['critical'] += 1
            for j, (lab, c) in enumerate(zip(labels, cands)):
                if j == dflt or c.get('alive') or c.get('skipped'):
                    continue
                s = c['outcome'] == 'success'
                if is_look(lab):
                    cf['crit_look_cands'] += 1; cf['crit_look_fix'] += int(s)
                else:
                    cf['crit_other_cands'] += 1; cf['crit_other_fix'] += int(s)
    cf['look_success_rate'] = round(cf['look_succ']/max(cf['look_cands'], 1), 3)
    cf['other_success_rate'] = round(cf['other_succ']/max(cf['other_cands'], 1), 3)
    cf['crit_look_fix_rate'] = round(cf['crit_look_fix']/max(cf['crit_look_cands'], 1), 3)
    cf['crit_other_fix_rate'] = round(cf['crit_other_fix']/max(cf['crit_other_cands'], 1), 3)
    print(json.dumps(dict(policy=out, counterfactual=cf), indent=1))


if __name__ == '__main__':
    main()
