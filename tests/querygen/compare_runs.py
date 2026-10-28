"""Compare two official runs episode by episode (identity checks between code versions).

  python tests/querygen/compare_runs.py RUN_A RUN_B      (directories with <suite>_<cfg>/<episode>.json)
Prints per common episode whether the verdict, the Oracle edge count, the decision sequence (kind, label, cont,
horizon, via, heading, duration, meta) and the replay actions / official metrics are identical."""
import glob
import json
import os
import sys


def load(root):
    out = {}
    for f in glob.glob(os.path.join(root, '*', '*.json')):
        if f.endswith(('run.json', 'summary.json')):
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        out[(os.path.basename(os.path.dirname(f)), r['key'])] = r


    return out


def dec(r):
    return [(d.get('kind'), d.get('label'), d.get('cont'), d.get('horizon'), d.get('via'), d.get('heading'),
             d.get('duration'), json.dumps(d.get('meta'), sort_keys=True)) for d in r.get('decisions', [])]


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(a) & set(b))
    same_all = True
    for k in common:
        x, y = a[k], b[k]
        rx, ry = x.get('replay', {}), y.get('replay', {})
        checks = dict(verdict=x['verdict'] == y['verdict'],
                      edges=x['oracle']['stats']['children'] == y['oracle']['stats']['children'],
                      decisions=dec(x) == dec(y),
                      steps=x['oracle']['steps'] == y['oracle']['steps'],
                      metrics=rx.get('metrics') == ry.get('metrics'))
        ok = all(checks.values())
        same_all &= ok
        print(('SAME ' if ok else 'DIFF ')+k[0]+' '+k[1], '' if ok else {c: v for c, v in checks.items() if not v},
              'edges', x['oracle']['stats']['children'], y['oracle']['stats']['children'],
              'sec', round(x['oracle']['seconds']), round(y['oracle']['seconds']))
    print(f'{len(common)} common episodes:', 'ALL_IDENTICAL' if same_all and common else 'DIFFERENCES')


if __name__ == '__main__':
    main()
