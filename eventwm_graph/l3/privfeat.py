"""ANALYSIS ONLY (privileged): re-compile the labelled decisions of a run with the TRUE future of every person in place
of Stage 1's bank (oraclebank.py, long horizon), so that a critic can be trained on the same counterfactual labels with
privileged inputs (``ident.py --run <out>``).  Answers: are the labels predictable if the future of the people were
known?  If yes, the legal critic's gap is human prediction (horizon); if no, the outcomes are not determined by
anything a selector could know at decision time.

Each labelled episode of ``<run>/cf_online`` is replayed decision by decision in the L3 environment (same generator,
sandbox and compiler as the actors; scans if the episode had them); at labelled decisions the graph is compiled with
the analysis bank (env L3_ORACLE_BANK=all, L3_ORACLE_BANK_STEPS) and packed.  Output: ``<out>/spool/<episode>`` (rows
with packed graphs at the labelled decisions only) and ``<out>/cf_online/<episode>.json`` (the labels).

    L3_LAYOUT=v4 L3_INHERIT=1 L3_ORACLE_BANK=all L3_ORACLE_BANK_STEPS=186 python scripts/l3_run.py eventwm_graph.l3.privfeat \
        --run output/runs/R10 --out output/privfeat_R10 [--limit 200 --shard i --shards n]
"""
import argparse
import json
import os
import shutil
import zlib
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--reverse', action='store_true', help='process the shard in reverse order (helper workers)')
    a = ap.parse_args()
    assert os.environ.get('L3_ORACLE_BANK'), 'set L3_ORACLE_BANK=all (analysis only)'
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from eventwm_graph.sandbox.core import WorldModelSandbox, SandboxConfig
    from eventwm_graph.querygen.native import EpisodeSpec
    from .actor import read_episode, write_episode
    from .env import L3Episode, ConfigSpec
    from .compiler import GraphCompiler
    from .batch import pack
    from .cfonline import _query
    out = Path(a.out)
    (out/'spool').mkdir(parents=True, exist_ok=True)
    (out/'cf_online').mkdir(parents=True, exist_ok=True)
    sb = WorldModelSandbox.from_release(SandboxConfig(materialize=False, route=True), map_device='cuda:%d' % (a.shard % 2))
    comp = GraphCompiler(brake_tail=True)
    files = sorted(Path(a.run, 'cf_online').glob('*.json'))
    files = [f for f in files if zlib.crc32(f.name.encode()) % a.shards == a.shard][:a.limit]
    if a.reverse:
        files = files[::-1]
    for f in files:
        name = f.name[:-len('.json')]
        if (out/'spool'/name).exists():
            continue
        rec = json.loads(f.read_text())
        targets = {int(k) for k, v in rec['states'].items() if 'candidates' in v}
        if not targets:
            continue
        try:
            d = read_episode(Path(a.run, 'spool', rec['episode']))
        except Exception:
            continue
        cfg = ConfigSpec.parse(d['config'])
        ep = L3Episode(EpisodeSpec(**d['spec']), cfg, sb, compiler=comp, scan=bool(d.get('scan')))
        rows = d['rows']
        packed = [None]*len(rows['labels'])
        ok = True
        try:
            for i, dec in enumerate(d['metrics']['decision_list']):
                if ep.over or i > max(targets):
                    break
                if i in targets:
                    dd = ep.observe()
                    if dd.labels != rows['labels'][i]:
                        ok = False
                        break
                    packed[i] = pack(dd.graph)
                    ep.act(int(rows['actions'][i]))
                else:
                    ep.ep.run(None if dec['cont'] else _query(dec), horizon=int(dec['horizon']))
                    ep.decisions += 1
        except Exception as exc:
            print(json.dumps(dict(episode=name, error=repr(exc))), flush=True)
            ok = False
        finally:
            ep.close()
        if not ok:
            continue
        new_rows = dict(rows, packed=packed)
        write_episode(out/'spool', name[:-len('.ep')] if name.endswith('.ep') else name, d['metrics'], new_rows,
                      {k: v for k, v in d.items() if k not in ('metrics', 'rows')})
        shutil.copy(f, out/'cf_online'/f.name)
        print(json.dumps(dict(episode=name, labelled=len(targets))), flush=True)
    sb.close()


if __name__ == '__main__':
    main()
