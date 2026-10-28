"""Build the results page (index.html + assets) from report_official.py's JSON.

  python tests/querygen/build_page.py --report output/runs/official/report.json --out DIR
         [--video FILE|CAPTION ...] [--figure FILE|CAPTION ...] [--extra JSON]

Charts are inline SVG (native <title> tooltips per mark); colors follow the dataviz reference palette: categorical
slots in fixed order per configuration, status colors (with text labels) for episode outcomes.
"""
import argparse
import html
import json
import os
import shutil

CFG = [('c1k10', 'v10 C1 ORCA K=10', 'var(--s1)'), ('c1k6', 'v10 C1 ORCA K=6', 'var(--s2)'),
       ('c2k10', 'v10 C2 DWA K=10', 'var(--s3)'), ('c1k10@v11', 'v11 C1 ORCA K=10', 'var(--s4)')]
SUITES = [('dev', 'DEV'), ('unseen', 'Unseen 1'), ('unseen2', 'Unseen 2')]
OUTCOMES = [('success', 'success', 'var(--good)'), ('human_collision', 'human collision', 'var(--critical)'),
            ('timeout', 'timeout', 'var(--warning)'), ('other', 'other', 'var(--muted)')]


def esc(s):
    return html.escape(str(s))


def capability_svg(cap):
    W, H, L, B, T = 640, 300, 44, 40, 16
    groups = [s for s, _ in SUITES if any(f'{s}_{c}' in cap for c, _, _ in CFG)]
    cfgs = [c for c in CFG if any(f'{s}_{c[0]}' in cap for s in groups)]
    gw = (W-L-10)/max(1, len(groups))
    bw = min(44, (gw-24)/max(1, len(cfgs)))
    y = lambda v: T+(H-T-B)*(1-v)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Oracle success rate by suite and configuration">']
    for v in (0, .25, .5, .75, 1.):
        out.append(f'<line x1="{L}" x2="{W-10}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                   f'<text x="{L-6}" y="{y(v)+4:.1f}" class="tick" text-anchor="end">{int(v*100)}%</text>')
    out.append(f'<line x1="{L}" x2="{W-10}" y1="{y(.95):.1f}" y2="{y(.95):.1f}" class="target"/>'
               f'<text x="{W-12}" y="{y(.95)-5:.1f}" class="tick" text-anchor="end">95% target</text>')
    for gi, s in enumerate(groups):
        x0 = L+gi*gw+(gw-len(cfgs)*(bw+4)+4)/2
        for ci, (c, name, color) in enumerate(cfgs):
            d = cap.get(f'{s}_{c}')
            if not d:
                continue
            v = d['passed']/d['denominator'] if d['denominator'] else 0
            vf = d['passed']/d['finished'] if d['finished'] else 0
            x = x0+ci*(bw+4)
            tip = (f"{name}, {dict(SUITES)[s]}: {d['passed']} of {d['denominator']} episodes not proven infeasible "
                   f"({100*v:.1f}%); finished {d['finished']}")
            out.append(f'<g><title>{esc(tip)}</title><rect x="{x:.1f}" y="{y(v):.1f}" width="{bw:.1f}" '
                       f'height="{max(0, y(0)-y(v)):.1f}" rx="3" fill="{color}"/></g>')
            out.append(f'<text x="{x+bw/2:.1f}" y="{y(v)-5:.1f}" class="val" text-anchor="middle">{100*v:.0f}</text>')
        out.append(f'<text x="{L+gi*gw+gw/2:.1f}" y="{H-14}" class="lab" text-anchor="middle">{esc(dict(SUITES)[s])}</text>')
    out.append(f'<line x1="{L}" x2="{W-10}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="axis"/></svg>')
    return '\n'.join(out)


def floor_svg(floor):
    arms = [a for a in ('exec', 'exec_ub', 'exec_ube', 'exec_v12') if a in floor]
    names = {'exec': 'v10 baseline', 'exec_ub': 'breaker + bump memory', 'exec_ube': 'v11 (+ evasion reflex)',
             'exec_v12': 'v12 (+ plan keeping, bump-aware reflex)'}
    rows = [(a, p) for a in arms for p in ('orca', 'dwa') if p in floor[a]]
    W, rh, L = 640, 30, 300
    H = 20+len(rows)*rh+30
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Default-policy outcomes by arm and planner">']
    for i, (a, p) in enumerate(rows):
        c = floor[a][p]
        n = max(1, c['n'])
        parts = [(k, c.get(k, 0)) for k, _, _ in OUTCOMES[:3]]
        parts.append(('other', n-sum(v for _, v in parts)))
        yy = 14+i*rh
        out.append(f'<text x="{L-8}" y="{yy+15}" class="lab" text-anchor="end">{esc(names[a])} - {p.upper()} (n={c["n"]})</text>')
        x = L
        for (k, v), (_, lab, col) in zip(parts, OUTCOMES):
            w = (W-L-20)*v/n
            if w <= 0:
                continue
            out.append(f'<g><title>{esc(f"{names[a]}, {p.upper()}: {lab} {v} of {n} ({100*v/n:.0f}%)")}</title>'
                       f'<rect x="{x:.1f}" y="{yy}" width="{max(0, w-2):.1f}" height="{rh-10}" rx="3" fill="{col}"/></g>')
            if w > 34:
                out.append(f'<text x="{x+w/2-1:.1f}" y="{yy+14}" class="inbar" text-anchor="middle">{100*v/n:.0f}%</text>')
            x += w
    lx = L
    for _, lab, col in OUTCOMES:
        out.append(f'<rect x="{lx}" y="{H-18}" width="10" height="10" rx="2" fill="{col}"/>'
                   f'<text x="{lx+14}" y="{H-9}" class="tick">{esc(lab)}</text>')
        lx += 14+7*len(lab)+16
    out.append('</svg>')
    return '\n'.join(out)


def crit_svg(m):
    hist = m['critical_hist']
    keys = [k for k in hist]
    W, H, L, B, T = 420, 220, 36, 34, 14
    mx = max([hist[k] for k in keys]+[1])
    bw = (W-L-10)/len(keys)-8
    y = lambda v: T+(H-T-B)*(1-v/mx)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Critical decisions per solved episode">']
    for i, k in enumerate(keys):
        x = L+i*(bw+8)
        v = hist[k]
        lab = k if k != '6' else '6+'
        out.append(f'<g><title>{esc(f"{v} episodes need {lab} non-default decisions")}</title>'
                   f'<rect x="{x:.1f}" y="{y(v):.1f}" width="{bw:.1f}" height="{max(0, y(0)-y(v)):.1f}" rx="3" fill="var(--s1)"/></g>'
                   f'<text x="{x+bw/2:.1f}" y="{y(v)-5:.1f}" class="val" text-anchor="middle">{v}</text>'
                   f'<text x="{x+bw/2:.1f}" y="{H-16}" class="lab" text-anchor="middle">{lab}</text>')
    out.append(f'<line x1="{L}" x2="{W-10}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="axis"/></svg>')
    return '\n'.join(out)


def readiness_svg(rd):
    order = [('any_good_upper_bound', 'some candidate is good (upper bound)'), ('oracle_choice', "Oracle's choice"),
             ('legal_logreg_loeo', 'legal features, logistic regression (held-out episodes)'),
             ('legal_progress_minus_risk', 'legal: progress - 3 x overlap probability'),
             ('legal_progress', 'legal: sandbox progress'), ('legal_neg_p_overlap', 'legal: lowest overlap probability'),
             ('default', 'default (continue / goal)'), ('random', 'random candidate (expected)')]
    rows = [(k, lab, rd['top1_success'][k]) for k, lab in order if rd['top1_success'].get(k) is not None]
    nst = rd['states']
    W, rh, L = 640, 26, 330
    H = 10+len(rows)*rh+10
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Share of decision states where each selection rule picks a good candidate">']
    for i, (k, lab, v) in enumerate(rows):
        yy = 8+i*rh
        w = (W-L-60)*v
        out.append(f'<text x="{L-8}" y="{yy+14}" class="lab" text-anchor="end">{esc(lab)}</text>'
                   f'<g><title>{esc(f"{lab}: good candidate chosen at {100*v:.0f}% of {nst} states")}</title>'
                   f'<rect x="{L}" y="{yy}" width="{max(1, w):.1f}" height="{rh-8}" rx="3" '
                   f'fill="{"var(--muted)" if k in ("any_good_upper_bound", "random") else "var(--s1)"}"/></g>'
                   f'<text x="{L+w+6:.1f}" y="{yy+14}" class="val">{100*v:.0f}%</text>')
    out.append('</svg>')
    return '\n'.join(out)


def good_hist_svg(rd):
    hist = {int(k): v for k, v in rd['good_candidates_hist'].items()}
    keys = sorted(hist)
    W, H, L, B, T = 420, 220, 36, 34, 14
    mx = max([hist[k] for k in keys]+[1])
    bw = (W-L-10)/len(keys)-6
    y = lambda v: T+(H-T-B)*(1-v/mx)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Good candidates per decision state">']
    for i, k in enumerate(keys):
        x = L+i*(bw+6)
        v = hist[k]
        out.append(f'<g><title>{esc(f"{v} states with {k} good candidates")}</title>'
                   f'<rect x="{x:.1f}" y="{y(v):.1f}" width="{bw:.1f}" height="{max(0, y(0)-y(v)):.1f}" rx="3" fill="var(--s1)"/></g>'
                   f'<text x="{x+bw/2:.1f}" y="{y(v)-5:.1f}" class="val" text-anchor="middle">{v}</text>'
                   f'<text x="{x+bw/2:.1f}" y="{H-16}" class="lab" text-anchor="middle">{k}</text>')
    out.append(f'<line x1="{L}" x2="{W-10}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="axis"/></svg>')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--video', nargs='*', default=[])
    ap.add_argument('--figure', nargs='*', default=[])
    ap.add_argument('--extra', default='')
    a = ap.parse_args()
    rep = json.load(open(a.report))
    extra = json.load(open(a.extra)) if a.extra else {}
    os.makedirs(os.path.join(a.out, 'media'), exist_ok=True)
    media = []
    for kind, items in (('video', a.video), ('figure', a.figure)):
        for it in items:
            path, cap = it.split('|', 1)
            dst = os.path.join('media', os.path.basename(path))
            shutil.copy(path, os.path.join(a.out, dst))
            media.append((kind, dst, cap))
    cap = rep['capability']
    rows = []
    for c, name, color in CFG:
        for s, sname in SUITES:
            d = cap.get(f'{s}_{c}')
            if not d:
                continue
            fails = ', '.join(d['failures']) or 'none'
            spl = '' if d['spl'] is None else '%.3f' % d['spl']
            psc = '' if d['psc'] is None else '%.3f' % d['psc']
            steps = '' if d['steps_p50'] is None else '%.0f' % d['steps_p50']
            edges = '' if d['edges_p50'] is None else '%.0f / %.0f' % (d['edges_p50'], d['edges_p95'])
            rows.append(f'<tr><td><span class="sw" style="background:{color}"></span>{esc(name)}</td><td>{esc(sname)}</td>'
                        f'<td class="num">{d["passed"]} / {d["denominator"]}</td><td class="num">{100*d["sr"]:.1f}%</td>'
                        f'<td class="num">{d["finished"]}</td><td class="num">{spl}</td><td class="num">{psc}</td>'
                        f'<td class="num">{d["hcoll_among_solved"]:.0f}</td><td class="num">{steps}</td>'
                        f'<td class="num">{edges}</td><td class="wrap">{esc(fails)}</td></tr>')
    tiles = extra.get('tiles', [])
    tile_html = '\n'.join(f'<div class="tile"><div class="tv">{esc(t["value"])}</div><div class="tl">{esc(t["label"])}</div>'
                          f'<div class="ts">{esc(t.get("sub", ""))}</div></div>' for t in tiles)
    notes = '\n'.join(f'<li>{esc(n)}</li>' for n in extra.get('notes', []))
    media_html = '\n'.join(
        (f'<figure><video controls preload="metadata" src="{esc(p)}"></video><figcaption>{esc(c)}</figcaption></figure>'
         if k == 'video' else f'<figure><img src="{esc(p)}" alt="{esc(c)}"><figcaption>{esc(c)}</figcaption></figure>')
        for k, p, c in media)
    m = rep['minint']
    rd = rep.get('readiness')
    readiness_html = good_html = ''
    if rd:
        rc = rd.get('recovery') or {}
        rec_txt = (f"One-decision recovery: after a candidate whose default continuation fails, one later decision "
                   f"still saves the episode in {100*rc['recovered']:.0f}% of {rc['tested']} tested cases"
                   if rc.get('recovered') is not None else '')
        readiness_html = f"""<section class="card"><h2>Can legal information pick a good query?</h2><p>Share of {rd['states']} decision states ({rd['episodes']} solved episodes) where each selection rule picks a candidate that the default policy then carries to an official success. Grey bars are references.</p>{readiness_svg(rd)}</section>"""
        good_html = f"""<div class="card"><h2>Good candidates per decision state</h2><p>Candidates (of {rd['K_mean']:.1f} on average) after which the default policy succeeds; 1 means a single right answer, 2 or more means several reasonable choices. {esc(rec_txt)}.</p>{good_hist_svg(rd)}</div>"""
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Falcon Query-Space Foundation</title>
<style>
:root {{
  /* layout: one reading column, summary tiles first, then charts, tables, recordings */
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b; --ring:rgba(11,11,11,0.10);
  --font: system-ui, -apple-system, "Segoe UI", sans-serif;
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --plane:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#2c2c2a;
    --axis:#383835; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --ring:rgba(255,255,255,0.10); color-scheme: dark; }}
}}
:root[data-theme="dark"] {{ --plane:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#2c2c2a;
  --axis:#383835; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --ring:rgba(255,255,255,0.10); color-scheme: dark; }}
body {{ margin:0; background:var(--plane); color:var(--ink); font:15px/1.5 var(--font); }}
main {{ max-width:1080px; margin:0 auto; padding:28px 20px 60px; display:grid; gap:28px; }}
h1 {{ font-size:26px; margin:0; text-wrap:balance; }} h2 {{ font-size:18px; margin:0 0 10px; text-wrap:balance; }}
p {{ max-width:70ch; margin:0; color:var(--ink2); }}
.card {{ background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:18px; display:grid; gap:12px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
.tile {{ background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:14px; }}
.tv {{ font-size:28px; font-weight:650; }} .tl {{ color:var(--ink); font-weight:550; }} .ts {{ color:var(--ink2); font-size:13px; }}
svg {{ width:100%; height:auto; }} svg .grid {{ stroke:var(--grid); stroke-width:1; }} svg .axis {{ stroke:var(--axis); stroke-width:1; }}
svg .target {{ stroke:var(--ink2); stroke-dasharray:4 4; stroke-width:1; }}
svg text {{ fill:var(--ink2); font:12px var(--font); }} svg .val {{ fill:var(--ink); font-weight:600; }}
svg .lab {{ fill:var(--ink); }} svg .tick {{ fill:var(--muted); }} svg .inbar {{ fill:#ffffff; font-weight:600; }}
.two {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
.tablewrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--grid); white-space:nowrap; }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }} td.wrap {{ white-space:normal; min-width:160px; }}
.sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:baseline; }}
.media {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:16px; }}
figure {{ margin:0; display:grid; gap:6px; }} video, img {{ max-width:100%; border-radius:8px; border:1px solid var(--ring); background:var(--surface); }}
figcaption {{ color:var(--ink2); font-size:13px; }} ul {{ margin:0; padding-left:20px; color:var(--ink2); }}
:focus-visible {{ outline:2px solid var(--s1); outline-offset:2px; }}
</style></head><body><main>
<header style="display:grid;gap:8px"><h1>Falcon query-space foundation, official benchmark rules</h1>
<p>{esc(extra.get('lead', ''))}</p></header>
<section class="tiles">{tile_html}</section>
<section class="card"><h2>Oracle capability ceiling</h2>
<p>Share of episodes solved by the privileged Oracle selecting among the generator's legal queries, over every episode not proven natively infeasible (unfinished and unresolved episodes count as failures). Hover a bar for counts.</p>
<div style="display:flex;gap:16px;flex-wrap:wrap">{''.join(f'<span><span class="sw" style="background:{c}"></span>{esc(n)}</span>' for _, n, c in CFG)}</div>
{capability_svg(cap)}</section>
<section class="card"><h2>Default policy without selection</h2><p>Keep the running query, else the goal query; no Layer 3. Outcomes per foundation variant and planner (hover for counts).</p>{floor_svg(rep['floor'])}</section>
{readiness_html}
<section class="two">
<div class="card"><h2>Critical decisions per solved episode</h2><p>Minimum number of non-default queries the Oracle needs (C1 ORCA, K=10; {m['episodes']} episodes, median {m['critical_median']}).</p>{crit_svg(m)}</div>
{good_html}
</section>
<section class="card"><h2>Official metrics on solved episodes</h2><div class="tablewrap"><table>
<thead><tr><th>Configuration</th><th>Suite</th><th class="num">Solved</th><th class="num">SR</th><th class="num">Finished</th><th class="num">SPL</th><th class="num">PSC</th><th class="num">H-Coll</th><th class="num">Steps P50</th><th class="num">Oracle edges P50 / P95</th><th>Failures</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>
<section class="card"><h2>Navigation recordings</h2><div class="media">{media_html}</div></section>
<section class="card"><h2>Notes</h2><ul>{notes}</ul></section>
</main></body></html>"""
    with open(os.path.join(a.out, 'index.html'), 'w') as fh:
        fh.write(page)
    print('wrote', os.path.join(a.out, 'index.html'), 'media', len(media))


if __name__ == '__main__':
    main()
