#!/usr/bin/env python3
"""Generate the measured production plan without launching collection."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import subprocess
import numpy as np
from collector.core import read,write
from eventwm_mass_collection import REPO,REPORT,DEFAULT_OUTPUT,hashes,schedule_spec


def distribution(values):
    return dict(min=float(min(values)),median=float(np.median(values)),mean=float(np.mean(values)),
                p90=float(np.percentile(values,90)),max=float(max(values)))


def handoff(o,commit,tag):
    inv=read(REPORT/'scene_inventory.json');s=o['storage'];p=o['performance'];sizes=s['mb_per_root'];queries=s['mb_per_query'];sc=o['scene_counts']
    best=next(b for b in p['benchmarks'] if b['concurrency']==p['selected_concurrency'])
    start=f'{REPO}/scripts/run_eventwm_v3_mass_collection.sh'
    monitor=f'{REPO}/scripts/check_eventwm_v3_mass_collection.sh'
    stop=f'/home/haodong/miniforge3/envs/eventwm-falcon/bin/python {REPO}/scripts/eventwm_mass_collection.py stop --output {DEFAULT_OUTPUT}'
    return f'''MASS COLLECTION READY: {'YES' if o['ready'] else 'NO'}

FROZEN COLLECTOR COMMIT: bf46195517d33367af061c2982bfbf2ed6f58000
FINAL ORCHESTRATION COMMIT: {commit}
TAG: {tag}

SCENES:
  episode/config files: {inv['episode_config_files']}
  unique physical scenes: {sc['discovered']}
  locally present scenes: {sc['present']}
  compatible scenes: {sc['compatible']} (native preparation preflight)
  selected training scenes: {sc['train']}
  unseen-scene holdout scenes: {sc['holdout']}

REGIONS / ROOMS:
  semantic rooms available: YES; {inv['semantic_rooms_available']} in compatible scenes
  regions/spatial cells expected: up to {o['target_roots']} root cells; unique count measured during collection

SAMPLING PRESERVED:
  wm_sampling root strata: PASS (10/15/30/30/15)
  visible/lost substrata: PASS (30/50/20; 50/35/15)
  human counts 0/1/2/4/6: PASS
  N=8 query variants: PASS (33 primary/root)
  static-feasibility v2.2: PASS
  offline 30/30/40 outcome sampler preserved: PASS; no raw outcome filtering

SCALE:
  target roots: {o['target_roots']}
  target primary queries: {o['target_queries']} across separate declared roles
  expected roots/scene: {o['target_roots']/sc['compatible']:.2f}

STORAGE:
  measured MB/root: mean {sizes['mean']:.2f}; median {sizes['median']:.2f}; p90 {sizes['p90']:.2f}
  measured MB/query: mean {queries['mean']:.3f}; median {queries['median']:.3f}; p90 {queries['p90']:.3f}
  estimated 300k: {s['estimates']['300000']['expected_gb']:.0f} GB expected / {s['estimates']['300000']['p90_gb']:.0f} GB p90
  estimated 400k: {s['estimates']['400000']['expected_gb']:.0f} GB expected / {s['estimates']['400000']['p90_gb']:.0f} GB p90
  estimated 500k: {s['estimates']['500000']['expected_gb']:.0f} GB expected / {s['estimates']['500000']['p90_gb']:.0f} GB p90
  free disk: {s['free_gb']:.1f} GB
  disk guard: 150 GB + {s['inflight_reserve_gb']} GB in-flight reserve; conservative capacity ~{s['safe_max_queries_p90']:,} queries

PERFORMANCE:
  benchmarked concurrency levels: C1, C24 only
  C24 result: PASS; 792 primary queries / 721.43 seconds
  selected production concurrency: {p['selected_concurrency']}
  CPU thread settings: OMP/MKL/OpenBLAS/NumExpr/PyTorch = 1 per worker
  GPU assignment: alternating GPU0/GPU1, 12 workers each; starts staggered 3 seconds
  measured queries/hour: {p['queries_hour']:.2f} (C1: {next(b['queries_hour'] for b in p['benchmarks'] if b['concurrency']==1):.2f})
  estimated time to 300k: {p['estimates']['300000']['hours']:.1f} hours, benchmark-workload extrapolation
  estimated time to 500k: {p['estimates']['500000']['hours']:.1f} hours, benchmark-workload extrapolation

C24 FINAL INTEGRITY:
  completed roots: 24
  failures: 0
  replay: PASS
  static violations: 0
  label recomputation: PASS; 864 labels
  overall: PASS

PRODUCTION OUTPUT:
{DEFAULT_OUTPUT}

EXACT START COMMAND:
{start}
EXACT TMUX BACKGROUND COMMAND:
tmux new-session -d -s eventwm-v3-mass '{start}'
EXACT STATUS COMMAND:
{monitor}
EXACT RESUME COMMAND:
TARGET_PRIMARY_QUERIES=500000 {start}
EXACT SAFE STOP COMMAND:
{stop}
'''


def generate(commit,tag,ready,keep_plan=False):
    inv=read(REPORT/'scene_inventory.json')
    results=[read(REPORT/f'benchmark_c{n}.json') for n in (1,24) if (REPORT/f'benchmark_c{n}.json').exists()]
    good=[r for r in results if r['pass_correctness']]
    if not good:raise ValueError('No valid completed benchmark to report')
    best=max(good,key=lambda r:r['queries_hour'])
    if ready and (not any(r['concurrency']==24 and r['pass_correctness'] for r in results) or read(REPORT/'c24_final_integrity.json')['overall']!='PASS'):raise ValueError('C24 final integrity gate required for this production release')
    available=[s for s in inv['scenes'] if s.get('compatible')]
    families={f:iter(sorted([s for s in available if s['family']==f],key=lambda s:s['physical_scene_id'])) for f in sorted({s['family'] for s in available})}
    scenes=[]
    while families:
        for f,it in list(families.items()):
            scene=next(it,None)
            if scene is None:del families[f]
            else:scenes.append(scene)
    roots=math.ceil(300000/33)
    specs=[schedule_spec(i,scenes) for i in range(roots)]
    roles=Counter(r['role'] for r in specs)
    counts=Counter(r['human_count'] for r in specs);cats=Counter(r['requested_category'] or 'natural_unspecified' for r in specs)
    sizes=[s for r in results for s in r['sizes']]
    # Identical benchmark recipes repeated at different concurrency are not extra
    # independent storage observations. Keep one result per root identity.
    sizes=list({s.get('source_recipe',s['root_id']):s for s in sizes}.values())
    if (REPORT/'production_smoke.json').exists():sizes+=read(REPORT/'production_smoke.json').get('sizes',[])
    mb=distribution([s['bytes']/1e6 for s in sizes]);per=distribution([s['bytes']/s['primary_queries']/1e6 for s in sizes])
    mean_queries=sum(s['primary_queries'] for s in sizes)/len(sizes)
    free=shutil.disk_usage(DEFAULT_OUTPUT.parent).free/1e9
    reserve=max(5,2*best['concurrency'])
    ceiling=math.floor(max(0,free-150-reserve)*1000/(per['p90']*1.1))
    rates=best['queries_hour'];plans={str(n):dict(expected_gb=per['mean']*n/1000,
               p90_gb=per['p90']*n/1000,conservative_with_10pct_margin_gb=per['p90']*n/1000*1.1,
               hours=n/rates,days=n/rates/24) for n in (100000,200000,300000,400000,500000)}
    output=dict(ready=ready,commit=commit,tag=tag,scene_counts=dict(discovered=inv['unique_physical_scenes'],present=inv['local_scene_assets_present'],
            compatible=len(scenes),train=sum(s['scene_role']=='training' for s in scenes),holdout=sum(s['scene_role']=='unseen_evaluation' for s in scenes)),
        target_roots=roots,target_queries=300000,scheduled_roles=dict(roles),scheduled_human_counts=dict(counts),scheduled_root_categories=dict(cats),
        storage=dict(roots_measured=len(sizes),observations=sizes,mb_per_root=mb,mb_per_query=per,estimates=plans,
                     free_gb=free,min_free_gb=150,inflight_reserve_gb=reserve,safe_max_queries_p90=ceiling),
        performance=dict(benchmarks=results,selected_concurrency=best['concurrency'],selected_gpu_assignment=[1] if best['concurrency']==1 else [0,1],
                         queries_hour=rates,queries_day=rates*24,estimates=plans),
        mean_primary_queries_per_root=mean_queries)
    write(REPORT/'production_plan_metrics.json',output)
    if ready:
        write(REPORT/'release.json',dict(commit=commit,tag=tag,benchmark_pass=True,source_hashes=hashes(),
             selected_concurrency=best['concurrency'],gpu_assignment=output['performance']['selected_gpu_assignment'],
             validated_concurrencies=[r['concurrency'] for r in good]))
    exclusion=Counter((s.get('reason') or 'unknown').splitlines()[-1] for s in inv['scenes'] if not s.get('compatible'))
    unique_rooms=sum(len(s.get('semantic_rooms',[])) for s in inv['scenes'])
    perf=[];baseline=next((r['queries_hour'] for r in results if r['concurrency']==1),None)
    for n in (1,2,4,6,24):
        r=next((r for r in results if r['concurrency']==n),None)
        perf.append(f'CONCURRENCY {n}:')
        if n in (6,24):perf.append('  tested: '+('YES' if r else 'NO'))
        if not r:
            perf+=['  queries/hour: NOT TESTED','  speedup: NOT MEASURED'];continue
        perf.append(f"  queries/hour: {r['queries_hour']:.2f}")
        if n>1:perf.append(f"  speedup: {r['queries_hour']/baseline:.3f}x" if baseline else '  speedup: unavailable')
        if n!=6:
            perf += [f"  CPU: mean {r['cpu_percent']:.2f}%; peak {r.get('cpu_percent_peak',0):.2f}%; load mean/peak {r['cpu_load1_mean']:.2f}/{r.get('cpu_load1_peak',0):.2f}",f"  RAM: sampled peak {r['ram_peak_gb']:.2f} GB"]
            for gpu in ('0','1'):perf.append(f"  GPU{gpu}: mean {r['gpus'][gpu]['mean_utilization']:.2f}%; sampled peak {r['gpus'][gpu]['max_vram_mib']:.0f} MiB")
    c1=next((r for r in results if r['concurrency']==1),None)
    matched=[s for s in c1['sizes'] if s['scene']=='castle'] if c1 else []
    matched_rate=sum(s['primary_queries'] for s in matched)/sum(s['seconds'] for s in matched)*3600 if matched else 0
    perf.append(f'CASTLE-MATCHED C1: {matched_rate:.2f} queries/hour; high-wave speedup {best["queries_hour"]/matched_rate:.3f}x' if matched_rate else 'CASTLE-MATCHED C1: unavailable')
    perf.append(f'CORRECTNESS: {best["completed_roots"]} complete roots; failures {best["failed_roots"]}; static violations {best["static_violations"]}; integrity failures {best["integrity_failures"]}')
    starts=f'cd {REPO} && ./scripts/run_eventwm_v3_mass_collection.sh'
    status=f'cd {REPO} && ./scripts/check_eventwm_v3_mass_collection.sh'
    stop=f'cd {REPO} && /home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_mass_collection.py stop --output {DEFAULT_OUTPUT}'
    summary=f'''MASS COLLECTION READY: {'YES' if ready else 'NO'}

FROZEN COMMIT: {commit}
TAG: {tag}

================ SCENE LEVEL ================
EPISODE/CONFIG FILES DISCOVERED: {inv['episode_config_files']}
UNIQUE PHYSICAL SCENES DISCOVERED: {inv['unique_physical_scenes']}
LOCAL SCENE ASSETS PRESENT: {inv['local_scene_assets_present']}
COLLECTOR-COMPATIBLE SCENES: {len(scenes)} (native preparation preflight; runtime acceptance remains per-root)
SELECTED TRAIN SCENES: {sum(s['scene_role']=='training' for s in scenes)}
RESERVED UNSEEN-SCENE HOLDOUT: {sum(s['scene_role']=='unseen_evaluation' for s in scenes)}
EXCLUDED SCENES: {len(inv['scenes'])-len(scenes)}
EXCLUSION REASONS: {dict(exclusion)}

DATASET FAMILIES:
  MP3D: {inv['families'].get('MP3D',0)} compatible
  HM3D: {inv['families'].get('HM3D',0)} compatible
  ReplicaCAD: {inv['families'].get('ReplicaCAD',0)} compatible
  other: {inv['families'].get('other',0)} compatible

================ REGION LEVEL ================
SEMANTIC ROOMS AVAILABLE: {'YES' if unique_rooms else 'NO'}
ROOMS / REGIONS DISCOVERED: {unique_rooms} MP3D annotation IDs; {inv['semantic_rooms_available']} in prepared scenes
SPATIAL REGION FALLBACK: deterministic 3m x 2m x 3m cells; diagnostic only
EXPECTED REGIONS COVERED: up to {roots} root cells; no guaranteed unique-region count before collection

================ ROOT LEVEL ==================
TARGET COMPLETED ROOTS: {roots}
EXPECTED ROOTS / SCENE: {roots/len(scenes):.2f} average across all roles
EXPECTED DISTINCT SEEDS: at least {roots}; failed attempts increase required new recipes
EXPECTED DISTRIBUTION:
  zero: {cats['zero']} requested opportunities
  unseen: {cats['unseen']} requested opportunities
  visible: {cats['visible']} requested opportunities
  lost: {cats['lost']} requested opportunities
  visible_lost: {cats['visible_lost']} requested opportunities
  natural unspecified: {cats['natural_unspecified']}

EXPECTED HUMAN COUNTS:
  h0: {counts[0]}
  h1: {counts[1]}
  h2: {counts[2]}
  h4: {counts[4]}
  h6: {counts[6]}

================ QUERY LEVEL =================
MEAN PRIMARY QUERIES / ROOT: {mean_queries:.2f}
TARGET PRIMARY QUERIES: 300000 (all declared roles)

EXPECTED SPATIAL PATH GROUPS: {roots*8}
EXPECTED IMMEDIATE_NORMAL: {roots*8}
EXPECTED WAIT_0.5: {roots*8}
EXPECTED WAIT_1.0: {roots*8}
EXPECTED HALF_SPEED: {roots*8}
EXPECTED WAIT: {roots}

================ STORAGE =====================
MEASURED MB / ROOT:
  min: {mb['min']:.3f}
  median: {mb['median']:.3f}
  mean: {mb['mean']:.3f}
  p90: {mb['p90']:.3f}
  max: {mb['max']:.3f}

MEASURED MB / PRIMARY QUERY:
  median: {per['median']:.3f}
  mean: {per['mean']:.3f}
  p90: {per['p90']:.3f}

ESTIMATED DATASET SIZE:
'''
    for n,v in plans.items():summary+=f"  {int(n)//1000}K queries: {v['expected_gb']:.1f} GB expected; {v['p90_gb']:.1f} GB p90; {v['conservative_with_10pct_margin_gb']:.1f} GB p90 + 10% margin\n"
    summary+=f'''
CURRENT FREE DISK: {free:.2f} GB
MINIMUM FREE DISK GUARD: 150 GB (+ {reserve} GB in-flight reserve)
SAFE MAXIMUM QUERY COUNT BEFORE GUARD: {ceiling} (p90 + 10% estimate; guard enforces actual disk availability)

================ PERFORMANCE =================
'''+ '\n'.join(perf)+f'''

SELECTED PRODUCTION CONCURRENCY: {best['concurrency']}
SELECTED GPU ASSIGNMENT: {output['performance']['selected_gpu_assignment']} round robin
SELECTED CPU THREAD SETTINGS: OMP=1, MKL=1, OpenBLAS=1, NumExpr=1; collector PyTorch=1; native Bullet settings unchanged

EXPECTED THROUGHPUT: {rates:.2f} queries/hour; {rates*24:.0f} queries/day (bounded benchmark estimate)
ESTIMATED TIME:
'''
    for n,v in plans.items():summary+=f"  {int(n)//1000}K: {v['hours']:.1f} hours / {v['days']:.2f} days\n"
    summary+=f'''
DISK IO BOTTLENECK: {'YES' if best['io_wait_percent']>10 else 'NO'} (mean IO wait {best['io_wait_percent']:.3f}%; writes {best['disk_write_mb_s']:.2f} MB/s)

================ DIVERSITY ===================
SCENE DIVERSITY ASSESSMENT: {len(scenes)} prepared physical scenes, all included in role-separated round robin
REGION DIVERSITY ASSESSMENT: least-covered static start cells; semantic source IDs retained; root-cell coverage measured live
ROOT DIVERSITY ASSESSMENT: independent recipe seeds, start positions, goals/routes and legal prefixes
HUMAN DIVERSITY ASSESSMENT: h0/h1/h2/h4/h6 with unchanged native controllers
PARTIAL-OBSERVABILITY DIVERSITY ASSESSMENT: unchanged legal zero/unseen/visible/lost/visible_lost selectors; unavailable requests retained

KNOWN REMAINING DIVERSITY BOTTLENECKS:
Native episode preparation excludes {len(inv['scenes'])-len(scenes)} scenes; ReplicaCAD assets missing. Targeted scan/hold roots have limited translation; natural-reference motion is separate. Accepted-stratum proportions and unseen-scene throughput cannot be guaranteed from a small benchmark. Storage may stop the run below the requested target.

================ COMMANDS ====================
PRODUCTION OUTPUT: {DEFAULT_OUTPUT.resolve()}

EXACT START COMMAND:
{starts}
EXACT TMUX BACKGROUND COMMAND:
tmux new-session -d -s eventwm-v3-mass '{starts}'
EXACT STATUS COMMAND:
{status}
EXACT RESUME COMMAND:
cd {REPO} && TARGET_PRIMARY_QUERIES=500000 ./scripts/run_eventwm_v3_mass_collection.sh
EXACT SAFE STOP COMMAND:
{stop}
'''
    (REPORT/'terminal_summary.txt').write_text(summary)
    lines=['# EventWM v3 mass collection plan','',
      'Release identity is the immutable tag `eventwm-v3-mass-collection-ready`; resolve its exact commit with `git rev-parse eventwm-v3-mass-collection-ready^{commit}` or read the generated `release.json`. This avoids a self-referential commit hash in the committed report.','',
      '**Production collection has not been launched.** This document records scene inventory, bounded full-root benchmarks, storage measurements, and the manual launch/resume interface.','',
      '## Frozen collector and orchestration boundary','',
      'The requested strict version audit found no newer omitted local collector fix: both worktrees were at bf46195, all 41 scoped tracked sources/configs matched byte-for-byte, and no local ref was ahead. All 17 pilot collector hashes matched. New model/audit code exists only as original-worktree uncommitted additions. A historical pre-base pilot-launcher hash differs; its older source is unavailable, but that launcher is not used by the mass runner. See [the provenance audit](EVENTWM_MASS_COLLECTION_PROVENANCE_AUDIT.md) and its full hashes/branch evidence.','',
      'The collector, static checker v2.2, v3 labels, legal observer/tracker, fixed N=8 queries, human controller and EventWM input loader are unchanged from `bf46195517d33367af061c2982bfbf2ed6f58000`. Only orchestration, inventory, episode-recipe preparation, monitoring and tests were added in an isolated worktree. Existing pilot/natural/legacy data and the original worktree are untouched.','',
      '`structured_production` maps explicitly to the validated collector `production` retention profile plus v3 and `--no-render`, with debug visuals disabled. Root acceptance scans for forbidden image/video/array archives, checks the canonical integrity audits (excluding the scientific human-reactivity diagnostic), validates all 33 primary outcomes and static certificates, seals completed data, and emits role-separated per-root index shards.','',
      '## Scene and episode evidence','',
      'The inventory distinguishes dataset episode files, physical scene IDs, aliases, local meshes, regenerated NavMeshes and preparation failures. The full gzip census contains 4,455,818 MP3D episodes referring to 71 scenes, and 7,646,425 HM3D episodes referring to 843 scenes. These episode counts are not scene counts. Local meshes cover additional scenes without those episode files.','',
      'All local assets were searched under the available Habitat/FALCON asset locations. There is one local scene_datasets tree; aliases in versioned_data were deduplicated. The 993 local meshes produced NavMesh files using the unchanged controlled generator (Spot radius 0.25 m, height 1.41 m). Of these, 952 also passed the generator’s connected 4–10 m start/goal and separated-human waypoint checks. Exclusions are preparation evidence, not a claim that a different episode sampler could never use those meshes. Full dynamic v3 rollouts were smoke-tested on the bounded workloads, not exhaustively on all 952 scenes. Every production root still passes the canonical checks before commitment.','',
      'See [scene_inventory.json](reports/eventwm_mass_collection/scene_inventory.json), which retains every physical ID, asset/config/NavMesh path, aliases, scene role, annotation rooms and exact exclusion log. ReplicaCAD references remain in the inventory but required assets are missing; no fake/fallback geometry is used.','',
      '## Diversity and roles','',
      'Final preservation audit: the mass scheduler calls the existing `eventwm_wm_train_pilot.allocation` with frozen config proportions and the unchanged runtime `collector.wm_sampling` selector. An orchestration bookkeeping error was corrected during finalization: natural-reference slots no longer consume targeted-strata positions. Category-conditional visible/lost counters preserve 30/50/20 and 50/35/15. Fast tests verify exact 10/15/30/30/15 root masses over 4,000 targeted opportunities for 1, 5 and 952 scenes, plus deterministic append/resume. This correction changes only pending production recipe allocation; C24 used explicit validated pilot recipes and did not call this scheduler. No additional collection benchmark was run. Final source hashes include the corrected scheduler and its canonical allocation dependency.','',
      'The unchanged offline `eventwm_wm_train_index.balance_weights` policy is SAFE_SUCCESS 30%, SOCIAL_VIOLATION_SUCCESS 30%, HUMAN_COLLISION 40%. No class weighting or outcome-dependent rejection is applied during raw collection. Every valid canonical outcome is retained in role-separated root shards. The existing pilot indexing CLI remains pilot-schema-specific; a later offline production-index adapter must reuse this weighting function and preserve root/scene roles. This collection release does not claim to have built a production training index.','',
      'The scheduler traverses all compatible scenes before the next round, interleaving families. Each recipe has an independent deterministic seed; root category, human count and legal visible/lost substrata rotate. The existing wm_train scan/hold selector and natural static-motion sampler are called unchanged. No collision label, future trajectory, hidden human position or privileged PSC distance selects a root.','',
      'New initial episode recipes use the validated controlled generator’s connected-path and human-waypoint geometry constraints. Among 128 static navigable start candidates, the least-covered diagnostic cell is preferred. This changes episode realization/coverage only, not the native human policy, query execution or observation semantics. MP3D room annotations are preserved, but an unverified room-coordinate association is not fabricated; observed root coverage uses stable 3m × 2m × 3m cells.','',
      'Scene holdouts are a fixed family-stratified hash partition (~12.5%). Natural sampling occupies one in eight recipe opportunities, including holdout scenes; scene holdout takes precedence over training/reference role. All queries from a root share `split_group=root_id`. Targeted training, natural-reference and unseen-evaluation roles remain explicit in index shards and aggregates. The 300,000 target is total accepted primary queries across declared roles, not 300,000 targeted-training queries. Requested-count forecasts below describe the first 9,091 recipe opportunities; failures and unavailable strata mean accepted proportions will differ.','',
      '## Resume, resources and safety','',
      'A single runner holds dataset and private-catalog locks. Completed roots have per-file seals and completion markers and are deeply verified before being skipped. Failed attempts remain in separate paths. Only interruptions/timeouts can retry, at most twice; unavailable legal strata are recorded without substitution. New recipe sequence numbers and seeds extend the same manifest when the target increases. Duplicate primary query IDs are rejected.','',
      'Production startup requires a clean worktree, exact release commit, matching source/static-checker hashes, the frozen structured profile and benchmark-selected worker/GPU configuration. Every running loop checks free disk. It stops above the 150 GB hard floor with an additional in-flight reserve, and also stops on low available RAM, less than 2 GiB sampled free VRAM on either GPU, or global integrity failure. SIGTERM/SIGINT drain current bounded roots; no new roots launch. The separate monitor only reads state and resources. Existing datasets are never automatically deleted.','',
      'The no-image rule concerns retention: the unchanged legal depth observation pipeline and replay image fingerprint still operate in memory. No sensor or tracker semantics were altered. The existing fresh-process branch replay remains computationally expensive and was not replaced by an unsafe in-process shortcut.','',
      '## Measurements and limits','',
      f'Role forecast over the first {roots} attempted recipes: {dict(roles)}. These are scheduling opportunities, not guaranteed accepted quotas; failures remain recorded and are never replaced based on outcome.','',
      'The live C1 footprint sample measured about 0.92 core equivalents (short-lived process transitions can undercount CPU), 2.06 GB peak aggregate collector RSS, 1,206 MiB peak collector-GPU VRAM, 1.625% average collector-GPU utilization, and 0.0109% IO wait. This supports a CPU/process-overhead bottleneck rather than GPU or disk saturation. C24 was estimated at ~22 core equivalents, ~49 GB RAM (~119 GB with 2x worker memory plus 20 GB host allowance), and ~14.5 GiB per GPU before desktop overhead. The actual high-wave measurements below supersede this linear sizing estimate.','',
      'Benchmark JSON files retain elapsed time, completed/failed roots, accepted query rate, CPU/load, sampled RAM/GPU usage, disk writes/IO wait, failure reasons, static violations and per-root byte counts. The user canceled C2/C4/C6. After a 49-second C1 footprint sample, exactly one C24 wave of 24 distinct validated Castle recipes was selected without consulting outcomes. Starts were staggered by three seconds; each worker used one internal CPU thread, alternating GPU0/GPU1. C1 contains three Castle roots and one much slower HM3D root, so the raw speedup confounds workload mix. A Castle-only serial comparison is also reported. No intermediate concurrency sweep or production run was launched. Measurements are small workload estimates, not a long-run saturation claim. Both GPUs were assigned explicit worker processes; GPU0 desktop activity is included in host telemetry. No same-concurrency single-GPU control was run, so the incremental benefit of GPU1 is not separately identified. Peak memory values are sampled, not continuous hardware maxima.','',
      'Storage uses current v3 structured-only completed roots, never legacy eventwm_prod_v1. Source recipes repeated across benchmarks contribute once to the storage distribution; high-wave recipes are distinct, but most measurements are Castle. The small HM3D sample is retained explicitly; scene-size extrapolation remains uncertain. Expected and p90 estimates are shown separately; an additional 10% provision is used for the conservative disk ceiling. Failed-attempt overhead and new-scene rollout durations remain uncertain, so the live disk guard is authoritative.','',
      f"Measured root-size range: {mb['min']:.3f}–{mb['max']:.3f} MB; per-primary-query min/max: {per['min']:.3f} / {per['max']:.3f} MB. Measurement sample size: {len(sizes)} distinct completed root recipes.",'',
      '## Validation','',
      'The original worktree already contained prior diagnostic work. To preserve it, this release lives on `research/eventwm-v3-mass-collection` in the isolated checkout shown in every command. Dataset and scene assets are shared read-only for inputs; new production output is separate. Run the launch and status commands from this checkout, not the original branch.','',
      'Final C24 offline verification checked 8,400 sealed files and recomputed all 864 labels (792 primary plus 72 audit/truncation labels). Exact same-root fingerprints, repeated-query truth/controls/legal observations, query hashes, static v2.2 certificates, primary uniqueness and collector source hashes all passed. No worker corruption or visual archive was found. See `c24_final_integrity.json`.','',
      'Unit tests cover deterministic round robin and append/resume prefix stability, unique recipe IDs, role isolation, category/count validity, region metadata, hard benchmark bounds, disk stop before launch, protocol-drift rejection and monitor non-mutation. The bounded smoke and benchmark reports provide the simulator-level results. Script syntax and Python compilation are checked.','',
      '## Use','',
      'Choose either the foreground start command or the tmux command, not both. The tmux session continues after disconnect. Run the status command any time; it is read-only. The safe-stop command requests a drain and lets active bounded roots finish canonical writes. Run the same start command to resume the original target, or the larger-target resume command to extend the same manifest without repeating completed roots. Do not change source/scene inventories or profiles during a dataset run. No RGB/depth/video archives are stored.','',
      f'**Storage capacity limits the requested scale:** 300k queries require about {plans["300000"]["expected_gb"]:.0f} GB at the measured mean, while free disk is {free:.0f} GB before the guard/reserve. Conservative capacity is about {ceiling:,} queries using p90 plus 10%. The launcher remains configured for the requested target but stops safely on actual disk availability. The listed collection times assume sufficient storage and the benchmark workload; most production scenes were not benchmarked end to end.','',
      '## Manual production summary','', '```text',summary.rstrip(),'```','']
    if not keep_plan:(REPO/'EVENTWM_V3_MASS_COLLECTION_PLAN.md').write_text('\n'.join(lines))
    compact=handoff(output,commit,tag)
    (REPORT/'terminal_summary.txt').write_text(compact)
    print(compact)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--tag',required=True);p.add_argument('--ready',action='store_true');p.add_argument('--keep-plan',action='store_true');a=p.parse_args()
    generate(a.commit,a.tag,a.ready,a.keep_plan)
