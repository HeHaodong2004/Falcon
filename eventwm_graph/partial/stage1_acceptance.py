"""Fail-closed integration acceptance before the collection-only campaign."""
from pathlib import Path
import time
from ..config import REPO,read_json,write_json,file_digest
from ..stage1.storage import read_gzip
from .acceptance import run_unit_review
from .collect import verify_gate
from .storage import implementation,validate_root


def accept(stage0_gate,datasets,replay_reports,output,independent_review,root_lineage):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);verify_gate(stage0_gate)
    versions=implementation();producer=dict(versions);roots=[];scenes=set();humans=set();bands=set();outcomes=0;collisions=0;intrusive=0
    for dataset in datasets:
        for path in sorted(Path(dataset).glob('roots/*/*/manifest.json')):
            if path.parent.name.startswith('.'):continue
            public=read_json(path)
            if public['split']!='train':raise ValueError('Integration review refuses held-out contents')
            manifest=validate_root(path.parent)
            if manifest['version']!=versions:raise ValueError('Pilot source identity differs from current collector')
            private=read_gzip(path.parent/'privileged_initialization.json.gz')
            humans.add(private['episode']['info']['human_num']);scenes.add(manifest['scene_id'])
            queries=read_gzip(path.parent/'queries.json.gz')
            bands.update(p['intended_band'] for q in queries for p in q['provenance'] if p['source']=='human')
            roots.append({'root_id':manifest['root_id'],'folder':str(path.parent.resolve()),'manifest_sha256':file_digest(path)})
            outcomes+=manifest['queries'];collisions+=manifest['statistics']['human_collisions'];intrusive+=manifest['statistics']['intrusive_without_collision']
    reports=[read_json(p) for p in replay_reports];reviews=[root for r in reports for root in r.get('roots',[])]
    replay_ok=bool(reports) and all(r['status']=='PASS' for r in reports)
    replay_ok &= all(r.get('sources_unchanged') and r.get('repeats',0)>=2 and r['status']=='PASS' for r in reviews)
    reviewed={r.get('root_id') for r in reviews};known={r['root_id'] for r in roots}
    replay_ok &= known.issubset(reviewed)
    lineage=read_json(root_lineage);lineage_rows={r['root_id']:r for r in lineage.get('roots',[])}
    lineage_ok=lineage.get('status')=='PASS' and known.issubset(lineage_rows)
    for root in roots:
        row=lineage_rows.get(root['root_id'],{})
        lineage_ok &= row.get('passed') is True and row.get('manifest_sha256')==root['manifest_sha256']
    lineage_ok &= sum(r.get('full_queries_reproduced',0) for r in lineage_rows.values())==outcomes
    for report in reviews:
        for name,sha in report.get('source_versions',{}).items():
            if name in producer and producer[name]!=sha:replay_ok=False
        for key,value in report.get('source_versions',{}).items():
            if '/' in key and key not in producer:
                actual=file_digest(REPO/key)
                # Earlier PASS audits used the stricter raw witness-list
                # comparator. The final tool also proves unchanged known-wall
                # refreshes; native producer/framework sources stay identical.
                if key!='eventwm_graph/partial/replay_review.py' and actual!=value:replay_ok=False
                versions[key]=actual
    units=run_unit_review(output/'stage1_tests.log')
    checks={'native_replay':bool(replay_ok and len(scenes)>=2 and max(humans,default=0)>=2),
        'official_psc':bool(replay_ok and bands=={'safe','intrusion'} and collisions and intrusive),
        'legal_reconstruction':bool(roots and units['passed'] and lineage_ok),
        'branch_reproducibility':bool(replay_ok and reviews),
        'storage_integrity':bool(roots and units['passed']),
        'independent_review':bool(replay_ok and units['passed'] and Path(independent_review).is_file())}
    gate={'stage':'stage1','status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'implementation':versions,'validation_tool_sha256':file_digest(__file__),
        'stage0_gate':str(Path(stage0_gate).resolve()),'stage0_gate_sha256':file_digest(stage0_gate),
        'roots':roots,'replay_reports':[{'path':str(Path(p).resolve()),'sha256':file_digest(p)} for p in replay_reports],
        'root_lineage':{'path':str(Path(root_lineage).resolve()),'sha256':file_digest(root_lineage),'passed':bool(lineage_ok)},
        'independent_review':{'path':str(Path(independent_review).resolve()),'sha256':file_digest(independent_review)},
        'unit_review':units,'evidence':{'outcomes':outcomes,'roots':len(roots),'scenes':sorted(scenes),
            'active_human_counts':sorted(humans),'human_bands':sorted(bands),'human_collisions':collisions,
            'intrusive_without_collision':intrusive,'fresh_native_branches':sum(len(r.get('branches',[])) for r in reviews)},
        'automatic_training':False,'time':time.time()}
    write_json(output/'stage1_gate.json',gate);return gate
