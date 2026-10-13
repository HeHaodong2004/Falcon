from settings import PROJECT as PROJECT_PATH, DATA as DATA_PATH, SCIENCE as SCIENCE_PATH, TARGET as TARGET_QUERIES
"""Require target, seals, full supervision coverage, and final input/target audits."""
import hashlib,json,pathlib,sqlite3,subprocess,sys,time
from active_release import active_repo
P=PROJECT_PATH;CONTROL=pathlib.Path(__file__).resolve().parent;DEST=P/'output/eventwm_300k_supervision'
sys.path.insert(0,str(active_repo()/'scripts'))
from eventwm_pipeline_io import read_status
from monitor import snapshot
from eventwm_validate_staged import durable_json
check_only='--check-only' in sys.argv
while True:
    integrity_path=P/'output/eventwm_300k_combined_index/integrity.json';integrity=json.loads(integrity_path.read_text())
    state=read_status(DATA_PATH)['state']
    ready=bool(integrity.get('passed') and integrity.get('total_verified',0)>=TARGET_QUERIES and state in ('stopped_by_user','target_reached'))
    with sqlite3.connect('file:'+str(P/'output/eventwm_300k_combined_index/verified.sqlite3')+'?mode=ro',uri=True) as db:
        roots=list(db.execute('SELECT root_id,dataset,path,role,queries FROM roots'))
    missing=[];covered=0
    for rid,dataset,path,role,queries in roots:
        marker=DEST/f'{rid}.provenance.json'
        if not marker.exists():missing.append(rid);continue

        try:proof=json.loads(marker.read_text())
        except (FileNotFoundError,json.JSONDecodeError):missing.append(rid);continue
        assert proof['root_id']==rid and proof['source_dataset']==dataset and proof['sealed_root_path']==path and proof['role']==role and proof['samples']==queries and proof['supervision_only']
        if not (DEST/f'{rid}.jsonl.gz').exists():missing.append(rid);continue
        covered+=queries
    result=dict(timestamp=time.time(),collector_state=state,deep_verified_total=integrity['total_verified'],supervision_covered_queries=covered,missing_supervision_roots=len(missing),retention_target_ready=ready,mission_complete=False)
    durable_json(P/'output/eventwm_72h_finalization_status.json',result);print(json.dumps(result),flush=True)
    if check_only:break
    if ready and not missing:
        for rid,_,_,_,_ in roots:
            proof=json.loads((DEST/f'{rid}.provenance.json').read_text())
            assert hashlib.sha256((DEST/f'{rid}.jsonl.gz').read_bytes()).hexdigest()==proof['artifact_sha256']
        for name in ('audit_combined_inputs.py','audit_successor.py'):
            subprocess.run([sys.executable,str(CONTROL/name)],cwd=P,check=True)
        inputs=json.loads((P/'output/eventwm_72h_combined_input_audit.json').read_text());successor=json.loads((P/'output/eventwm_72h_successor_audit.json').read_text())
        assert inputs['passed'] and successor['passed'] and inputs['primary_query_inputs']==covered
        assert successor['queries']==covered
        target=json.loads((P/'output/eventwm_72h_TARGET_REACHED.json').read_text())
        final=snapshot();final.update(mission_target_reached=True,mission_complete=True,within_deadline=target['within_deadline'],first_verified_target_observed_london=target['now_london'],deep_verified_total=integrity['total_verified'],supervision_verified_queries=covered,integrity_report=str(integrity_path),public_input_audit=str(P/'output/eventwm_72h_combined_input_audit.json'),successor_audit=str(P/'output/eventwm_72h_successor_audit.json'))
        durable_json(P/'output/eventwm_72h_MISSION_RESULT.json',final);print('MISSION_RESULT',json.dumps(final),flush=True);break
    time.sleep(90)
