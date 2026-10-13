"""Explicit nonvisual comparison projection; no numerical tolerance/rounding."""
import copy,json,hashlib
from pathlib import Path
from nonvisual_runtime import scientific_history,scientific_root,digest

def scientific_hashes(output,worker_id):
    out=Path(output)
    def read(rel):return json.loads((out/rel).read_text())
    terminal=read('diagnostic_jobs/'+worker_id+'/terminal.json')
    history=read('legal_observations/history.json') if (out/'legal_observations/history.json').exists() else terminal['legal']
    result={}
    privileged=out/'roots/privileged_root.json'
    root_hash=digest(scientific_root(read('roots/privileged_root.json')['state'],history)) if privileged.exists() else None
    terminal['legal']=scientific_history(terminal['legal'])
    result['terminal']=digest(terminal)
    for rel in ['roots/replay.json','roots/legal_root.json','queries/queries.json','audits/generation_v2.json','audits/root_sampling.json','audits/natural_prefix.json']:
        if (out/rel).exists():result[rel]=hashlib.sha256((out/rel).read_bytes()).hexdigest()
    if (out/'legal_observations/history.json').exists():result['history']=digest(scientific_history(history))
    for rel in ['roots/privileged_root.json',f'audits/workers/{worker_id}/root_fingerprint.json']:
        if (out/rel).exists():
            value=read(rel);value['state']=scientific_root(value['state'],history);value['sha256']=digest(value['state']);result[rel]=digest(value)
    rel=f'audits/workers/{worker_id}/branch.json'
    if (out/rel).exists():
        branch=read(rel);branch['root_fingerprint']=root_hash;branch['legal_observations']=scientific_history(branch['legal_observations']);result['branch']=digest(branch)
    rel=f'audits/workers/{worker_id}/observation_diagnostics.json'
    if (out/rel).exists():result['observation_diagnostics']=hashlib.sha256((out/rel).read_bytes()).hexdigest()
    return result
