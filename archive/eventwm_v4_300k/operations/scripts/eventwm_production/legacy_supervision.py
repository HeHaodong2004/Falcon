from settings import PROJECT as PROJECT_PATH, DATA as DATA_PATH, SCIENCE as SCIENCE_PATH, TARGET as TARGET_QUERIES
"""Backfill horizon masks from the verified legacy seals; raw remains read-only."""
import concurrent.futures as cf,hashlib,json,multiprocessing,pathlib,time
import successor_masks as derived
P=PROJECT_PATH
PROOF=P/'output/eventwm_72h_legacy_verified.json';OUT=P/'output/eventwm_72h_legacy_supervision_progress.json'

def one(root):
    proof=json.loads(PROOF.read_text());derived.DATA=pathlib.Path(proof['source'])
    marker=derived.DEST/f"{root['root_id']}.provenance.json"
    if marker.exists():
        existing=json.loads(marker.read_text())
        assert existing['source_dataset']==str(derived.DATA) and existing['samples']==root['stats']['primary_queries']
        assert hashlib.sha256((derived.DEST/f"{root['root_id']}.jsonl.gz").read_bytes()).hexdigest()==existing['artifact_sha256']
        return root['stats']['primary_queries']
    return derived.process(root)['samples']

def main():
    proof=json.loads(PROOF.read_text());assert proof['all_seals_verified'] and not proof['errors']
    base=pathlib.Path(proof['source']);assert hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest()==proof['manifest_sha256']
    roots=[r for r in json.loads((base/'manifest.json').read_text())['roots'] if r['state']=='completed']
    started=time.time();count=completed=0
    with cf.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for samples in pool.map(one,roots,chunksize=1):
            count+=samples;completed+=1
            result=dict(timestamp=time.time(),state='complete' if completed==len(roots) else 'running',completed_roots=completed,total_roots=len(roots),primary_queries=count,total_primary_queries=proof['retained'],seconds=time.time()-started,source_dataset=str(base),supervision_directory=str(derived.DEST),raw_data_mutated=False,supervision_only=True)
            temp=OUT.with_suffix('.tmp');temp.write_text(json.dumps(result));temp.replace(OUT)
            print(json.dumps(result),flush=True)
    assert count==proof['retained']

if __name__=='__main__':main()
