from settings import PROJECT as PROJECT_PATH, DATA as DATA_PATH, SCIENCE as SCIENCE_PATH, TARGET as TARGET_QUERIES
import collections,gzip,hashlib,json,math,pathlib,time
P=PROJECT_PATH;D=P/'output/eventwm_300k_supervision'
counts=collections.Counter();horizons=collections.defaultdict(collections.Counter);ids=set();roots=0;started=time.time()
for proof_file in sorted(D.glob('*.provenance.json')):
    proof=json.loads(proof_file.read_text());artifact=D/f"{proof['root_id']}.jsonl.gz"
    assert proof['supervision_only'] and hashlib.sha256(artifact.read_bytes()).hexdigest()==proof['artifact_sha256']
    n=0
    with gzip.open(artifact,'rt') as f:
        for line in f:
            row=json.loads(line);assert row['supervision_only'] and row['sr_valid'] and row['sr_label'] in (0,1)
            assert row['query_id'] not in ids;ids.add(row['query_id']);n+=1
            assert [h['horizon_seconds'] for h in row['horizons']]==[.25,.5,1.,2.,3.]
            for h in row['horizons']:
                c=horizons[str(h['horizon_seconds'])];valid=h['censoring_masks']['pose']
                c['valid']+=int(valid);c['censored']+=int(not valid)
                if not valid:assert h['actual_human_poses'] is None and h['derived_human_velocities'] is None and h['actual_sample_timestamp'] is None
                else:
                    assert h['actual_sample_timestamp']<=row['end_state']['timestamp']+1e-8
                    # Native timestep .008s; always preserve the actual timestamp.
                    assert abs(h['actual_sample_timestamp']-h['target_timestamp'])<.02
                    assert len(h['actual_human_poses'])==len(h['derived_human_velocities'])
                    for v in h['derived_human_velocities']:
                        dt=v['source_timestamps'][1]-v['source_timestamps'][0]
                        assert dt>0 and all(math.isfinite(x) for x in v['position_difference_velocity'])
                        counts['velocity_vectors']+=1;counts['moving_velocity_vectors']+=int(any(abs(x)>1e-6 for x in v['position_difference_velocity']))
                if h['horizon_physical_collision'] is True:
                    assert h['horizon_sr']==0 and h['censoring_masks']['physical_event']
                    c['determined_early_collision']+=int(not valid)
                if not h['censoring_masks']['physical_event']:assert h['horizon_sr'] is None
            counts['sr_'+str(row['sr_label'])]+=1
    assert n==proof['samples'];roots+=1
report=dict(passed=True,timestamp=time.time(),roots=roots,queries=len(ids),counts=dict(counts),horizons={k:dict(v) for k,v in horizons.items()},
            seconds=time.time()-started,scope='All currently emitted supervision artifacts; no raw or model-input changes',velocities_from_timestamps_and_positions=True)
(P/'output/eventwm_72h_successor_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
