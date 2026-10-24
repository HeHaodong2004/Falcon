"""Verify public input guard leaves fitted normalized predictive law unchanged."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from eventwm_graph.l2_partial.data import read
from eventwm_graph.l2_partial.snapshot import sha
from .data import OUT,SNAP
from .model import load_model
from .evaluate import restore,OPTICS


def main():
    packaged=Path('docs/l2/graph_wm_120k_20261021/artifacts/source_fit_model.py')
    source=packaged.read_bytes() if packaged.exists() else subprocess.check_output(
        ['git','show','99668ecf:eventwm_graph/l2_120k/model.py'])
    expected=json.loads((OUT/'training/configuration.json').read_text())['sources']['eventwm_graph/l2_120k/model.py']
    assert hashlib.sha256(source).hexdigest()==expected
    (OUT/'source_fit_model.py').write_bytes(source)
    namespace={'__name__':'fit_time_model_reference'};exec(compile(source,'<fit-time-model.py>','exec'),namespace)
    population=json.loads((OUT/'population_prior.json').read_text());motion=json.loads((OUT/'motion.json').read_text())
    args=(OUT/'training/checkpoint.pt',population,OPTICS,motion)
    old=namespace['load_model'](*args);new=load_model(*args);assert old.model_version==new.model_version
    rows=np.load(OUT/'dev.npy',allow_pickle=True).tolist()[:12];comparisons=[]
    for row in rows:
        legal=read(SNAP/'roots'/row['root_id']/'legal_root.json.gz')
        a=restore(old,row,legal['partial_graph']);b=restore(new,row,legal['partial_graph'])
        assert a.fingerprint()==b.fingerprint()
        for worlds in (8,16,32):
            left=old.predict_shared(a,3.,worlds,27);right=new.predict_shared(b,3.,worlds,27)
            for key in ('times','positions','velocities','weights'):
                assert np.array_equal(getattr(left,key),getattr(right,key),equal_nan=True),key
            assert left.rng_state==right.rng_state
            comparisons.append(dict(root_id=row['root_id'],worlds=worlds,arrays_bitwise_equal=True))
    partial=b.fork(hypotheses=[h.fork(weight=.7*h.weight) for h in b.hypotheses],unresolved_mass=.3).validate()
    try:new.predict_shared(partial,.048,16,27)
    except ValueError as error:assert 'explicit channel conversion' in str(error)
    else:raise AssertionError('incomplete measure silently normalized')
    result=dict(comparisons=comparisons,passed=True,valid_predictive_law_unchanged=True,
        rejected_incomplete_measure=True,fit_source_sha256=expected,
        selected_checkpoint_sha256=sha(OUT/'training/checkpoint.pt'),
        dev_cache_sha256=sha(OUT/'dev.npy'),manifest_sha256=sha(SNAP/'manifest.json'),
        final_source_sha256=sha(Path(__file__).with_name('model.py')),
        scope='36 real legal forecasts at8/16/32worlds: exact arrays and RNG versus fit-time source; boundary guard only, no fitting')
    (OUT/'boundary_compatibility.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
