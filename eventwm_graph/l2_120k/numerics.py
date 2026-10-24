"""124 optimized/reference public-path comparisons; legal inputs only."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from eventwm_graph.l2_partial.data import read
from .data import OUT,SNAP
from .systems import configured,snapshot_prefix


def main():
    sys.path.insert(0,str(Path('scripts').resolve()))
    from l2_partial_structural_benchmark import compare_same_law
    from l2_partial_repair_systems_stress import fixture
    left,right=configured(True),configured(False);rows=np.load(OUT/'dev.npy',allow_pickle=True).tolist()[:60]
    comparisons=[];failures=[];started=time.perf_counter()
    for i,r in enumerate(rows):
        folder=SNAP/'roots'/r['root_id'];l=read(folder/'legal_root.json.gz');queries=read(folder/'queries.json.gz')[:1]
        try:
            bl=left.initialize(l['history'][:-1],snapshot_prefix(l));br=right.initialize(l['history'][:-1],snapshot_prefix(l))
            for W in (16,8 if i%2==0 else 32):
                al=left.online_call(bl,l['history'][-1],queries,num_worlds=W,seed=27,budget_ms=1e6)
                ar=right.online_call(br,l['history'][-1],queries,num_worlds=W,seed=27,budget_ms=1e6)
                comparisons.append(dict(root_id=r['root_id'],worlds=W,**compare_same_law(al,ar)))
        except Exception as e:failures.append(dict(root_id=r['root_id'],error=repr(e)))
        if (i+1)%10==0:print('numerics',i+1,'seconds',round(time.perf_counter()-started,1),flush=True)
    for N in (6,12):
        for cert in (False,True):
            try:
                bl,ob,qs=fixture(left,N,cert);br,_,_=fixture(right,N,cert)
                al=left.online_call(bl,ob,qs[:1],num_worlds=16,seed=27,budget_ms=1e6)
                ar=right.online_call(br,ob,qs[:1],num_worlds=16,seed=27,budget_ms=1e6)
                comparisons.append(dict(synthetic_tracked=N,certificate=cert,**compare_same_law(al,ar)))
            except Exception as e:failures.append(dict(synthetic_tracked=N,certificate=cert,error=repr(e)))
    report=dict(comparisons=comparisons,failures=failures,expected=124,actual=len(comparisons),
        passed=not failures and len(comparisons)==124 and all(c['passed'] for c in comparisons),
        max_absolute_error=max((c['maximum_absolute_numeric_error'] for c in comparisons),default=None),
        workload='60 legal real prefixes x2 world budgets +4 dense stress; one fixed query, all returned successors; no truth or planner',
        seconds=time.perf_counter()-started)
    (OUT/'numerics.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('comparisons','failures')}),flush=True)


if __name__=='__main__':main()
