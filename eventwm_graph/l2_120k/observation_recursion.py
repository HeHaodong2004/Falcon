"""Finite shared-world prefix/continuation probability identity on real histories."""
import json
import numpy as np
from eventwm_graph.l2_partial.data import read
from .data import OUT,SNAP
from .evaluate import model_set,restore
from .observations import predict_observations


def main():
    wm=model_set()['D_trained_graph'];records=[];failures=[]
    for c in np.load(OUT/'dev.npy',allow_pickle=True).tolist()[:12]:
        meta=dict(root_id=c['root_id'],scene_id=c['scene_id'])
        try:
            folder=SNAP/'roots'/c['root_id'];l=read(folder/'legal_root.json.gz');qs=read(folder/'queries.json.gz')
            q=max(qs,key=lambda q:(len(q['compiled']['actions']),q['query_id']))
            b=restore(wm,c,l['partial_graph']);f=wm.predict_shared(b,3.,16,41)
            direct=predict_observations(wm,b,q,f)
            if not direct['successor_supported']:raise ValueError('unsupported declared route')
            n=len(direct['endpoint_times']);k=max(1,n//2)
            if k==n:raise ValueError('query has no interior endpoint')
            times=np.asarray(direct['endpoint_times']);indices=np.searchsorted(f.times,times-1e-9)
            poses=np.asarray(q['compiled']['planned_poses'])
            pd=wm.observation_law.detection(f.positions[:,indices[:k]],poses[1:k+1],
                np.array([h.visibility_mode for h in f.worlds]))
            prefix_none=np.prod(1-pd,axis=(1,2));pfirst=float(f.weights@prefix_none)
            # Declared external coarse event over legal future emissions. It is
            # not an exact hypothetical center measurement or an empty region.
            state=wm.successor_at(f,int(indices[k-1]),f.weights*prefix_none,poses[k],
                cursor=dict(query_id=q['query_id'],action_index=k,total_actions=len(q['compiled']['actions'])),
                refinement=('declared-prefix-no-detection',k))
            remaining=3.-(state.timestamp-b.timestamp)
            ff=wm.predict_shared(state,remaining,16,41)
            second=predict_observations(wm,state,q,ff)
            product=pfirst*second['none_probability']
            records.append({**meta,'prefix_actions':k,'total_scored_actions':n,
                'direct_none':direct['none_probability'],'prefix_none':pfirst,
                'conditional_remaining_none':second['none_probability'],
                'segmented_none':product,'probability_error':abs(direct['none_probability']-product),
                'posterior_weight_error':float(np.max(abs(ff.weights-f.weights*prefix_none/pfirst))),
                'normalization_error':abs(float(ff.weights.sum())-1.),
                'same_worlds':len(ff.worlds)==len(f.worlds)})
        except Exception as e:failures.append({**meta,'error':repr(e)})
    result=dict(records=records,failures=failures,expected_roots=12,
        passed=len(records)==12 and not failures and all(r['probability_error']<1e-10 and r['posterior_weight_error']<1e-10 for r in records),
        scope='first12 real legal development histories, single immutable query and held16-world law; no exact imaginary measurements or future truth')
    (OUT/'observation_recursion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
