"""Explicitly synthetic density/world/query and graph-overlay stress measurements."""
import argparse,json,pathlib,time,resource
import numpy as np
from .state import *
from .model import GraphBeliefWorldModel
from .neural import LearnedDynamics
from .queries import NativeEventModel
from .diagnostics import make_query

def run(checkpoint,event_checkpoint,output,repeats=12):
    wm=GraphBeliefWorldModel(dynamics=LearnedDynamics.load(checkpoint),event_model=NativeEventModel(json.loads(pathlib.Path(event_checkpoint).read_text())['coefficients']))
    records=[]
    workloads=[(h,3000,24,16) for h in (0,3,6,12,24)]+[(6,g,24,16) for g in (500,12000)]+[(6,3000,q,16) for q in (1,4)]+[(6,3000,24,w) for w in (8,32)]
    for humans,cells,candidates,worlds in workloads:
        side=int(np.ceil(np.sqrt(cells)));g=ObservedGraph('stress-'+str(cells),0,.2,0.,{(i-side//2,j-side//2):(1,0.) for i in range(side) for j in range(side)})
        means=np.zeros((humans,4));means[:,0]=1+np.arange(humans)%6;means[:,1]=(np.arange(humans)//6-1)*.6
        h=Hypothesis(1.,means,np.tile(np.diag([.01,.01,.2,.2]),(humans,1,1)),np.zeros(humans),np.arange(humans),np.array([1.]))
        b=BeliefState(g,[h],0.,np.zeros(4));queries=[make_query(b.robot_pose,['left']*(i%12)+['right']*(i//12)+['forward']*(1+i%4),'stress-'+str(i)) for i in range(candidates)]
        for mode in ('warm','version_invalidation'):
            for rep in range(repeats):
                c=b
                if mode=='version_invalidation':
                    # Synthetic arrived certificate far from routes, alters namespace.
                    c=b.fork(graph=g.update([{'cell':[1000+rep,1000],'state':1,'height':0.,'source':'synthetic_test_certificate'}]))
                row={'timestamp':.048,'robot_pose':[0.,0.,0.,0.],'tracks':[{'position':[p[0],0.,p[1]],'last_seen_time':.048,'visible_now':True,'valid':True} for p in means], 'static_updates':[]}
                t=time.perf_counter();updated=wm.observe(c,row);f=wm.predict_shared(updated,3.,worlds,seed=31);batch=wm.evaluate_queries(updated,queries,f);elapsed=(time.perf_counter()-t)*1000
                records.append({'humans':humans,'graph_cells':len(g.cells),'candidates':candidates,'worlds':worlds,'mode':('cold_initial' if mode=='warm' and rep==0 else mode),'complete_ms':elapsed,'refined':min(4,candidates)})
    # Force every overlay depth, including ninth-update materialization.
    graph_times=[];g=b.graph
    for i in range(100):
        t=time.perf_counter();g=g.update([{'cell':[2000+i,2000],'state':1,'height':0.,'source':'synthetic_test_certificate'}]);graph_times.append({'depth':g.depth,'ms':(time.perf_counter()-t)*1000})
    result={'scope':'synthetic stress, distinct queries; not natural visitation or production coexistence','records':records,'graph_updates':graph_times,'rss_max_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'support_cache_entries':len(wm.graph_cache)}
    pathlib.Path(output).write_text(json.dumps(result,indent=2));print('saved',len(records),'calls',output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('event_checkpoint');p.add_argument('output');p.add_argument('--repeats',type=int,default=12);run(**vars(p.parse_args()))
