"""Executed synthetic probability oracles and bounded rollout value experiment."""
import argparse,json,pathlib,time,resource
import numpy as np
from .state import *
from .model import GraphBeliefWorldModel
from .probability import advance

class KnownEventOracle:
    def hazards(self,features):
        hazard=np.zeros((*features.shape[:-1],3));hazard[...,0]=features[...,1]>np.exp(-.65**2/(2*.4**2));hazard[...,1]=features[...,2]>np.exp(-1./(2*.8**2));return hazard

def initial():
    g=ObservedGraph('oracle',0,.2,0.,{(i,j):(1,0.) for i in range(-60,61) for j in range(-60,61)})
    hs=[]
    for z in (0.,5.):
        hs.append(Hypothesis(.5,np.array([[1.,z,0.,0.]]),np.zeros((1,4,4)),np.array([0.]),np.array([0]),np.array([1.]),
                             intent=np.zeros((1,2)),diffusion=np.zeros(1),mode=np.zeros(1,int),relaxation=0.))
    return BeliefState(g,hs,0.,np.zeros(4))

def make_query(pose,actions,name):
    p=np.asarray(pose).copy();poses=[p.tolist()]
    for a in actions:
        if a=='forward':p[0]+=25/120*np.cos(p[3]);p[2]-=25/120*np.sin(p[3])
        else:p[3]=np.arctan2(np.sin(p[3]+(10/120 if a=='left' else -10/120)),np.cos(p[3]+(10/120 if a=='left' else -10/120)))
        poses.append(p.tolist())
    return {'query_id':name,'executor_version':EXECUTOR_VERSION,'compiled':{'actions':actions,'planned_poses':poses,'graph_revision':0},'waypoints':[poses[-1][:3]],'final_yaw':poses[-1][3]}

def utility(model,b,queries,seed=1):
    f=model.predict_shared(b,3.,16,seed=seed);batch=model.evaluate_queries(b,queries,f)
    # Short route +1, detour -.5, contact penalty6; synthetic decision objective.
    values=np.array([(1. if q['query_id']=='short' else -.5)-6*p.collision_probability for q,p in zip(queries,batch.predictions)])
    return float(values.max()),int(values.argmax()),batch

def run(output,repeats=50):
    wm=GraphBeliefWorldModel(kind='memory_cv',event_model=KnownEventOracle());b=initial();observe=make_query(b.robot_pose,['left']*6,'look')
    short=make_query(b.robot_pose,['forward']*10,'short');long=make_query(b.robot_pose,['left']*38+['forward']*10,'long')
    timings={0:[],1:[],2:[]};utilities={0:[],1:[],2:[]};branch_changes=[]
    for seed in range(repeats):
        t=time.perf_counter();u0=1.-6*.5;timings[0].append((time.perf_counter()-t)*1000);utilities[0].append(u0)
        t=time.perf_counter();u1,choice,_=utility(wm,b,[short,long],seed);timings[1].append((time.perf_counter()-t)*1000);utilities[1].append(u1)
        t=time.perf_counter();f=wm.predict_shared(b,.288,16,seed);p=wm.evaluate_queries(b,[observe],f).predictions[0];u2=-.1;choices=[]
        for branch in p.observation_branches:
            if branch.successor.terminal:u2-=6*branch.probability;continue
            c=branch.successor;q1=make_query(c.robot_pose,['forward']*10,'short');q2=make_query(c.robot_pose,['left']*32+['forward']*10,'long')
            v,chosen,_=utility(wm,c,[q1,q2],seed+100);u2+=branch.probability*v;choices.append({'observation':branch.observation['kind'],'probability':branch.probability,'choice':chosen})
        timings[2].append((time.perf_counter()-t)*1000);utilities[2].append(u2)
        if seed==0:branch_changes=choices
    closure=[]
    b=initial();b=b.fork(hypotheses=[h.fork(intent=np.array([[.5,0.]]),diffusion=np.array([.2]),relaxation=.7) for h in b.hypotheses])
    analytic=[advance(h,3.) for h in b.hypotheses];true_mean=sum(h.weight*h.mean[0,:2] for h in analytic)
    for depth in (1,2,3):
        errors=[];second=[];times=[];component_mass=[]
        for seed in range(32):
            c=b;t=time.perf_counter()
            for j in range(depth):
                f=wm.predict_shared(c,3/depth,16,seed=seed*7+j,times=[0.,3/depth]);c=wm.successor_at(f,1,f.weights,c.robot_pose)
            mean=sum(h.weight*h.mean[0,:2] for h in c.hypotheses);errors.append(float(np.linalg.norm(mean-true_mean)));times.append((time.perf_counter()-t)*1000)
            component_mass.append(c.unresolved_mass);second.append(float(sum(h.weight*(h.mean[0,:2]@h.mean[0,:2]+np.trace(h.covariance[0,:2,:2])) for h in c.hypotheses)))
        closure.append({'depth':depth,'mean_error':float(np.mean(errors)),'mean_error_std':float(np.std(errors)),'second_moment':float(np.mean(second)),'latency_ms':float(np.median(times)),'unresolved_mass_max':max(component_mass)})
    h=b.hypotheses[0];direct=advance(h,3.);composed=h
    for _ in range(30):composed=advance(composed,.1)
    # Repeated same-state calls exercise bounded cache allocations, after warmup.
    mem0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for i in range(200):utility(wm,initial(),[short,long],i)
    mem1=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    value={'scope':'known synthetic two-route observation-opportunity oracle; NOT native or closed-loop navigation evidence',
           'depth':[{'depth':d,'expected_utility':float(np.mean(utilities[d])),'latency_p50_ms':float(np.median(timings[d])),'latency_p95_ms':float(np.percentile(timings[d],95))} for d in (0,1,2)],
           'observation_choices':branch_changes,'recursive_fixed_3s':closure,'composition_mean_max_error':float(np.max(abs(direct.mean-composed.mean))),
           'composition_covariance_max_error':float(np.max(abs(direct.covariance-composed.covariance))),
           'successor_position_variance_added_per_coordinate':.005**2,'rss_max_kib_growth_200_calls':mem1-mem0,'cache_entries':len(wm.graph_cache)}
    pathlib.Path(output).write_text(json.dumps(value,indent=2));print(json.dumps(value,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--repeats',type=int,default=50);a=p.parse_args();run(**vars(a))
