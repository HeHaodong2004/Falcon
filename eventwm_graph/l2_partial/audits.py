"""Executed failure probes and native one-step diagnostics; never model selection."""
import argparse,json,pathlib
import numpy as np
import torch
from .data import rows,root_data
from .model import GraphBeliefWorldModel
from .neural import LearnedDynamics
from .evaluate import visible_ids,score_future,scene_ci
from .diagnostics import initial,make_query

class ZeroEvents:
    def hazards(self,x):return np.zeros((*x.shape[:-1],3))

def run(snapshot,checkpoint,population,output):
    prior=json.loads(pathlib.Path(population).read_text());learned=LearnedDynamics.load(checkpoint)
    target='7415a33faee8c8a4501d51a76a230a4c8fea97f3f4bbaf460a85147069c4efc6'
    meta,folder=next((m,p) for m,p in rows(snapshot,'dev') if m['root_id']==target);d=root_data(folder)
    history=d['legal']['history'];wm=GraphBeliefWorldModel(kind='memory_cv',population_prior=prior)
    # initialize checks final graph timestamp; replay prefixes from their legal deltas.
    from ..partial.map import PartialGraph
    g=PartialGraph(d['legal']['partial_graph']['resolution'],d['legal']['partial_graph']['height']);count=[]
    for i,obs in enumerate(history):
        g.update(obs);b=wm.initialize(history[:i+1],g.snapshot());mass=sum(h.weight for h in b.hypotheses)
        count.append({'frame':i,'visible':sum(t['visible_now'] for t in obs['tracks']),'max_slots':max(h.n for h in b.hypotheses),
            'retained_expected_total':sum(h.weight*(h.n+h.expected_hidden) for h in b.hypotheses)/mass,
            'retained_probability_above_6':sum(h.weight*sum(p for k,p in enumerate(h.hidden_pmf) if h.n+k>6) for h in b.hypotheses)/mass,'unresolved':b.unresolved_mass})
    b=initial();h=b.hypotheses[0].fork(weight=1.);b=b.fork(hypotheses=[h]);zero=GraphBeliefWorldModel(kind='memory_cv',event_model=ZeroEvents());q=make_query(b.robot_pose,['left','right'],'segmentation')
    direct=zero.evaluate_queries(b,[q],zero.predict_shared(b,.096)).predictions[0]
    first=zero.evaluate_queries(b,[q],zero.predict_shared(b,.048)).predictions[0];branch=next(x for x in first.observation_branches if x.observation['kind']=='survived_no_detection')
    second=zero.continue_rollout(branch.successor,q,horizon=.048)
    no=lambda p:next(x.probability for x in p.observation_branches if x.observation['kind']=='survived_no_detection')
    segmentation={'direct_no_detection':no(direct),'two_segments_no_detection':branch.probability*no(second)}
    models={'memory_cv':GraphBeliefWorldModel(kind='memory_cv',population_prior=prior),'learned':GraphBeliefWorldModel(dynamics=learned,population_prior=prior)}
    records=[];entropy=[];separation=[];mode_weight=[];crossing=[]
    for m,p in rows(snapshot,'dev'):
        d=root_data(p);ids=visible_ids(d)
        if not ids or len(d['times'])<2:continue
        b=models['memory_cv'].initialize(d['legal']['history'],d['legal']['partial_graph']);dt=float(d['times'][1]-b.timestamp)
        for name,model in models.items():
            c=b
            if name=='memory_cv':
                c=model.prepare(b);c=c.fork(hypotheses=[h.fork(diffusion=np.full(h.n,.64)) for h in c.hypotheses])
            f=model.predict_shared(c,dt,16,seed=43,times=[0.,dt]);s=score_future(f,d['truth'][1],ids,d['truth'][0],d['times'][1])
            if s:records.append({'root_id':m['root_id'],'scene_id':m['scene_id'],'model':name,**s})
        h=max(b.hypotheses,key=lambda h:h.weight)
        if h.n:
            targets,diff,probs=learned.parameters_for(h,b.graph,b.timestamp);entropy.extend((-probs*np.log(np.maximum(probs,1e-12))).sum(1).tolist());mode_weight.extend(probs.min(1).tolist())
            separation.extend(np.mean([np.linalg.norm(targets[:,i]-targets[:,j],axis=1) for i,j in ((0,1),(0,2),(1,2))],axis=0).tolist())
    result={'ghost_population':{'root_id':target,'offline_actual_count':int(root_data(folder)['truth'].shape[1]),'replay':count},
            'observation_segmentation':segmentation,'one_native_step':{name:scene_ci([r for r in records if r['model']==name],'energy_score') for name in models},
            'one_step_scope':'teacher-conditioned legal-root one native transition, offline-aligned originally-visible targets; not privileged state inputs',
            'mode_diagnostics':{'mean_entropy_nats':float(np.mean(entropy)),'mean_minimum_mode_probability':float(np.mean(mode_weight)),'mean_target_pair_separation_mps':float(np.mean(separation)),'scope':str(len({r['root_id'] for r in records}))+' development roots with native-step supervision; parameter multimodality, not behavioral mode calibration'},
            'source_revision':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip()}
    pathlib.Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('snapshot');p.add_argument('checkpoint');p.add_argument('population');p.add_argument('output');run(**vars(p.parse_args()))
