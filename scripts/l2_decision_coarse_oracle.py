"""Decision-sufficiency witness under a declared finite optical/physical law.

Exact left/right center events are atoms of this synthetic joint-world law.
The oracle reweights those atoms; it does not refine a sampled Gaussian carrier
with an arbitrary continuous detection. Physical conflict here is a declared
geometric proxy, NOT native articulated contact or official geodesic PSC.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from eventwm_graph.l2_partial.state import Hypothesis,ObservedGraph,EXECUTOR_VERSION
from eventwm_graph.l2_partial.observation import EndpointObservationLaw
from eventwm_graph.l2_partial.structural_model import StructuralGraphBeliefWorldModel
from eventwm_graph.l2_partial.structural_state import StructuralBeliefState
from eventwm_graph.l2_partial.planning import packet_from_batch,pessimistic_scores
from eventwm_graph.partial.map import PartialGraph
from eventwm_graph.partial.query import compile_query
from eventwm_graph.l2_partial.repair_profile import _publish,_sources,_sha


class FiniteOptics(EndpointObservationLaw):
    version='synthetic_finite_two_regime_02_08_1'
    def detection(self,p,robot_pose,regime):
        p=np.asarray(p);r=np.asarray(regime);prob=np.asarray([.2,.8])[r]
        if r.ndim:prob=prob.reshape(r.shape+(1,)*(p.ndim-1-r.ndim))
        return np.where(np.isfinite(p[...,0]),np.broadcast_to(prob,p.shape[:-1]),0.)


class ProxyConflict:
    def hazards(self,features):
        result=np.zeros(features.shape[:-1]+(3,))
        result[...,0]=features[...,1]>math.exp(-.45**2/(2*.4**2))
        result[...,1]=features[...,1]>math.exp(-1.**2/(2*.4**2))
        return result


def run():
    model=StructuralGraphBeliefWorldModel(kind='memory_cv',population_prior=[1.],maneuver_rate=0.,
        observation_law=FiniteOptics(),event_model=ProxyConflict())
    graph=ObservedGraph('finite-decision-oracle',1,.2,0.,{(x,z):(1,0.) for x in range(-20,21) for z in range(-20,21)})
    hs=[]
    for regime,z in enumerate((.8,-.8)):
        hs.append(Hypothesis(.5,np.array([[1.5,z,0.,0.]]),np.zeros((1,4,4)),np.array([-1.]),np.array([0]),np.array([1.]),
            intent=np.zeros((1,2)),diffusion=np.zeros(1),mode=np.zeros(1,int),height=np.zeros(1),visibility_mode=regime))
    belief=StructuralBeliefState(graph,hs,0.,np.zeros(4),model_version=model.model_version,
        evidence={'belief_role':'synthetic_finite_prior','residual_count_support':[1]})
    # A completed one-action query provides a supported next decision endpoint.
    first={'query_id':'look','executor_version':EXECUTOR_VERSION,'waypoints':[[0.,0.,0.]],'final_yaw':10/120,
        'compiled':{'planned_poses':[[0.,0.,0.,0.],[0.,0.,0.,10/120]],'actions':['left'],'graph_revision':1}}
    future=model.predict_shared(belief,.048,16,41)
    result=model.evaluate_queries(belief,[first],future,budget_ms=10000.)
    root_packet=packet_from_batch(belief,[first],result,pointgoal=[3.,0.,0.])
    next_pose=np.array([0.,0.,0.,10/120]);stage0=PartialGraph(.2,0.)
    stage0.cells={k:{'state':s,'height':h,'timestamp':.048,'source':'declared_synthetic_support'} for k,(s,h) in graph.materialize().items()}
    stage0.timestamp=.048;stage0.revision=1
    next_queries=[]
    for i,z in enumerate((.8,-.8)):
        q={'query_id':'route_'+str(i),'executor_version':EXECUTOR_VERSION,'waypoints':[[2.,0.,z]],'final_yaw':0.}
        q['compiled']=compile_query(stage0,q,{'robot_pose':next_pose},max_actions=180)
        if not q['compiled'].get('ok'):raise AssertionError(q['compiled'])
        next_queries.append(q)
    def evaluate(b):
        f=model.predict_shared(b,6.,16,41)
        batch=model.evaluate_queries(b,next_queries,f,budget_ms=10000.,refined_queries=0)
        p=packet_from_batch(b,next_queries,batch,pointgoal=[3.,0.,0.])
        return p,pessimistic_scores(p)
    coarse=[];detect=None
    for branch in root_packet.candidates[0].observation_branches:
        if branch.kind.startswith('survived'):
            p,scores=evaluate(branch.successor.resolve())
            coarse.append({'kind':branch.kind,'probability':branch.lower_mass,'scores':scores.tolist(),
                'choice':int(np.argmax(scores)),'collision_intervals':[c.collision_risk_interval for c in p.candidates],
                'next_action_selectable':branch.successor.can_select_new_query})
            if branch.kind=='survived_detection':detect=(branch,scores)
    branch,coarse_scores=detect
    detection=np.asarray([(.2,.8)[h.visibility_mode] for h in future.worlds])
    exact=[];loss=0.;disagreement=0.
    coarse_choice=int(np.argmax(coarse_scores))
    for regime in (0,1):
        group=np.asarray([h.visibility_mode==regime for h in future.worlds])
        masses=future.weights*detection*group;prob=float(masses.sum())
        conditioned=model.successor_at(future,1,masses,next_pose,refinement='exact supported finite center atom')
        p,scores=evaluate(conditioned);choice=int(np.argmax(scores))
        conditional_probability=prob/branch.lower_mass
        local_loss=float(scores[choice]-scores[coarse_choice])
        loss+=conditional_probability*local_loss
        disagreement+=conditional_probability*bool(choice!=coarse_choice)
        exact.append({'center':[1.5,0.,(.8,-.8)[regime]],'same_coarse_category':'survived_detection',
            'conditional_probability':conditional_probability,'scores':scores.tolist(),'choice':choice,
            'coarse_choice_loss':local_loss})
    return {'schema':'l2_coarse_decision_sufficiency_oracle_1','coarse_branches':coarse,'exact_supported_observations':exact,
        'within_detection_ranking_disagreement_probability':disagreement,'conditional_decision_value_loss':loss,
        'unconditional_detection_decision_value_loss':branch.lower_mass*loss,
        'finding':'coarse detection is insufficient' if loss>1e-8 else 'no action loss found in this fixture',
        'law':'two stationary one-person states, equal prior; endpoint detection .2/.8; deterministic .45m conflict and1m social proxies',
        'scope':'finite synthetic oracle and actual runtime continuation; no native utility/calibration claim or continuous bridge',
        'source_hashes':_sources(),'script_sha256':_sha(__file__)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);a=parser.parse_args()
    path=Path(a.output).resolve();root=Path(__file__).resolve().parents[1]/'output/l2_decision_ready_20261021'
    if not path.is_relative_to(root) or path.exists():raise ValueError('new decision output required')
    value=run();_publish(path,value);print(json.dumps(value),flush=True)
