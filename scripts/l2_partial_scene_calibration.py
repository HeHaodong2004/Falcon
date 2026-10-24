"""One controlled calibration repair, declared before fresh payload access.

The v4 Brier-optimal intercept reintroduced credible scene-mean PSC
underprediction. Keep the trained epoch9 coefficients and fit only three
intercepts to calibration-in-the-large at the declared equal-scene/root
estimand. No candidate, high-risk bin or administrative positive is hidden.
This forces development marginal means by construction; fresh and conditional
calibration remain empirical questions. No inference-core source is changed.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
import torch
from eventwm_graph.l2_partial.repair_events import (
    load_cache,_json,_sha,_sources,RESIDUAL_HAZARD,EVENT_NAMES,probabilities,
    _nll,_root_groups)


def scene_weights(records):
    counts=collections.Counter(r['root_id'] for r in records)
    scenes=collections.defaultdict(set)
    for r in records:scenes[r['scene_id']].add(r['root_id'])
    return np.array([1./(len(scenes)*len(scenes[r['scene_id']])*counts[r['root_id']])
                     for r in records])


def cached_curves(coefficients,records,head):
    curves=[]
    for r in records:
        x=np.asarray(r['x'],float);n=r['planned_steps']
        base=x@coefficients[head]
        if head:
            hc=expit(np.clip(x@coefficients[0],-30.,30.))*r['allowed'][:,None,0]
            before=np.c_[np.ones(len(x)),np.cumprod(1-hc,axis=1)[:,:-1]] if n else np.empty((len(x),0))
            residual=r['unresolved']*float(np.sum((1-RESIDUAL_HAZARD)**(2*np.arange(n))*RESIDUAL_HAZARD))
        else:before=None;residual=r['unresolved']*(1-(1-RESIDUAL_HAZARD)**n)
        curves.append((base,np.asarray(r['allowed'])[:,head,None],np.asarray(r['weights']),before,residual))
    return curves


def curve_probabilities(curves,head,offset):
    values=[]
    for base,allowed,weights,before,residual in curves:
        hz=expit(np.clip(base+offset,-30.,30.))*allowed
        if head:
            prior=np.c_[np.ones(len(hz)),np.cumprod(1-hz,axis=1)[:,:-1]] if hz.shape[1] else hz
            p=np.sum(before*prior*hz,axis=1)
        else:p=1-np.prod(1-hz,axis=1)
        values.append(float(weights@p+residual))
    return np.array(values)


def fit(cache,prefit,output):
    torch.set_num_threads(1)
    directory=Path(output).resolve().parent
    if (directory/'fresh_design.json').exists() or (directory/'fresh_snapshot').exists():
        raise RuntimeError('Fresh evaluation boundary already crossed; calibration selection is closed')
    records,metadata=load_cache(cache,'dev')
    if metadata['source_sha256']!=_sources():raise ValueError('Frozen feature sources differ')
    parent=json.loads(Path(prefit).read_text());coeff=np.asarray(parent['coefficients'],float)
    start=time.perf_counter();diagnostics=[];offsets=[]
    for head,name in enumerate(EVENT_NAMES):
        eligible=[r for r in records if r['calibration_eligible'][head]]
        if not eligible:raise ValueError('No complete-followup calibration support')
        w=scene_weights(eligible);y=np.array([r['first'][head]>=0 for r in eligible],float)
        curves=cached_curves(coeff,eligible,head)
        before=curve_probabilities(curves,head,0.)
        target=float(w@y)
        objective=lambda delta:float(w@curve_probabilities(curves,head,delta)-target)
        if objective(-12.)>0 or objective(12.)<0:raise ValueError('Marginal target outside supported event family')
        offset=brentq(objective,-12.,12.,xtol=1e-10,rtol=1e-11)
        after=curve_probabilities(curves,head,offset);coeff[head,0]+=offset
        # Independent scalar runtime helper, rather than trusting the fit curve.
        reference=np.array([probabilities(coeff,r)[head] for r in eligible])
        np.testing.assert_allclose(reference,after,atol=2e-12,rtol=1e-12)
        diagnostics.append({'event':name,'queries':len(eligible),'roots':len({r['root_id'] for r in eligible}),
            'scenes':len({r['scene_id'] for r in eligible}),'observed_scene_mean':target,
            'prefit_scene_mean':float(w@before),'calibrated_scene_mean':float(w@after),
            'prefit_scene_brier':float(w@np.square(before-y)),
            'calibrated_scene_brier':float(w@np.square(after-y)),
            'bias':float(w@(after-y)),'offset':float(offset),
            'excluded_administrative_queries':len(records)-len(eligible)})
        offsets.append(float(offset));print(json.dumps(diagnostics[-1]),flush=True)
    value={**parent,'schema':'partial_graph_event_scene_calibration_1','calibrated':True,
        'coefficients':coeff.tolist(),'intercept_offsets':offsets,'calibration':diagnostics,
        'criterion':'equal-scene average of root-mean predicted probability equals native observed event frequency; one scalar intercept per head',
        'calibration_limitation':'development marginal bias is zero by construction; no claim of fresh or high-risk-stratum calibration',
        'controlled_revision':'v4 Brier calibration produced PSC bias -0.02010 [-0.03827,-0.00420] at a different scene-weighted evaluation estimand; no dynamics or non-intercept coefficient changed',
        'scene_calibration_parent_sha256':_sha(prefit),'dev_cache_sha256':metadata['cache_sha256'],
        'scene_calibration_script_sha256':_sha(__file__),'source_sha256':_sources(),
        'final_dev_censored_nll':_nll(torch.tensor(coeff,dtype=torch.float32),_root_groups(records),4),
        'scene_calibration_elapsed_s':time.perf_counter()-start,'fresh_payload_opened':False}
    _json(output,value);return value


def selftest():
    class Tests(unittest.TestCase):
        def test_scene_root_query_weighting(self):
            rows=[{'root_id':r,'scene_id':s} for r,s in [('a','x'),('a','x'),('b','x'),('c','y')]]
            np.testing.assert_array_equal(scene_weights(rows),[.125,.125,.25,.5])
        def test_one_endpoint_mean_target_is_exact(self):
            curves=[(np.zeros((1,1)),np.ones((1,1)),np.ones(1),None,0.)]
            offset=brentq(lambda b:curve_probabilities(curves,0,b)[0]-.3,-12,12)
            self.assertAlmostEqual(offset,math.log(.3/.7),places=10)
        def test_zero_exposure_has_zero_risk_for_all_intercepts(self):
            c=[(np.zeros((2,0)),np.ones((2,1)),np.array([.3,.7]),None,0.)]
            for offset in [-12.,0.,12.]:self.assertEqual(curve_probabilities(c,0,offset)[0],0.)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest',action='store_true');parser.add_argument('--cache');parser.add_argument('--prefit');parser.add_argument('--output')
    args=parser.parse_args()
    if args.selftest:selftest()
    else:fit(args.cache,args.prefit,args.output)
