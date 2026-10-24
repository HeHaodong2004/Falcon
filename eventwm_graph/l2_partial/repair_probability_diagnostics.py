"""Bounded probability diagnostics, separate from the frozen inference core.

This module never runs on import, never opens holdout payloads, and never
overwrites an artifact. Synthetic identities, finite-world identities, and
recursive API approximation errors are deliberately reported separately.

Example (run only when one research CPU slot is available):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m \
    eventwm_graph.l2_partial.repair_probability_diagnostics NEW_OUTPUT.json \
    --snapshot SNAPSHOT --checkpoint DYNAMICS --population POPULATION.json \
    --observation-checkpoint OPTICS.json --event-checkpoint EVENTS.json

The ghost diagnostic replays the ACTUAL history_filter used by initialize;
offline population is read only after replay and never passed to the model.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import gzip
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from .hidden import missed_hidden_moments, prepare_prior
from .observation import EndpointObservationLaw
from .queries import NativeEventModel, _support, event_features
from .repair_model import RepairedGraphBeliefWorldModel
from .state import BeliefState, Hypothesis, ObservedGraph

SCHEMA = "partial_graph_probability_repair_diagnostics_1"
GHOST_ROOT = "7415a33faee8c8a4501d51a76a230a4c8fea97f3f4bbaf460a85147069c4efc6"
DT = .048
FUNCTIONAL_NAMES = ("population", "relative_x_intensity", "relative_z_intensity",
                    "relative_squared_distance_intensity", "nearby_gaussian_intensity")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    package = Path(__file__).parent
    names = ('data.py','model.py','state.py','probability.py','neural.py','queries.py',
             'maneuver.py','observation.py','repair_filter.py','repair_model.py','hidden.py',
             'repair_probability_diagnostics.py')
    return {name:_sha(package/name) for name in names}


def _read(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def _publish(path, value, text=False):
    path = Path(path).resolve()
    if "l2_partial_graph_wm_explore_20261021" in path.parts:
        raise ValueError("historical research outputs are immutable")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(item):
        if isinstance(item,np.ndarray):return item.tolist()
        if isinstance(item,np.generic):return item.item()
        raise TypeError('unsupported diagnostic JSON type '+type(item).__name__)
    payload = value.encode() if text else json.dumps(value, indent=2, allow_nan=False,default=convert).encode()
    fd, temporary = tempfile.mkstemp(prefix=".probability_repair_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _oracles():
    test_dir = Path(__file__).resolve().parents[2] / "tests" / "l2_partial"
    if str(test_dir) not in sys.path:
        sys.path.insert(0, str(test_dir))
    return importlib.import_module("test_observation_repair")


def synthetic_segmentation():
    """All four histories, actual updates, coarse unions, and prior recovery."""
    o = _oracles(); records = []
    alpha = np.asarray(o.ConstantLaw.regime_prior); pd = np.asarray(o.ConstantLaw.in_view)
    for start in (0., .013):
        model, belief, query = o._wm(), o._belief(start), o._query()
        direct = o._branches(model, belief, query, .096)
        first = o._branches(model, belief, query, .048)
        histories = []; recovered = np.zeros(2); any_mass = np.zeros(2)
        for d1, d2 in itertools.product((0, 1), repeat=2):
            branch1 = first[d1]
            branch2 = o._branches(model, branch1.successor, query, .048)[d2]
            probability = branch1.probability * branch2.probability
            raw = alpha * (pd if d1 else 1-pd) * (pd if d2 else 1-pd)
            actual = model.observe(belief, o._observation(start + .048, d1,
                                    query['compiled']['planned_poses'][1]))
            actual = model.observe(actual, o._observation(start + .096, d2,
                                    query['compiled']['planned_poses'][2]))
            posterior = o._regimes(branch2.successor)
            recovered += probability * posterior
            if d1 or d2:
                any_mass += probability * posterior
            histories.append({"history": ''.join('D' if d else 'N' for d in (d1, d2)),
                "recursive_probability": probability, "oracle_probability": float(raw.sum()),
                "recursive_regime_posterior": posterior.tolist(),
                "exact_observation_regime_posterior": o._regimes(actual).tolist(),
                "oracle_regime_posterior": (raw/raw.sum()).tolist(),
                "probability_absolute_error": float(abs(probability-raw.sum())),
                "posterior_max_error": float(np.max(np.abs(posterior-raw/raw.sum()))),
                "exact_posterior_max_error": float(np.max(np.abs(o._regimes(actual)-raw/raw.sum()))),
                "successor_timestamp": branch2.successor.timestamp})
        direct_recovery = sum(br.probability * o._regimes(br.successor) for br in direct.values())
        records.append({"start_time": start, "histories": histories,
            "sum_history_probability": sum(r['recursive_probability'] for r in histories),
            "recursive_prior_reconstruction_l1": float(np.abs(recovered-alpha).sum()),
            "direct_prior_reconstruction_l1": float(np.abs(direct_recovery-alpha).sum()),
            "coarse_any_probability_error": float(abs(direct[1].probability-any_mass.sum())),
            "coarse_any_posterior_l1": float(np.abs(o._regimes(direct[1].successor)-any_mass/any_mass.sum()).sum())})
    return {"scope": "Known two-regime stationary one-human law, zero events and zero successor regularization; numerical exactness only in this oracle",
            "records": records}


def quadrature_errors(law, reference_order=151, check_order=251):
    """Measure the actual configured rule against two finer positive rules."""
    o = _oracles(); rows = []
    for variance in (.001, .01, .1, 1., 4., 16., 100.):
        h = o._hypothesis(regime=1, position=(2., 1.8)).fork(
            covariance=np.diag([variance, variance, .25, .25])[None])
        mass, mean, covariance = law.missed_moments(h, np.zeros(4))
        reference, rm, rc = missed_hidden_moments(h.mean, h.covariance, np.zeros(4), 1, law,
                                                 order=reference_order)
        check, _, _ = missed_hidden_moments(h.mean, h.covariance, np.zeros(4), 1, law,
                                           order=check_order)
        rows.append({"position_variance": variance, "configured_mass": float(mass[0]),
            "reference_mass": float(reference[0]), "check_mass": float(check[0]),
            "mass_absolute_error": float(abs(mass[0]-reference[0])),
            "mean_l2_error_m": float(np.linalg.norm(mean-rm)),
            "covariance_frobenius_error": float(np.linalg.norm(covariance-rc)),
            "reference_check_mass_difference": float(abs(reference[0]-check[0]))})
    mean, covariance = prepare_prior(np.zeros(4))
    coarse, _, _ = missed_hidden_moments(mean, covariance, np.zeros(4), 1, law)
    reference, _, _ = missed_hidden_moments(mean, covariance, np.zeros(4), 1, law,
                                         order=reference_order)
    return {"reference_order": reference_order, "check_order": check_order,
        "law_parameters": {k: getattr(law, k) for k in ('in_view', 'out_view', 'regime_prior')},
        "tracked_misses": rows, "broad_hidden_prior_miss": {
            "order15": float(coarse[0]), "reference": float(reference[0]),
            "absolute_error": float(abs(coarse[0]-reference[0]))},
        "qualification": "Finer quadrature is a convergence reference, not an exact ground-truth posterior or universal error bound."}


def _allowed_rows(snapshot, cohort=None):
    manifest = json.loads((Path(snapshot) / 'manifest.json').read_text())
    rows = [r for r in manifest['rows'] if r['split'] == 'dev']
    if cohort:
        value = json.loads(Path(cohort).read_text())
        requested = value.get('rows', value.get('root_ids')) if isinstance(value, dict) else value
        ids = {r['root_id'] if isinstance(r, dict) else r for r in requested}
        if not ids <= {r['root_id'] for r in rows}:
            raise ValueError('probability diagnostics allow original development roots only')
        rows = [r for r in rows if r['root_id'] in ids]
    return rows


def _population(belief, true_count=None):
    size = max([1] + [h.n + len(h.hidden_pmf) for h in belief.hypotheses])
    pmf = np.zeros(size)
    for h in belief.hypotheses:
        pmf[h.n:h.n+len(h.hidden_pmf)] += h.weight*h.hidden_pmf
    mass = float(pmf.sum())
    result = {"resolved_pmf": pmf.tolist(), "resolved_mass": mass,
        "unresolved_mass": belief.unresolved_mass,
        "conditional_expected_total": float(pmf @ np.arange(size)/mass) if mass else None,
        "conditional_localized_count": sum(h.weight*h.n for h in belief.hypotheses)/mass if mass else None,
        "maximum_slots": max((h.n for h in belief.hypotheses), default=0)}
    if true_count is not None:
        truth_mass = float(pmf[true_count]) if true_count < size else 0.
        result.update(true_count_probability_lower=truth_mass,
            true_count_probability_upper=min(1., truth_mass+belief.unresolved_mass),
            conditional_probability_above_truth=float(pmf[true_count+1:].sum()/mass) if mass else None)
    return result


def ghost_replay(model, snapshot):
    """Actual history-filter internals, with raw legal detections and no IDs."""
    metadata = next((r for r in _allowed_rows(snapshot) if r['root_id'] == GHOST_ROOT), None)
    if metadata is None:
        return {"status": "unavailable in original development split", "root_id": GHOST_ROOT}
    folder = Path(snapshot)/'roots'/GHOST_ROOT
    legal = _read(folder/'legal_root.json.gz'); history = legal['history']
    model._initial_legal_pose = np.asarray(history[0]['robot_pose'], float)
    actual_filter = model.history_filter()
    graph = ObservedGraph('empty', 0, float(legal['partial_graph']['resolution']),
                          float(legal['partial_graph']['height']), {})
    h = Hypothesis(1., np.empty((0,4)), np.empty((0,4,4)), np.empty(0), np.empty(0,int),
                   model.population_prior.copy())
    belief = BeliefState(graph, model.initial_hypotheses(h), float(history[0]['timestamp']),
                        np.asarray(history[0]['robot_pose'],float), model_version=actual_filter.model_version)
    records = []
    for frame, row in enumerate(history):
        prior = actual_filter.propagate(belief, float(row['timestamp']))
        detections = sorted([r for r in row.get('tracks',[]) if r.get('visible_now') and r.get('valid',True)],
                            key=lambda r: tuple(r['position']))
        xy = np.asarray([r['position'] for r in detections],float).reshape(-1,3)[:,[0,2]]
        innovations = []
        for hi, h in enumerate(prior.hypotheses):
            for i, slot in enumerate(h.slot_ids):
                for j, z in enumerate(xy):
                    conditional, log_likelihood = actual_filter.condition_track(h,i,z)
                    delta = z-h.mean[i,:2]
                    innovations.append({"prior_component":hi,"prior_weight":h.weight,
                        "slot":int(slot),"detection_index":j,"age_s":float(prior.timestamp-h.last_seen[i]),
                        "residual_m":float(np.linalg.norm(delta)),
                        "gaussian_summary_chi2":float(delta @ np.linalg.pinv(h.covariance[i,:2,:2]+np.eye(2)*1e-12) @ delta),
                        "maneuver_mixture_log_position_likelihood":float(log_likelihood) if np.isfinite(log_likelihood) else None,
                        "supported":conditional is not None})
        belief = actual_filter.observe(belief,row)
        evidence = belief.evidence
        records.append({"frame":frame,"timestamp":belief.timestamp,"visible_count":len(detections),
            "detections_xz":xy.tolist(),"prior_population":_population(prior),
            "posterior_population":_population(belief),
            "birth_probability":evidence.get('birth_probability'),
            "pruned_posterior_mass":evidence.get('pruned_mass'),
            "unsupported_observation":evidence.get('unsupported_observation'),
            "assignment_evidence_exact":evidence.get('assignment_evidence_exact'),
            "association_alternatives":evidence.get('association_alternatives',[]),
            "association_diagnostics":evidence.get('association_diagnostics',[]),
            "prior_innovations":innovations,
            "posterior_slots":[{"weight":h.weight,"slot_ids":h.slot_ids.tolist(),
                "last_seen":h.last_seen.tolist(),"positions_xz":h.mean[:,:2].tolist(),
                "birth_time":h.birth_time.tolist() if h.birth_time is not None else None,
                "birth_position":h.birth_position.tolist() if h.birth_position is not None else None}
                for h in belief.hypotheses]})
    # Offline population truth is read AFTER all filtering, solely to score it.
    with np.load(folder/'privileged_future.npz') as archive:
        true_count = int(archive['positions'].shape[1])
    for record in records:
        pmf = record['posterior_population']['resolved_pmf']; mass = sum(pmf)
        record['posterior_population']['conditional_probability_above_offline_truth'] = sum(pmf[true_count+1:])/mass if mass else None
    initialized = model.initialize(history,legal['partial_graph'])
    return {"root_id":GHOST_ROOT,"scene_id":metadata['scene_id'],"split":"dev",
        "frames":len(history),"offline_population_after_inference":true_count,
        "history_filter_model_version":actual_filter.model_version,"online_model_version":model.model_version,
        "trace":records,"final_history_count":_population(belief,true_count),
        "final_public_initialize_count":_population(initialized,true_count),
        "history_vs_initialize_count_pmf_l1":float(np.abs(np.pad(np.asarray(_population(belief)['resolved_pmf']),
            (0,max(0,len(_population(initialized)['resolved_pmf'])-len(_population(belief)['resolved_pmf']))))-
            np.pad(np.asarray(_population(initialized)['resolved_pmf']),
            (0,max(0,len(_population(belief)['resolved_pmf'])-len(_population(initialized)['resolved_pmf']))))).sum()),
        "ancestry_scope":"Internal anonymous slot birth/association lineage; source tracker IDs are neither used nor treated as true identities. No retrospective future matching.",
        "innovation_scope":"Full maneuver-mixture position likelihood and Gaussian-summary residual chi2 are distinct diagnostics; chi2 is not a calibrated mixture test."}


def _functionals(belief):
    """Gaussian moments and a smooth spatial probe; none is a collision label."""
    result = np.zeros(len(FUNCTIONAL_NAMES)); robot = belief.robot_pose[[0,2]]
    for h in belief.hypotheses:
        groups = [(h.mean,h.covariance,1.)]
        if h.hidden_mean is not None:
            groups.append((h.hidden_mean,h.hidden_covariance,h.expected_hidden if h.hidden_exchangeable else 1.))
        for mean,covariance,multiplier in groups:
            if not len(mean):continue
            delta = mean[:,:2]-robot; p = covariance[:,:2,:2]
            matrix = p + np.eye(2)*.64
            smooth = .64/np.sqrt(np.linalg.det(matrix))*np.exp(-.5*np.einsum('ni,nij,nj->n',delta,np.linalg.inv(matrix),delta))
            result += h.weight*multiplier*np.array([len(mean),delta[:,0].sum(),delta[:,1].sum(),
                (np.sum(delta*delta,axis=1)+np.trace(p,axis1=1,axis2=2)).sum(),smooth.sum()])
    return result


def fixed_path_factorization(model, belief, query, total_steps, seed, worlds):
    future = model.predict_shared(belief,total_steps*DT,num_worlds=worlds,seed=seed)
    poses = np.asarray(query['compiled']['planned_poses'],float)[:total_steps+1]
    indices = np.searchsorted(future.times,belief.timestamp+np.arange(total_steps+1)*DT-1e-8)
    hidden = np.array([h.expected_hidden for h in future.worlds])
    age = np.mean([max(0.,belief.timestamp-h.last_seen.mean()) if h.n else 0. for h in future.worlds])
    times = np.arange(total_steps+1)*DT
    features = event_features(future.positions[:,indices],future.velocities[:,indices],poses,hidden,age+times,times)
    event = model.event_model or NativeEventModel()
    hazards = event.hazards(features)[:,1:,0]
    # Match the executor's physical empty-population constraint as well as the
    # fitted hazard. Without this mask the diagnostic invents empty-world risk.
    hazards[np.asarray([h.n for h in future.worlds])+hidden==0] = 0.
    per_survival = 1-hazards
    survival = per_survival.prod(1)
    no = model.observation_law.no_detection(future,indices,poses)
    prediction = model.evaluate_queries(belief,[query],future).predictions[0]
    api = {b.observation['kind']:b.probability for b in prediction.observation_branches}
    rows = []
    for depth in (1,2,3):
        chunks = np.array_split(np.arange(total_steps),depth)
        segment_no = []; segment_survival = []
        for chunk in chunks:
            start,end = int(chunk[0]),int(chunk[-1]+1)
            segment_no.append(model.observation_law.no_detection(future,indices[start:end+1],poses[start:end+1]))
            segment_survival.append(per_survival[:,start:end].prod(1))
        recovered = np.zeros_like(future.weights); records = []
        for bits in itertools.product((0,1),repeat=depth):
            likelihood = survival.copy()
            for bit,none in zip(bits,segment_no):likelihood *= 1-none if bit else none
            unnormalized = future.weights*likelihood; mass = float(unnormalized.sum())
            posterior = unnormalized/mass if mass else np.zeros_like(unnormalized)
            recovered += mass*posterior
            records.append({"history":''.join('D' if b else 'N' for b in bits),"mass":mass,
                            "posterior_world_weights":posterior.tolist()})
        terminal = future.weights*(1-survival)
        rows.append({"depth":depth,"history_probabilities_and_posteriors":records,
            "no_detection_factorization_max_error":float(np.max(np.abs(np.prod(segment_no,axis=0)-no))),
            "survival_factorization_max_error":float(np.max(np.abs(np.prod(segment_survival,axis=0)-survival))),
            "posterior_reconstruction_l1_including_terminal":float(np.abs(recovered+terminal-future.weights).sum()),
            "survival_posterior_reconstruction_l1":float(np.abs(recovered-future.weights*survival).sum())})
    return {"seed":seed,"worlds":len(future.worlds),"unresolved_mass":future.unresolved_mass,
        "physical_horizon_s":total_steps*DT,"rows":rows,
        "public_query_no_detection_error":float(abs(api.get('survived_no_detection',0.)-future.weights@(survival*no))),
        "public_query_detected_error":float(abs(api.get('survived_detection',0.)-future.weights@(survival*(1-no)))),
        "qualification":"Exact arithmetic identity for these fixed sampled physical paths and this event/optical law; not an accuracy test or proof for the continuous posterior."}


def recursive_api(model, belief, query, total_steps, depth, seed, worlds):
    """Bounded binary observation tree, with every omitted branch recorded."""
    nodes = [(1.,belief,0,False)]; terminal = unresolved = no_detection = 0.
    functional = np.zeros(len(FUNCTIONAL_NAMES)); survival = 0.; calls = 0
    started = time.perf_counter(); chunks = np.array_split(np.arange(total_steps),depth)
    for stage,chunk in enumerate(chunks):
        next_nodes = []
        for mass,current,history,ever in nodes:
            node_seed = int(seed+1009*stage+2039*history)
            future = model.predict_shared(current,len(chunk)*DT,num_worlds=worlds,seed=node_seed)
            prediction = model.evaluate_queries(current,[query],future).predictions[0]; calls += 1
            accounted = 0.
            for branch in prediction.observation_branches:
                joint = mass*branch.probability; accounted += branch.probability
                kind = branch.observation['kind']
                if kind.startswith('unresolved_'):
                    unresolved += joint
                elif branch.successor.terminal:
                    terminal += joint
                elif kind in ('survived_no_detection','survived_detection'):
                    detected = kind=='survived_detection'
                    next_nodes.append((joint,branch.successor,2*history+int(detected),ever or detected))
                else:
                    raise ValueError('unexpected supported-query branch '+kind)
            if abs(accounted-1.) > 1e-7:
                raise ValueError('query branches failed probability accounting')
        nodes = next_nodes
    for mass,posterior,_,ever in nodes:
        survival += mass; no_detection += mass*(not ever)
        functional += mass*_functionals(posterior)
    if abs(survival+terminal+unresolved-1.) > 1e-7:
        raise ValueError('recursive tree lost probability mass')
    return {"depth":depth,"seed":seed,"calls":calls,"wall_ms":(time.perf_counter()-started)*1000,
        "physical_horizon_s":total_steps*DT,"resolved_survival":survival,
        "resolved_terminal":terminal,"unresolved_stopped_branch_mass":unresolved,
        "resolved_survived_no_detection":no_detection,
        "resolved_survived_any_detection":survival-no_detection,
        "survival_weighted_functionals":dict(zip(FUNCTIONAL_NAMES,map(float,functional))),
        "survival_conditional_functionals":dict(zip(FUNCTIONAL_NAMES,map(float,functional/survival))) if survival else None}


def _paired_summary(records):
    keys = ('resolved_survival','resolved_terminal','unresolved_stopped_branch_mass',
            'resolved_survived_no_detection','resolved_survived_any_detection')
    result = []
    for regularization in sorted({r['regularization'] for r in records}):
        base = {(r['root_id'],r['seed']):r for r in records if r['regularization']==regularization and r['depth']==1}
        for depth in (2,3):
            deltas = []
            for r in records:
                if r['regularization']!=regularization or r['depth']!=depth:continue
                ref = base[(r['root_id'],r['seed'])]
                deltas.append({**{k:r[k]-ref[k] for k in keys},
                    **{'survival_functional_'+k:r['survival_weighted_functionals'][k]-ref['survival_weighted_functionals'][k]
                       for k in FUNCTIONAL_NAMES},'root_id':r['root_id'],'seed':r['seed']})
            metrics = {}
            for key in (*keys,*('survival_functional_'+k for k in FUNCTIONAL_NAMES)):
                values = np.asarray([r[key] for r in deltas])
                roots = sorted({r['root_id'] for r in deltas})
                within = [np.std([r[key] for r in deltas if r['root_id']==root],ddof=1)
                          for root in roots if sum(r['root_id']==root for r in deltas)>1]
                metrics[key] = {'mean_signed_difference':float(values.mean()) if len(values) else None,
                    'mean_absolute_difference':float(np.abs(values).mean()) if len(values) else None,
                    'maximum_absolute_difference':float(np.abs(values).max()) if len(values) else None,
                    'mean_within_root_seed_sd':float(np.mean(within)) if within else None}
            result.append({'regularization':regularization,'depth_vs_one':depth,'paired_replicates':len(deltas),
                           'metrics':metrics})
    return result


def _speed_summary(speed, weight):
    speed, weight = np.asarray(speed).ravel(), np.asarray(weight).ravel()
    keep = np.isfinite(speed) & (weight>0); speed,weight = speed[keep],weight[keep]
    if not len(speed):return {'samples':0,'mean_mps':None,'p95_mps':None,'maximum_mps':None}
    weight = weight/weight.sum(); order = np.argsort(speed)
    index = min(int(np.searchsorted(np.cumsum(weight[order]),.95)),len(speed)-1)
    return {'samples':len(speed),'mean_mps':float(weight@speed),'p95_mps':float(speed[order[index]]),
        'maximum_mps':float(speed.max()),'probability_above_3mps':float(weight@(speed>3.)),
        'probability_above_5mps':float(weight@(speed>5.))}


def speed_diagnostics(model,belief,folder,seeds,worlds):
    """Model speeds first; producer truth opens only after all predictions."""
    horizons = (.5,1.,2.,3.); endpoints = np.ceil(np.asarray(horizons)/DT-1e-9).astype(int)
    # A fixed extra native endpoint permits offline comparison to the actual
    # timestamp interval containing each target. Its duration is not truth-fed.
    horizon = float((endpoints.max()+1)*DT)
    times = np.unique(np.r_[np.arange(endpoints.max()+2)*DT,horizons])
    records = []; sampled = []
    for seed in seeds:
        future = model.predict_shared(belief,horizon,num_worlds=worlds,seed=seed,times=times)
        sampled.append((seed,future))
        weights = np.broadcast_to(future.weights[:,None],future.velocities[:,0,:,0].shape)
        for target,endpoint in zip(horizons,endpoints):
            index = int(np.searchsorted(future.times,belief.timestamp+target-1e-8))
            lo = int(np.searchsorted(future.times,belief.timestamp+(endpoint-1)*DT-1e-8))
            hi = int(np.searchsorted(future.times,belief.timestamp+endpoint*DT-1e-8))
            instant = np.linalg.norm(future.velocities[:,index],axis=-1)
            difference = np.linalg.norm(future.positions[:,hi]-future.positions[:,lo],axis=-1)/DT
            mass = float(future.weights.sum())
            records.append({'seed':seed,'horizon_s':target,'native_window_relative_s':[(endpoint-1)*DT,endpoint*DT],
                'unresolved_mass':future.unresolved_mass,
                'all_instantaneous':_speed_summary(instant,weights),
                'tracked_instantaneous':_speed_summary(instant,weights*future.localized_mask),
                'hidden_instantaneous':_speed_summary(instant,weights*~future.localized_mask),
                'all_native_window_difference':_speed_summary(difference,weights),
                'resolved_world_probability_any_above_3mps':float(future.weights@np.any(instant>3.,axis=1)/mass) if mass else None,
                'resolved_world_probability_any_above_5mps':float(future.weights@np.any(instant>5.,axis=1)/mass) if mass else None})
    # Evaluation-only target; never supplied to initialize, observe or forecast.
    with np.load(Path(folder)/'privileged_future.npz') as archive:
        truth_times = archive['timestamps'].copy(); truth = archive['positions'][...,[0,2]].copy()
    actual = []
    for target,endpoint in zip(horizons,endpoints):
        absolute = belief.timestamp+target
        index = int(np.searchsorted(truth_times,absolute-1e-8))
        if index<=0 or index>=len(truth_times):
            actual.append({'horizon_s':target,'status':'target outside available actual future'});continue
        dt = float(truth_times[index]-truth_times[index-1])
        speed = np.linalg.norm(truth[index]-truth[index-1],axis=-1)/dt
        comparison = []
        for seed,future in sampled:
            points = []
            for t in (truth_times[index-1],truth_times[index]):
                if t<future.times[0]-1e-8 or t>future.times[-1]+1e-8:break
                hi = min(max(int(np.searchsorted(future.times,t)),1),len(future.times)-1)
                alpha = (t-future.times[hi-1])/(future.times[hi]-future.times[hi-1])
                points.append((1-alpha)*future.positions[:,hi-1]+alpha*future.positions[:,hi])
            if len(points)==2:
                velocity = np.linalg.norm(points[1]-points[0],axis=-1)/dt
                comparison.append({'seed':seed,'forecast_same_actual_window_speed':_speed_summary(
                    velocity,np.broadcast_to(future.weights[:,None],velocity.shape))})
        actual.append({'horizon_s':target,'native_interval_absolute_s':[float(truth_times[index-1]),float(truth_times[index])],
            'actual_native_window_speed':_speed_summary(speed,np.ones_like(speed)),
            'forecast_same_actual_window':comparison})
    return {'producer_declared_motion_bound_mps':3.,'additional_extreme_threshold_mps':5.,
        'model_records':records,'offline_truth':actual,
        'qualification':'Descriptive predictive person-speed distribution conditional on retained mass, separately by tracked/hidden population. World exceedance accounts for dependence. Actual-native windows contain the target time and may differ from nominal .048s endpoints; forecast positions are interpolated offline from an already-generated fixed schedule to compare that same physical interval. Instantaneous latent velocity is a different estimand. Finite sample maximum is not a support bound. No clipping, tuning, or online use of truth.'}


def real_diagnostics(models, snapshot, cohort, limit, seeds, worlds, max_steps):
    rows = _allowed_rows(snapshot,cohort); selected = []; skipped = []; fixed = []; recursive = []; speeds = []
    for meta in rows:
        if len(selected)>=limit:break
        folder = Path(snapshot)/'roots'/meta['root_id']
        legal = _read(folder/'legal_root.json.gz'); queries = _read(folder/'queries.json.gz')
        main = models['configured']; belief = main.initialize(legal['history'],legal['partial_graph'])
        eligible = [q for q in queries if len(q['compiled']['actions'])>=3 and _support(main,belief.graph,q)]
        if not eligible:
            skipped.append({'root_id':meta['root_id'],'reason':'no certified legal query with at least three native actions'});continue
        query = max(eligible,key=lambda q:len(q['compiled']['actions']))
        total = min(max_steps,len(query['compiled']['actions']))
        selected.append({'root_id':meta['root_id'],'scene_id':meta['scene_id'],'query_id':query['query_id'],
                         'native_steps':total,'graph_cells':len(belief.graph.materialize())})
        for seed in seeds:
            fixed.append({'root_id':meta['root_id'],**fixed_path_factorization(main,belief,query,total,seed,worlds)})
        speeds.append({'root_id':meta['root_id'],**speed_diagnostics(main,belief,folder,seeds,worlds)})
        for name,model in models.items():
            initial = belief if name=='configured' else model.initialize(legal['history'],legal['partial_graph'])
            for seed in seeds:
                for depth in (1,2,3):
                    recursive.append({'root_id':meta['root_id'],'scene_id':meta['scene_id'],'regularization':name,
                        **recursive_api(model,initial,query,total,depth,seed,worlds)})
        print(json.dumps({'probability_diagnostic_root_complete':len(selected),'root_id':meta['root_id'],
                          'native_steps':total}),flush=True)
    regularization_effect = []
    paired = {(r['root_id'],r['seed'],r['depth']):r for r in recursive if r['regularization']=='zero'}
    for r in recursive:
        if r['regularization']!='configured':continue
        z = paired[(r['root_id'],r['seed'],r['depth'])]
        regularization_effect.append({'root_id':r['root_id'],'seed':r['seed'],'depth':r['depth'],
            'resolved_terminal_difference':r['resolved_terminal']-z['resolved_terminal'],
            'unresolved_difference':r['unresolved_stopped_branch_mass']-z['unresolved_stopped_branch_mass'],
            'survival_functional_difference':{k:r['survival_weighted_functionals'][k]-z['survival_weighted_functionals'][k]
                                               for k in FUNCTIONAL_NAMES}})
    return {'selection':'First permitted development roots with a certified query >=3 actions; choose longest legal query, cap native duration before predictions.',
        'requested_roots':limit,'selected':selected,'skipped':skipped,'seeds':seeds,'worlds':worlds,
        'fixed_path_factorization':fixed,'recursive_api':recursive,'depth_drift_summary':_paired_summary(recursive),
        'paired_regularization_effect':regularization_effect,'speed_plausibility':speeds,
        'qualification':'All depths cover the same physical interval. Fixed-path identities hold within the sampled measure. API differences combine finite-world sampling/allocation, pruning, Gaussian closure, belief-summary event features, and optional regularization; zero-regularization still has finite-sample error. Seed SD is Monte Carlo variability, not a scene-generalization confidence interval. Unresolved branches are stopped and fully accounted, not expanded using invented physical states. Smooth intensity is a state probe, not native collision risk or decision utility.'}


def _review(result):
    paragraphs = ["# Probability repair diagnostic review", "",
        "This artifact distinguishes exact finite-model identities from measured approximation error. Passing a synthetic oracle does not establish real predictive calibration or readiness for scaling.", ""]
    if 'synthetic_segmentation' in result:
        records = result['synthetic_segmentation']['records']
        worst = max(r['recursive_prior_reconstruction_l1'] for r in records)
        paragraphs += [f"The executed constant-regime segmentation oracle covers NN, ND, DN, and DD, including exact arrived observations. Maximum recursive prior-reconstruction L1 discrepancy: {worst:.8g}.", ""]
    if 'real' in result:
        paragraphs += [f"The real-data consistency study used {len(result['real']['selected'])} original development roots. It compares depths 1/2/3 over identical physical intervals, repeats seeded 16-world-style approximations, and separates zero from configured successor regularization. See the raw per-root records before interpreting aggregate drift.", ""]
        speeds = [r['all_instantaneous'] for root in result['real']['speed_plausibility']
                  for r in root['model_records'] if r['horizon_s']==3. and r['all_instantaneous']['samples']]
        if speeds:
            paragraphs += [f"At 3 seconds, the descriptive mean across root/seed records of retained-person P(speed >3 m/s) is {np.mean([r['probability_above_3mps'] for r in speeds]):.6g}; P(speed >5 m/s) is {np.mean([r['probability_above_5mps'] for r in speeds]):.6g}. The largest sampled speed is {max(r['maximum_mps'] for r in speeds):.6g} m/s. These are dependence-aware simulation summaries, not {len(speeds)} independent roots. They expose physical support mismatch requiring investigation.", ""]
    if 'ghost' in result and 'trace' in result['ghost']:
        g = result['ghost']; count = g['final_public_initialize_count']['conditional_expected_total']
        paragraphs += [f"The named ghost witness was replayed for {g['frames']} legal frames using the actual history filter. Its final resolved conditional expected population is {count}; the offline count, read after filtering, is {g['offline_population_after_inference']}. Slot ancestry describes anonymous internal hypotheses rather than ground-truth identities.", ""]
    paragraphs += ["The following limitations remain scientifically consequential:", "",
        "- The diffuse unresolved contamination law is not a conservative posterior bound for every pruned physical hypothesis; an omitted hypothesis can share a supported atom.",
        "- Finite Gauss-Hermite miss integration and Gaussian moment closure are approximations. The report measures convergence; it does not label the finite rule exact.",
        "- Coarse hypothetical observations expose an event and posterior mixture, rather than a complete synthesized exact-position detection set.",
        "- Recursive API drift combines sampling, allocation, pruning and closure. Exact fixed-path factorization alone cannot establish predictive reliability.",
        "- Native event features include a belief-aggregated track-age statistic. Recursive changes in this statistic can contribute additional event-law approximation drift, beyond physical transition sampling.",
        "- The mean-preserving Gaussian maneuver jump has unbounded velocity support. Reported exceedances of the declared 3 m/s producer bound and the 5 m/s diagnostic threshold must be assessed before claiming physical plausibility.",
        "- Native collision/PSC outputs remain fitted surrogates. The smooth spatial functional here is not a replacement for native event labels.",
        "- Actual-trace survival conditioning requires compatible component likelihoods; these diagnostics do not establish a general automatic trace adapter.",
        "- This report does not establish the latency gate, holdout performance, navigation utility, or promotion. Those require the separate frozen evaluation and full-call measurements.", ""]
    return '\n'.join(paragraphs)


def run(output, snapshot=None, checkpoint=None, population=None, observation_checkpoint=None,
        event_checkpoint=None, cohort=None, section='all', limit=10, seeds=(31,67,103,139),
        worlds=16, max_steps=60, reference_order=151, check_order=251,supersedes=None):
    if limit<1 or max_steps<3 or worlds<1:raise ValueError('positive bounded workload required')
    package = Path(__file__).parent; repo = package.parents[1]
    sources = _source_hashes()
    result = {'schema':SCHEMA,'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'source_hashes':sources,'timestamp_unix':time.time(),'split_scope':'original development only; no holdout payloads',
        'checkpoints':{name:{'path':str(path),'sha256':_sha(path)} for name,path in
            (('dynamics',checkpoint),('population',population),('observation',observation_checkpoint),('events',event_checkpoint)) if path}}
    started = time.perf_counter()
    if supersedes:
        result['supersedes_diagnostic'] = {'path':str(supersedes),'sha256':_sha(supersedes),
            'reason':'Correct diagnostic reference only: zero-population hazard mask; actual rather than nominal-only speed windows. Frozen core, model parameters and selection unchanged; original artifact preserved.'}
    if section in ('all','synthetic'):
        o = _oracles(); result['oracle_helper_sha256'] = _sha(o.__file__)
        result['synthetic_segmentation'] = synthetic_segmentation()
        result['hidden_coarse_exact'] = o.hidden_closure_diagnostic()
        result['retired_hidden_law_failure_witness'] = o.hidden_closure_diagnostic(o.RetiredHiddenFirstReportLaw())
        result['quadrature_default'] = quadrature_errors(EndpointObservationLaw(),reference_order,check_order)
        if observation_checkpoint:
            law = EndpointObservationLaw(json.loads(Path(observation_checkpoint).read_text())['parameters'])
            result['quadrature_fitted'] = quadrature_errors(law,reference_order,check_order)
        print('synthetic_probability_diagnostics_complete',flush=True)
    if section in ('all','real','ghost'):
        if not snapshot or not checkpoint or not population:
            raise ValueError('real/ghost diagnostics require snapshot, frozen dynamics and population artifacts')
        import torch
        from .neural import LearnedDynamics
        torch.set_num_threads(1)
        dynamics = LearnedDynamics.load(checkpoint)
        kwargs = {'dynamics':dynamics,'population_prior':json.loads(Path(population).read_text()),
                  'observation_checkpoint':observation_checkpoint}
        if event_checkpoint:kwargs['event_model'] = NativeEventModel(json.loads(Path(event_checkpoint).read_text())['coefficients'])
        models = {'configured':RepairedGraphBeliefWorldModel(**kwargs),
                  'zero':RepairedGraphBeliefWorldModel(**kwargs,successor_regularization=(0.,0.))}
        result['model_versions'] = {k:m.model_version for k,m in models.items()}
        result['model_configuration'] = {k:{name:getattr(m,name) for name in
            ('kind','max_hypotheses','observation_repair','maneuver_rate','reset_std','maneuver_retention',
             'successor_regularization','event_clock_version')} for k,m in models.items()}
        result['native_event_head_scope'] = 'Frozen supplied checkpoint, identified by hash; event-law drift is conditional on this head and is not a fresh calibration claim.' if event_checkpoint else 'Untrained default native event coefficients; diagnostic only.'
        result['snapshot_manifest_sha256'] = _sha(Path(snapshot)/'manifest.json')
        if section in ('all','ghost'):
            result['ghost'] = ghost_replay(models['configured'],snapshot)
            print('ghost_probability_diagnostics_complete',flush=True)
        if section in ('all','real'):
            result['real'] = real_diagnostics(models,snapshot,cohort,limit,list(seeds),worlds,max_steps)
    result['wall_seconds'] = time.perf_counter()-started
    current = _source_hashes()
    if sources != current:raise RuntimeError('source changed during diagnostic execution; refusing mixed-version artifact')
    result['source_hashes_after'] = current
    _publish(output,result); _publish(str(output)+'.md',_review(result),text=True)
    print(json.dumps({'output':str(output),'wall_seconds':result['wall_seconds'],'sections':list(result)},indent=2))
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output'); p.add_argument('--snapshot'); p.add_argument('--checkpoint'); p.add_argument('--population')
    p.add_argument('--observation-checkpoint'); p.add_argument('--event-checkpoint'); p.add_argument('--cohort')
    p.add_argument('--section',choices=('all','synthetic','real','ghost'),default='all')
    p.add_argument('--limit',type=int,default=10); p.add_argument('--worlds',type=int,default=16)
    p.add_argument('--max-steps',type=int,default=60); p.add_argument('--seeds',type=int,nargs='+',default=[31,67,103,139])
    p.add_argument('--reference-order',type=int,default=151); p.add_argument('--check-order',type=int,default=251)
    p.add_argument('--supersedes',help='Immutable earlier diagnostic artifact with a documented reference error')
    run(**vars(p.parse_args()))
