"""Two-stage L2.5 compiler; all candidates share one immutable joint tape."""
from collections import OrderedDict
import math
import time
import numpy as np

from eventwm_graph.l2_partial.queries import _support
from eventwm_graph.l2_intermittent.observations import condition_observation_event
from eventwm_graph.partial.query import NOMINAL_STEP_SECONDS
from .schema import (CONTRACT, VERSION, CompiledReference, ComputeBudget, ContextGraph,
                     DecisionPacket, SparseConditionalBranch, digest, readonly, immutable)
from .spatial import build_spatial, context_features, SPATIAL_FEATURES
from .numeric import aggregate_joint, optical_batch, finite_event_mi
from .graph import export_graph, QUERY_FEATURES


class LRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key):
        if key not in self.data:
            self.misses += 1
            return None
        self.hits += 1
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        while len(self.data) > self.capacity:
            self.data.popitem(last=False)
        return value

    def clear(self):
        self.data.clear()


class QueryOutcomeCompiler:
    """One compiler per stream/model; caller serializes actual-belief updates.

    L2's actual immutable Belief is the snapshot/reference. Stage A accepts only
    that belief's arrived graph and ego. Stage B never accepts private futures,
    actor identities, future sensor rows, L4 outcomes or labels.
    """
    def __init__(self, model, tile_size=4):
        if not isinstance(tile_size, int) or not 1 <= tile_size <= 16:
            raise ValueError('invalid tile size')
        self.model = model
        self.tile_size = tile_size
        self.caches = {k: LRU(v) for k, v in [('static', 4), ('geometry', 256), ('topology', 4), ('context', 4), ('future', 2), ('numeric', 4), ('branches', 8)]}
        self._actual_version = None
        self.forecast_calls = 0

    def clear_dynamic_caches(self):
        for name, cache in self.caches.items():
            if name not in ('static', 'geometry', 'topology'):
                cache.clear()
        validator = getattr(self.model, '_route_validator', None)
        if validator:
            for name in ('graph_cache', 'support_footprint_cache', 'endpoint_template_cache'):
                c = getattr(validator, name, None)
                if c is not None:
                    c.clear()

    def cache_statistics(self):
        return {k: dict(entries=len(v.data), hits=v.hits, misses=v.misses) for k, v in self.caches.items()}

    def build_context(self, belief_snapshot, legal_graph, ego_state, goal):
        b = belief_snapshot
        self.model._check(b, actual=True)
        if b.terminal:
            raise ValueError('terminal actual beliefs have no legal root candidates')
        if legal_graph is not b.graph:
            raise ValueError('legal_graph must be the arrived graph owned by this belief')
        ego = np.asarray(ego_state, float)
        goal = np.asarray(goal, float)
        if ego.shape != (4,) or not np.isfinite(ego).all() or not np.allclose(ego, b.robot_pose, atol=1e-8, rtol=0):
            raise ValueError('context ego must equal arrived belief pose')
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError('public goal must be finite XYZ meters')
        version = b.fingerprint()
        if self._actual_version != version:
            self.clear_dynamic_caches()
            self._actual_version = version
        key = (self.model.model_version, version, b.graph.namespace, tuple(goal), VERSION)
        cached = self.caches['context'].get(key)
        if cached is not None:
            return cached
        static_key = (b.graph.namespace, b.graph.version, b.graph.resolution, self.tile_size, VERSION)
        spatial = self.caches['static'].get(static_key)
        if spatial is None:
            spatial = self.caches['static'].put(static_key, build_spatial(b.graph, self.tile_size))
        tracks = immutable(self.model.get_previously_observed_tracks(b))
        counts = [sum(c.weight*c.n for c in b.components),
                  sum(c.weight*np.arange(len(c.hidden_pmf))@c.hidden_pmf for c in b.components)]
        metadata = dict(role='actual', expected_tracked_count=float(counts[0]), expected_hidden_count=float(counts[1]),
                        association_entropy=b.diagnostics.get('association_entropy'),
                        population_support='tracked hypotheses plus broad unlocalized prior; no certified absent humans',
                        visibility_support='unsupported; certified static support is not optical visibility',
                        spatial_summary='connected same-state tiled cells, <=2cm seed-height deviation; track means assigned to regions, not occupancy',
                        unknown='explicit external UNKNOWN node has no fabricated position or traversable edge',
                        graph_resolution=b.graph.resolution, static_cells=len(spatial.cell_to_node),
                        model_diagnostics=dict(b.diagnostics), goal_source='caller-supplied public navigation goal')
        context = ContextGraph(b, version, b.graph.namespace, b.model_version, b.timestamp,
                               readonly(ego), readonly(goal), spatial, context_features(spatial, ego, goal, tracks),
                               SPATIAL_FEATURES, tracks, immutable(metadata))
        return self.caches['context'].put(key, context)

    def _future(self, context, budget):
        key = (context.model_version, context.belief_version, context.graph_version,
               context.timestamp, budget.horizon, budget.worlds, budget.seed, VERSION)
        future = self.caches['future'].get(key)
        if future is None:
            self.forecast_calls += 1
            future = self.model.predict_shared(context.belief, budget.horizon, budget.worlds, budget.seed)
            size = sum(a.nbytes for a in (future.positions, future.velocities, future.hidden_positions))
            if size <= 32*1024*1024:  # do not retain oversized banks
                self.caches['future'].put(key, future)
        return future

    def evaluate_candidates(self, context, compiled_queries, compute_budget=None):
        tick = time.perf_counter()
        budget = compute_budget or ComputeBudget()
        b = context.belief
        self.model._check(b, actual=True)
        if context.belief_version != self._actual_version or b.fingerprint() != context.belief_version:
            raise ValueError('stale context after actual observation update')
        if context.model_version != self.model.model_version:
            raise ValueError('context/model version mismatch')
        refs = tuple(compiled_queries)
        if len(refs) > budget.max_candidates:
            raise ValueError('candidate budget exceeded; no silent candidate pruning')
        if any(not isinstance(q, CompiledReference) for q in refs):
            raise TypeError('supply CompiledReference.from_native compatibility views')
        if len({q.query_id for q in refs}) != len(refs):
            raise ValueError('duplicate root query IDs')
        for q in refs:
            q.verify()
            if abs(q.start_timestamp-context.timestamp) > 1e-8 or not np.allclose(q.poses[0], context.ego, atol=1e-8, rtol=0):
                raise ValueError('reference must start at current absolute time and ego')
            if q.graph_revision != b.graph.version:
                raise ValueError('reference graph revision differs; upstream revalidation required')
        if not refs:
            return DecisionPacket(context, (), (), export_graph(context, (), ()),
                                  dict(contract=CONTRACT, status='NO_CANDIDATES', forecast_calls=0, branches=0))
        if not hasattr(self.model, '_route_validator'):
            from eventwm_graph.l2_120k.model import load_model
            self.model._route_validator = load_model(kind='memory_cv')
        supported = [_support(self.model._route_validator, b.graph, q.native(), 0) for q in refs]
        ts = time.perf_counter()
        future = self._future(context, budget)
        self.model._check_future(future)
        tf = time.perf_counter()
        # Current native optical API caps prefixes at 120 actions. Retain this
        # explicit common supported horizon for all features, not a fake tail.
        limit = min(120, int(math.floor((future.times[-1]-context.timestamp+1e-8)/NOMINAL_STEP_SECONDS)))
        lengths = np.array([min(len(q.actions), limit) for q in refs], int)
        maxn = int(max(lengths))
        times = context.timestamp+np.arange(maxn+1)*NOMINAL_STEP_SECONDS
        indices = np.searchsorted(future.times, times-1e-9)
        if np.any(indices >= len(future.times)) or np.max(abs(future.times[indices]-times)) > 1e-7:
            raise ValueError('shared future lacks exact native endpoints')
        robots = np.stack([np.concatenate([q.poses[1:n+1], np.repeat(q.poses[n:n+1], maxn-n, 0)]) for q, n in zip(refs, lengths)])
        key = (future.identity(), tuple((q.query_id, q.content_hash, q.reference_hash) for q in refs), tuple(lengths), VERSION)
        numerical = self.caches['numeric'].get(key)
        if numerical is None:
            positions = np.concatenate([future.positions[:, indices[1:]], future.hidden_positions[:, indices[1:]]], axis=2)
            # Bound query-axis temporaries while preserving the identical tape.
            per_query = max(1, budget.worlds*maxn*max(1, positions.shape[2])*8*10)
            chunk = max(1, min(len(refs), (16*1024*1024)//per_query))
            parts, optics = [], []
            for start in range(0, len(refs), chunk):
                end = start+chunk
                parts.append(aggregate_joint(positions, robots[start:end], lengths[start:end], future.weights))
                optics.append(optical_batch(self.model, future, robots[start:end], lengths[start:end]))
            numeric = {k: np.concatenate([p[k] for p in parts]) if isinstance(parts[0][k], np.ndarray)
                       else sum([p[k] for p in parts], []) for k in parts[0]}
            optical = {k: np.concatenate([p[k] for p in optics]) for k in optics[0]}
            numerical = self.caches['numeric'].put(key, (numeric, optical))
        numeric, optical = numerical
        topology_key = (context.graph_version, tuple(context.goal), self.tile_size, VERSION)
        goal_distances = self.caches['topology'].get(topology_key)
        if goal_distances is None:
            goal_distances = self.caches['topology'].put(topology_key, context.spatial.goal_distances(context.goal))
        outcomes = [self._outcome(context, q, int(n), ok, numeric, optical, i, future, goal_distances)
                    for i, (q, n, ok) in enumerate(zip(refs, lengths, supported))]
        tn = time.perf_counter()
        # Deterministic identity tie break makes selection independent of input order.
        eligible = sorted((o for o in outcomes if o['candidate_legal'] and o['covered_duration'] > 0 and o['finite_event_mi_nats'] > 1e-8),
                          key=lambda o: (-o['finite_event_mi_nats'], o['query_id']))[:budget.branch_queries]
        branches = []
        for out in outcomes:
            out['branching'] = dict(status='NO_INFORMATIVE_EVENT' if out['finite_event_mi_nats'] <= 1e-8 else 'NOT_REQUESTED', represented_mass=0., residual_model_supported_mass=1.,
                                    model_external_unknown_mass=None, unknown_mass_supported=False,
                                    unknown_semantics='model misspecification/unknown geography is not a normalized omitted-event probability')
        for out in eligible:
            if (time.perf_counter()-tick)*1000 > budget.soft_budget_ms:
                out['branching']['status'] = 'BUDGET_UNEXPANDED'
                continue
            i = next(i for i, q in enumerate(refs) if q.query_id == out['query_id'])
            n = int(lengths[i])
            event = dict(successor_supported=True, future_identity=future.identity(),
                         per_world_none=optical['per_world_none'][i], index=int(indices[n]),
                         robot_pose=refs[i].poses[n].tolist(), region=None)
            masses = [('none', float(future.weights@event['per_world_none'])),
                      ('any', float(future.weights@(1-event['per_world_none'])))]
            represented = 0.
            for name, probability in sorted(masses, key=lambda p: (-p[1], p[0])):
                if probability <= 1e-15 or probability < budget.min_branch_probability or len(branches) >= budget.max_branches:
                    continue
                branch_key = (future.identity(), refs[i].content_hash, refs[i].reference_hash, refs[i].query_id, n, name, VERSION)
                branch = self.caches['branches'].get(branch_key)
                if branch is None:
                    successor = condition_observation_event(self.model, future, event, name)
                    weights = np.array([c.weight for c in successor.components])
                    summary = dict(role='imagined', approximation='coarse finite optical event; no exact observation refinement',
                                   belief_version=successor.fingerprint(), graph_version=successor.graph.namespace,
                                   expected_tracked_count=float(sum(c.weight*c.n for c in successor.components)),
                                   expected_hidden_count=float(sum(c.weight*np.arange(len(c.hidden_pmf))@c.hidden_pmf for c in successor.components)),
                                   conditional_ess=float(1/(weights@weights)),
                                   observation_interval=[context.timestamp, successor.timestamp], root_emission_excluded=True,
                                   event_semantics='at least one detection anywhere in prefix' if name == 'any' else 'no detection in prefix; not certified empty',
                                   successor_features_only=True, bellman_transition_supported=False,
                                   available_subsequent_decisions=('same reference continuation only; no new execution boundary' if out['censored_running'] else
                                                                  'upstream proposals at imagined time/pose; no executable actions supplied'),
                                   query_reference_complete=not out['censored_running'],
                                   static_revelation_supported=False, exact_reobservation_identity_supported=False)
                    branch = SparseConditionalBranch(refs[i].query_id, name, probability, successor.timestamp,
                                                     readonly(successor.robot_pose), successor, immutable(summary))
                    self.caches['branches'].put(branch_key, branch)
                branches.append(branch)
                represented += probability
            out['branching'].update(status='COARSE_FINITE_APPROXIMATION' if represented else 'UNEXPANDED',
                                    represented_mass=represented, residual_model_supported_mass=max(0., 1-represented))
        tb = time.perf_counter()
        graph = export_graph(context, outcomes, branches)
        end = time.perf_counter()
        timings = dict(static_support_ms=(ts-tick)*1000, forecast_ms=(tf-ts)*1000,
                       aggregation_ms=(tn-tf)*1000, branches_ms=(tb-tn)*1000,
                       graph_export_ms=(end-tb)*1000, total_ms=(end-tick)*1000,
                       incremental_ms=((end-tick)-(tf-ts))*1000)
        return DecisionPacket(context, tuple(outcomes), tuple(branches), graph,
            dict(contract=CONTRACT, compiler_version=VERSION, prediction_source='frozen L2 shared joint futures',
                 model_version=context.model_version, belief_version=context.belief_version, graph_version=context.graph_version,
                 future_identity=future.identity(), prediction_start_time=float(future.times[0]),
                 future_times=future.times.tolist(), worlds=budget.worlds, finite_bank_ess=float(1/(future.weights@future.weights)),
                 joint_future_shared_across_all_candidates=True, forecast_horizon=budget.horizon,
                 timings_ms=timings, branch_count=len(branches), cache=self.cache_statistics(),
                 soft_budget_ms=budget.soft_budget_ms, budget_exceeded=timings['total_ms'] > budget.soft_budget_ms,
                 budget_truncated_queries=sum(o['branching']['status']=='BUDGET_UNEXPANDED' for o in outcomes),
                 status='COMPLETE_CORE', no_guaranteed_safety=True,
                 limitations=['inherited belief and reobservation defects', 'broad unlocalized population',
                              'endpoint enclosing-body and horizontal social proxies', 'coarse optics; no calibrated occlusion',
                              'L4 completion and episode success unsupported']))

    def _outcome(self, context, ref, n, supported, numeric, optical, i, future, goal_distances):
        s = context.spatial
        poses = ref.poses
        end = poses[n]
        origin, endpoint = s.locate(poses[0, [0, 2]]), s.locate(end[[0, 2]])
        covered = n*NOMINAL_STEP_SECONDS
        censored = n < len(ref.actions)
        geometry_key = (context.graph_version, ref.content_hash, self.tile_size, VERSION)
        geometry = self.caches['geometry'].get(geometry_key)
        if geometry is None:
            cells = np.floor(poses[:, [0, 2]]/s.resolution).astype(int)
            nodes = np.array([s.cell_to_node.get(tuple(k), s.unknown_index) for k in cells], int)
            starts = np.flatnonzero(np.r_[True, nodes[1:] != nodes[:-1]])
            ends = np.r_[starts[1:]-1, len(nodes)-1]
            occurrences = [dict(node=int(nodes[a]), order=j, entry_time_relative=a*NOMINAL_STEP_SECONDS,
                                exit_time_relative=z*NOMINAL_STEP_SECONDS) for j, (a, z) in enumerate(zip(starts, ends))]
            route_length = float(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum())
            geometry = self.caches['geometry'].put(geometry_key, (nodes, occurrences, route_length))
        nodes, occurrences, route_length = geometry
        before = float(np.linalg.norm(poses[0, :3]-context.goal))
        remaining = float(np.linalg.norm(end[:3]-context.goal))
        topology_supported = bool(np.isfinite(goal_distances[[origin, endpoint]]).all())
        progress = float(goal_distances[origin]-goal_distances[endpoint]) if topology_supported else None
        def regions(masses):
            summed = np.bincount(nodes[1:n+1], weights=masses[:n], minlength=len(s.ids))
            return [dict(node=int(k), mass=float(summed[k])) for k in np.flatnonzero(summed > 0)]

        first = numeric['first_collision_mass'][i, :n]
        bins = np.linspace(0., covered, 5)
        hist = np.histogram(np.arange(1, n+1)*NOMINAL_STEP_SECONDS, bins=bins, weights=first)[0]
        p = float(numeric['p_collision'][i])
        ess = float(1/(future.weights@future.weights))
        probability = float(optical['p_any'][i])
        mi = finite_event_mi(1-optical['per_world_none'][i], future.weights)
        quantiles = numeric['minimum_quantiles'][i]
        vals = [p, float(numeric['p_social'][i]), float(numeric['expected_social_duration'][i]),
                float(numeric['expected_social_severity'][i]), *[v if v is not None else 0. for v in quantiles],
                float(numeric['p_no_humans'][i]), ref.duration, covered, ref.duration-covered, route_length,
                before-remaining, remaining, progress if progress is not None else 0., ess,
                probability, mi, float(censored), 0., 0., 0., 0.]
        orientation = ref.terminal_orientation
        vals.extend([float(poses[-1, 0]-poses[0, 0]), float(poses[-1, 2]-poses[0, 2]),
                     float(end[0]-poses[0, 0]), float(end[2]-poses[0, 2]),
                     math.sin(orientation) if orientation is not None else 0.,
                     math.cos(orientation) if orientation is not None else 0., math.sin(end[3]), math.cos(end[3])])
        mask = [bool(supported)]*len(vals)
        for j, q in enumerate(quantiles):
            mask[4+j] = bool(supported and q is not None)
        mask[14] = topology_supported and supported
        for name in ('p_l4_query_completion', 'p_episode_success', 'p_official_psc', 'p_nominal_mesh_contact'):
            mask[QUERY_FEATURES.index(name)] = False
        if orientation is None:
            mask[QUERY_FEATURES.index('terminal_yaw_sin')] = False
            mask[QUERY_FEATURES.index('terminal_yaw_cos')] = False
        legal = bool(supported and not context.belief.terminal)
        row = dict(query_id=ref.query_id, reference_hash=ref.reference_hash, reference_content_hash=ref.content_hash,
                   reference_version=ref.compiler_version, start_ego=poses[0].tolist(), start_timestamp=ref.start_timestamp,
                   target=ref.target, terminal_orientation=ref.terminal_orientation, reference_route=ref.reference_route,
                   expected_reference_endpoint=poses[-1].tolist(), expected_supported_endpoint=end.tolist(),
                   target_node=s.locate(np.asarray(ref.target)[[0, 2]]) if ref.target is not None else None,
                   origin_node=origin, endpoint_node=endpoint, route_occurrences=[dict(o) for o in occurrences],
                   route_length=route_length, nominal_duration=ref.duration, covered_duration=covered,
                   prediction_end_time=context.timestamp+covered, unsupported_duration=ref.duration-covered,
                   censored_running=censored, completion_rule=ref.completion_rule, execution_limit=ref.execution_limit,
                   candidate_legal=legal, status='SUPPORTED_PREFIX' if legal else 'UNSUPPORTED_KNOWN_SPACE_QUERY',
                   goal_distance_change=before-remaining, remaining_goal_distance=remaining,
                   observed_graph_progress=progress, topological_progress_supported=topology_supported,
                   progress_semantics='nominal supported endpoint; observed tiled-cell connectivity, not full-map geodesic',
                   p_nominal_collision=p if legal else None,
                   collision_semantics='native endpoint enclosing rectangle/disk overlap; conservative shape approximation, not mesh contact',
                   collision_first_times=(context.timestamp+np.arange(1, n+1)*NOMINAL_STEP_SECONDS).tolist(),
                   collision_first_mass=first.tolist() if legal else [], collision_histogram=hist.tolist() if legal else [],
                   collision_histogram_edges=(context.timestamp+bins).tolist(),
                   collision_regions=regions(first) if legal else [],
                   p_social_euclidean_proxy=float(numeric['p_social'][i]) if legal else None,
                   expected_social_proxy_duration=float(numeric['expected_social_duration'][i]) if legal else None,
                   expected_social_proxy_severity=float(numeric['expected_social_severity'][i]) if legal else None,
                   social_endpoint_times=(context.timestamp+np.arange(1, n+1)*NOMINAL_STEP_SECONDS).tolist(),
                   social_endpoint_probability=numeric['social_endpoint_probability'][i, :n].tolist() if legal else [],
                   social_regions=regions(numeric['social_endpoint_probability'][i]) if legal else [],
                   minimum_separation_quantiles=quantiles if legal else [None]*3,
                   p_no_modeled_humans=float(numeric['p_no_humans'][i]) if legal else None,
                   social_semantics='horizontal Euclidean center < 1m proxy; official geodesic/XYZ PSC unsupported',
                   p_any_observation=probability if legal else None, finite_event_mi_nats=mi if legal else 0.,
                   observation_first_mass=optical['first_detection_mass'][i, :n].tolist() if legal else [],
                   observation_endpoint_probability=optical['endpoint_any'][i, :n].tolist() if legal else [],
                   observation_regions=regions(optical['first_detection_mass'][i]) if legal else [],
                   observation_semantics='same L2 approximate optical law; no certified visibility or exact reobservation identity',
                   p_reobservation=None, visibility_occlusion_transitions=None,
                   p_l4_query_completion=None, p_episode_success=None, p_official_psc=None, p_nominal_mesh_contact=None,
                   support=dict(nominal_envelope_collision=legal, social_horizontal_proxy=legal, official_psc=False,
                                native_mesh_contact=False, l4_query_completion=False, episode_success=False,
                                exact_reobservation=False, calibrated_occlusion=False, continuous_sweep=False,
                                horizon_complete=not censored, finite_bank_only=True),
                   finite_bank_ess=ess, finite_bank_worlds=len(future.weights),
                   zero_hit_upper_95_iid=(1-.05**(1/len(future.weights))) if p == 0 and np.allclose(future.weights, 1/len(future.weights)) else None,
                   uncertainty_semantics='zero-hit bound only conditional iid Monte Carlo sampling; excludes model/ancestry/calibration error',
                   zero_hit_bound_iid_assumption_established=False,
                   source=dict(model_version=context.model_version, belief_version=context.belief_version,
                               graph_version=context.graph_version, future_identity=future.identity(),
                               forecast_horizon=float(future.times[-1]-context.timestamp), prediction_source='joint L2 bank including hidden pool'),
                   gnn_values=vals, gnn_support=mask)
        assert len(vals) == len(QUERY_FEATURES)
        return row
