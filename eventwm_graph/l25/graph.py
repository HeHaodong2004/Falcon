"""Small dependency-free typed graph and a message-passing integration consumer."""
import numpy as np
from .schema import readonly

QUERY_FEATURES = ('p_nominal_collision', 'p_social_euclidean_proxy', 'expected_social_proxy_duration',
                  'expected_social_proxy_severity', 'minimum_separation_p10', 'minimum_separation_p50',
                  'minimum_separation_p90', 'p_no_modeled_humans', 'nominal_duration', 'covered_duration',
                  'unsupported_duration', 'route_length', 'goal_distance_change', 'remaining_goal_l2',
                  'observed_graph_progress', 'finite_bank_ess', 'p_any_observation', 'finite_event_mi_nats',
                  'censored_running', 'p_l4_query_completion', 'p_episode_success', 'p_official_psc',
                  'p_nominal_mesh_contact', 'reference_endpoint_dx', 'reference_endpoint_dz',
                  'supported_endpoint_dx', 'supported_endpoint_dz', 'terminal_yaw_sin', 'terminal_yaw_cos',
                  'supported_endpoint_yaw_sin', 'supported_endpoint_yaw_cos')
BRANCH_FEATURES = ('event_probability', 'successor_time_relative', 'ego_x', 'ego_z', 'ego_yaw',
                   'expected_tracked_count', 'expected_hidden_count', 'conditional_ess')
EDGE_TYPES = ('spatial_connected', 'query_origin', 'query_endpoint', 'query_route_occurrence',
              'spatial_query_reverse', 'query_collision_region', 'query_observation_region',
              'query_conditional_outcome', 'outcome_spatial_endpoint')


def export_graph(context, outcomes, branches):
    s = context.spatial
    ns, nq, nb = len(s.ids), len(outcomes), len(branches)
    nf = len(context.feature_names)+len(QUERY_FEATURES)+len(BRANCH_FEATURES)
    x = np.zeros((ns+nq+nb, nf), np.float32)
    valid = np.zeros_like(x, bool)
    provenance = np.zeros_like(x, np.uint8)  # 0 unknown,1 observed,2 predicted,3 imagined,4 censored,5 unsupported
    x[:ns, :len(context.feature_names)] = context.features
    valid[:ns, :len(context.feature_names)] = True
    valid[s.unknown_index, :14] = False
    valid[s.unknown_index, 7] = True
    valid[:ns, 18] = False  # no calibrated static optical visibility
    provenance[:ns, :14] = 1
    provenance[:ns, 14:18] = 2
    provenance[:ns, 19:21] = 1
    provenance[~valid] = 5
    offset = len(context.feature_names)
    for i, out in enumerate(outcomes):
        values = out['gnn_values']
        mask = out['gnn_support']
        x[ns+i, offset:offset+len(values)] = values
        valid[ns+i, offset:offset+len(values)] = mask
        provenance[ns+i, offset:offset+len(values)] = np.where(mask, 2, 5)
    for i, b in enumerate(branches):
        start = offset+len(QUERY_FEATURES)
        values = [b.probability, b.timestamp-context.timestamp, b.ego[0], b.ego[2], b.ego[3],
                  b.summary['expected_tracked_count'], b.summary['expected_hidden_count'], b.summary['conditional_ess']]
        x[ns+nq+i, start:] = values
        valid[ns+nq+i, start:] = True
        provenance[ns+nq+i, start:] = 3
    edges, types, attrs = [], [], []

    def edge(a, b, kind, order=0, start=0., end=0., mass=0., length=0.):
        edges.append((a, b)); types.append(kind); attrs.append((order, start, end, mass, length))

    root_indices = {o['query_id']: ns+i for i, o in enumerate(outcomes)}
    for i, out in enumerate(outcomes):
        qi = ns+i
        edge(qi, out['origin_node'], 1)
        edge(qi, out['endpoint_node'], 2, start=out['covered_duration'], end=out['covered_duration'])
        edge(out['origin_node'], qi, 4)
        edge(out['endpoint_node'], qi, 4)
        for occurrence in out['route_occurrences']:
            edge(qi, occurrence['node'], 3, occurrence['order'], occurrence['entry_time_relative'], occurrence['exit_time_relative'])
            edge(occurrence['node'], qi, 4, occurrence['order'], occurrence['entry_time_relative'], occurrence['exit_time_relative'])
        for region in out['collision_regions']:
            edge(qi, region['node'], 5, mass=region['mass'])
        for region in out['observation_regions']:
            edge(qi, region['node'], 6, mass=region['mass'])
    for i, b in enumerate(branches):
        node = ns+nq+i
        edge(root_indices[b.query_id], node, 7, start=b.timestamp-context.timestamp, mass=b.probability)
        edge(node, s.locate(b.ego[[0, 2]]), 8)
    qindices = np.arange(ns, ns+nq, dtype=np.int64)
    candidates = np.zeros(len(x), bool)
    candidates[qindices] = [o['candidate_legal'] for o in outcomes]
    assert np.isfinite(x).all()
    static_attrs = np.zeros((s.edge_index.shape[1], 5), np.float32)
    static_attrs[:, 4] = s.edge_length
    all_edges = np.concatenate([s.edge_index, np.asarray(edges, np.int64).reshape(-1, 2).T], axis=1)
    all_types = np.r_[np.zeros(s.edge_index.shape[1], np.int64), types].astype(np.int64)
    all_attrs = np.concatenate([static_attrs, np.asarray(attrs, np.float32).reshape(-1, 5)])
    global_features = [context.metadata['expected_tracked_count'], context.metadata['expected_hidden_count'],
                       context.metadata['association_entropy'] or 0., context.timestamp, *context.ego, *context.goal]
    global_support = [True]*len(global_features)
    global_support[2] = context.metadata['association_entropy'] is not None
    return dict(node_features=readonly(x), feature_support=readonly(valid), feature_provenance=readonly(provenance),
                global_features=readonly([global_features], np.float32), global_support=readonly([global_support], bool),
                global_feature_names=('expected_tracked_count', 'expected_hidden_count', 'association_entropy', 'timestamp',
                                      'ego_x', 'ego_y', 'ego_z', 'ego_yaw', 'goal_x', 'goal_y', 'goal_z'),
                node_type=readonly([0]*ns+[1]*nq+[2]*nb, np.int64),
                stable_ids=tuple(s.ids)+tuple('query:'+o['query_id'] for o in outcomes)+tuple('outcome:'+b.query_id+':'+b.event for b in branches),
                edge_index=readonly(all_edges), edge_type=readonly(all_types), edge_attributes=readonly(all_attrs),
                query_node_indices=readonly(qindices), query_ids=tuple(o['query_id'] for o in outcomes),
                candidate_mask=readonly(candidates), candidate_validity=readonly([o['candidate_legal'] for o in outcomes], bool),
                batch_ptr=readonly([0, len(x)], np.int64), query_batch_ptr=readonly([0, nq], np.int64),
                feature_names=tuple(context.feature_names)+QUERY_FEATURES+BRANCH_FEATURES,
                edge_type_names=EDGE_TYPES, edge_attribute_names=('occurrence_order', 'entry_time_relative_s', 'exit_time_relative_s', 'probability_mass', 'length_m'),
                metadata=dict(belief_version=context.belief_version, graph_version=context.graph_version,
                              model_version=context.model_version, spatial_connectivity='adjacent certified cells; full body traversal must be validated upstream'))


def batch_graphs(graphs):
    """Disjoint union; no cross-scene messages or future actions in root mask."""
    if not graphs:
        raise ValueError('nonempty graph list required')
    feature_names = graphs[0]['feature_names']
    if any(g['feature_names'] != feature_names for g in graphs):
        raise ValueError('incompatible graph feature schema')
    sizes = [len(g['node_features']) for g in graphs]
    offsets = np.r_[0, np.cumsum(sizes)]
    qptr = np.r_[0, np.cumsum([len(g['query_node_indices']) for g in graphs])]
    result = {key: readonly(np.concatenate([g[key] for g in graphs], axis=0)) for key in
              ('node_features', 'feature_support', 'feature_provenance', 'node_type', 'edge_type', 'edge_attributes', 'candidate_mask', 'candidate_validity', 'global_features', 'global_support')}
    result['edge_index'] = readonly(np.concatenate([g['edge_index']+offsets[i] for i, g in enumerate(graphs)], axis=1))
    result['query_node_indices'] = readonly(np.concatenate([g['query_node_indices']+offsets[i] for i, g in enumerate(graphs)]))
    result.update(batch_ptr=readonly(offsets), query_batch_ptr=readonly(qptr), feature_names=feature_names,
                  stable_ids=tuple((i, k) for i, g in enumerate(graphs) for k in g['stable_ids']),
                  query_ids=tuple((i, k) for i, g in enumerate(graphs) for k in g['query_ids']),
                  metadata=tuple(g['metadata'] for g in graphs), edge_type_names=graphs[0]['edge_type_names'],
                  edge_attribute_names=graphs[0]['edge_attribute_names'], global_feature_names=graphs[0]['global_feature_names'])
    return result


def message_passing_logits(graph, seed=17):
    """Untrained shared-weight GNN-compatible consumer, one logit per root row.

    Dependencies are NumPy only; no RL or learned policy is introduced. Invalid
    candidates return a masked finite placeholder with an explicit Boolean mask.
    """
    x = graph['node_features']*graph['feature_support']
    rng = np.random.default_rng(seed)
    h = np.tanh(x@rng.normal(0, .02, (x.shape[1], 8)))
    src, dst = graph['edge_index']
    messages = np.zeros_like(h)
    count = np.zeros(len(h))
    np.add.at(messages, dst, h[src])
    np.add.at(count, dst, 1)
    h = np.tanh(h+messages/np.maximum(count[:, None], 1))
    logits = h[graph['query_node_indices']]@rng.normal(0, .1, 8)
    mask = graph['candidate_validity']
    return np.where(mask, logits, 0.), mask
