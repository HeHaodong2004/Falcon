"""Clearance-favoured robot regions and a directed reachability spanner.

All distances and macro-edge witnesses come from body-verified motions. The
native NavMesh supplies geometry/clearance features, never executable edges.
"""
from collections import Counter, defaultdict
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from ..native import path_length
from .config import ROBOT_ALGORITHM, TRAVERSABLE
from .roadmap import state_adjacency, join_poses


def predecessor_path(predecessors, start, end):
    path = [int(end)]
    while path[-1] != start:
        value = int(predecessors[path[-1]])
        if value < 0 or len(path) > len(predecessors):
            raise ValueError('No verified configuration path')
        path.append(value)
    return path[::-1]


def build_robot_topology(surface, roadmap, config, geometry=None):
    states, motions = roadmap['states'], roadmap['motions']
    n = len(states)
    empty = {'algorithm': ROBOT_ALGORITHM, 'directed': True, 'nodes': [], 'edges': [],
             'regions': [], 'floor_levels': surface.floor_levels,
             'foundation': 'robot_traversability', 'validation_level': config.validation_level,
             'state_regions': [-1]*n}
    if not motions:
        return empty
    active = sorted({i for e in motions for i in (e['source'], e['target'])})
    weights = {}
    for motion in motions:
        pair = tuple(sorted((motion['source'], motion['target'])))
        weights[pair] = min(weights.get(pair, math.inf), max(motion['length'], 1e-4))
    rr, cc, vv = [], [], []
    for (a, b), value in sorted(weights.items()):
        rr.extend([a, b]); cc.extend([b, a]); vv.extend([value, value])
    undirected = csr_matrix((vv, (rr, cc)), shape=(n, n))
    degree = np.diff(undirected.indptr)
    nearest = np.full(n, np.inf)
    nearest[[i for i in range(n) if i not in set(active)]] = 0.
    clearances = np.array([surface.nav.clearance(s['position'])['navmesh_boundary_distance'] for s in states])
    seeds = []
    while nearest.max() > config.region_radius:
        uncovered = np.flatnonzero(nearest > config.region_radius)
        score = clearances[uncovered]+.03*np.minimum(degree[uncovered], 10)+.05*np.minimum(nearest[uncovered], 2*config.region_radius)
        seed = int(uncovered[np.argmax(score)])
        seeds.append(seed)
        nearest = np.minimum(nearest, dijkstra(undirected, directed=False, indices=seed))
    seeds.sort()
    geo = dijkstra(undirected, directed=False, indices=seeds)
    labels = np.argmin(geo, axis=0)
    labels[~np.isfinite(geo).any(axis=0)] = -1
    nodes, regions = [], []
    for region, seed in enumerate(seeds):
        state = states[seed]
        owned = np.flatnonzero(labels == region).tolist()
        faces = sorted({states[i]['surface_id'] for i in owned})
        node = {'id': f'g{region:06d}', 'type': 'region_anchor', 'tags': ['region_anchor'],
                'position': state['position'], 'yaw': state['yaw'], 'state_id': seed,
                'surface_id': state['surface_id'], 'island': state['island'], 'floor_id': state['floor_id'],
                'strong_component': state['strong_component'], 'weak_component': state['weak_component'],
                'regions': [region], 'support_surface_ids': faces, 'visiting_yaws': [state['yaw']],
                'pose_validation': config.validation_level, 'robot_feasibility': TRAVERSABLE,
                'clearance': surface.nav.clearance(state['position'])}
        nodes.append(node)
        regions.append({'id': region, 'anchor': node['id'], 'state_ids': owned, 'surface_ids': faces,
                        'area': float(surface.areas[faces].sum()), 'island': state['island'],
                        'floor_ids': sorted({states[i]['floor_id'] for i in owned}),
                        'support_semantics': 'weak geodesic configuration region; directional reachability is separate; overlapping spatial supports'})
    face_owners = defaultdict(list)
    for state in states:
        if labels[state['id']] >= 0:
            face_owners[state['surface_id']].append(int(labels[state['id']]))
    for face in range(len(surface.triangles)):
        surface.region_ids[face] = min(face_owners[face]) if face_owners[face] else -1
    graph = state_adjacency(states, motions)
    pair_motion = {(m['source'], m['target']): m['id'] for m in motions}
    distances, predecessors = dijkstra(graph, directed=True, indices=seeds, return_predecessors=True)
    candidates = sorted((float(distances[i, s]), i, j) for i in range(len(seeds))
                        for j, s in enumerate(seeds) if i != j and np.isfinite(distances[i, s]))
    # Greedy directed spanner: every reachable anchor pair remains reachable.
    # Remove an edge only when a verified alternate macro-route already costs
    # no more than the stated time-stretch bound. The full roadmap is retained.
    import heapq
    adjacency = defaultdict(list)
    def existing_distance(source, target, limit):
        costs = {source: 0.}; queue = [(0., source)]
        while queue:
            cost, i = heapq.heappop(queue)
            if cost > limit:
                return math.inf
            if i == target:
                return cost
            if cost != costs[i]:
                continue
            for j, weight in adjacency[i]:
                value = cost+weight
                if value < costs.get(j, math.inf):
                    costs[j] = value; heapq.heappush(queue, (value, j))
        return math.inf
    edges = []
    for duration, source, target in candidates:
        if existing_distance(source, target, duration*config.spanner_stretch) <= duration*config.spanner_stretch:
            continue
        sequence = predecessor_path(predecessors[source], seeds[source], seeds[target])
        ids = [pair_motion[a, b] for a, b in zip(sequence[:-1], sequence[1:])]
        poses = join_poses([motions[i]['poses'] for i in ids])
        path = poses[:, :3]
        unique = [path[0]]+[q for p, q in zip(path[:-1], path[1:]) if np.linalg.norm(q-p)>1e-8]
        edges.append({'id': f'e{len(edges):06d}', 'source': nodes[source]['id'], 'target': nodes[target]['id'],
                      'motion_ids': ids, 'state_ids': sequence, 'path': np.asarray(unique).tolist(),
                      'length': path_length(path), 'duration': duration, 'robot_feasibility': TRAVERSABLE,
                      'validation_level': config.validation_level, 'floor_transition': bool(np.ptp(path[:,1])>=.75),
                      'height_range': [float(path[:,1].min()), float(path[:,1].max())],
                      'compiler_ref': {'kind': 'verified_robot_motion_sequence', 'motion_ids': ids}})
        adjacency[source].append((target, duration))
    outgoing = defaultdict(set)
    for edge in edges:
        outgoing[edge['source']].add(edge['target']); outgoing[edge['target']].add(edge['source'])
    for node in nodes:
        neighbours = sorted(outgoing[node['id']])
        state = states[node['state_id']]
        if len(neighbours) <= 1:
            node['type'] = 'terminal'
        elif len(state['heading_indices']) < config.heading_count//2:
            node['type'] = 'portal'
        elif len(neighbours) >= 3:
            targets = [nodes[int(x[1:])]['position'] for x in neighbours]
            if sum(not surface.nav.segment_free(p, q) for i,p in enumerate(targets) for q in targets[i+1:]) >= 2:
                node['type'] = 'junction'
        elif len(neighbours) == 2:
            p = np.asarray(node['position'])
            directions = [(np.asarray(nodes[int(x[1:])]['position'])-p)[[0,2]] for x in neighbours]
            directions = [v/max(np.linalg.norm(v),1e-9) for v in directions]
            if np.dot(*directions) > -.75:
                node['type'] = 'corridor_turn'
        node['tags'] = sorted(set(node['tags']+[node['type']]))
        if geometry is not None:
            rays = geometry.ray_distances(node['position'])
            node['visibility'] = {'distances':rays,'height':1.25,'range':8.,'source':'static_collision_rays'}
            if max(abs(np.roll(rays,1)-rays))>3:
                node['tags'].append('visibility_candidate')
    return {'algorithm':ROBOT_ALGORITHM,'foundation':'robot_traversability','directed':True,
            'validation_level':config.validation_level,'nodes':nodes,'edges':edges,'regions':regions,
            'floor_levels':surface.floor_levels,'state_regions':labels.tolist(),
            'statistics':{'nodes':len(nodes),'edges':len(edges),'node_types':dict(Counter(n['type'] for n in nodes)),
                          'reachable_anchor_pairs':len(candidates),'spanner_time_stretch_bound':config.spanner_stretch,
                          'isolated_stopping_states_retained_in_map':n-len(active)}}
