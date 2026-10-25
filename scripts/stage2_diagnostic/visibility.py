"""Legal-input, revelation-only C-space frontier baseline.

This is a horizontal base-yaw surrogate, not the articulated camera or physical
ray caster. UNKNOWN and inflated BLOCKED cells stop a ray before its target.
The target may be UNKNOWN; no occupancy label or calibrated probability is
assigned. Only supplied future endpoints are considered, with no interpolation.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


BASELINE_ID = "cspace_frontier_2d_surrogate_v1"
HEIGHT_TOLERANCE_M = 0.22


def _integer_keys(value):
    array = np.asarray(value)
    if array.size == 0 and array.shape in ((0,), (0, 2)):
        return np.empty((0, 2), np.int64)
    if array.ndim != 2 or array.shape[1] != 2 or array.dtype.kind not in ("i", "u"):
        raise ValueError("candidate keys must be integer[N,2]")
    if np.any(array < np.iinfo(np.int32).min) or np.any(array > np.iinfo(np.int32).max):
        raise ValueError("cell coordinates exceed the supported int32 domain")
    return np.asarray(array, dtype=np.int64)


def _supercover(a, b):
    """Closed 2D grid supercover, including cells touched only at a corner.

    Same grid-line interval construction as PartialGraph.segment_cells, kept
    local to avoid importing the simulator/package or mutating a graph object.
    Coordinates here are already in grid units.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    delta = b - a
    crossings = {0.0, 1.0}
    for axis in (0, 1):
        if abs(delta[axis]) < 1e-14:
            continue
        lo, hi = sorted((a[axis], b[axis]))
        for line in range(math.ceil(lo), math.floor(hi) + 1):
            t = (line - a[axis]) / delta[axis]
            if 0.0 <= t <= 1.0:
                crossings.add(float(t))
    ordered = sorted(crossings)
    found = set()
    for t in ordered + [(u + v) / 2 for u, v in zip(ordered, ordered[1:])]:
        point = a + delta * t
        choices = []
        for coordinate in point:
            nearest = round(float(coordinate))
            choices.append((nearest - 1, nearest) if abs(coordinate - nearest) < 1e-10
                           else (math.floor(coordinate),))
        found.update((x, z) for x in choices[0] for z in choices[1])
    return found


def _visibility(graph, trajectory, keys, sensor, require_prefix):
    """Return bool[N] potential revelations using legal starting inputs only.

    Rows correspond exactly to ``keys``. Initially known keys return False.
    At least one supplied future pose must place the candidate center inside
    the horizontal FOV/range with an all-known-FREE supercover prefix. The
    origin-containing cell is omitted from that prefix; this does not mark it
    or any other UNKNOWN cell FREE. Known-FREE prefix cells must be within
    0.22 m of the supplied pose height, matching the existing layer tolerance.

    Vertical FOV is validated but unused: future jaw camera articulation and
    UNKNOWN floor heights are absent from these inputs. Inflated C-space
    BLOCKED cells are a conservative stopping heuristic, not optical geometry.
    The output is neither a true visibility guarantee nor a FREE/BLOCKED label.
    """
    if not isinstance(graph, Mapping) or not isinstance(sensor, Mapping):
        raise TypeError("graph and sensor must be mappings")
    resolution = float(graph["resolution"])
    timestamp = float(graph["timestamp"])
    if not np.isfinite([resolution, timestamp]).all() or resolution <= 0:
        raise ValueError("finite timestamp and positive resolution required")
    path = np.asarray(trajectory, np.float64)
    if path.ndim != 2 or path.shape[1] != 5 or not len(path) or not np.isfinite(path).all():
        raise ValueError("trajectory must be finite [time,x,y,z,yaw] rows")
    if abs(path[0, 0] - timestamp) > 1e-9 or np.any(np.diff(path[:, 0]) <= 0):
        raise ValueError("trajectory must start at graph time and use strict future boundaries")
    hfov = float(sensor["horizontal_fov_deg"])
    vfov = float(sensor["vertical_fov_deg"])
    radius = float(sensor["max_range_m"])
    if (not np.isfinite([hfov, vfov, radius]).all() or not 0 < hfov < 180
            or not 0 < vfov < 180 or radius <= 0):
        raise ValueError("finite valid FOVs and positive sensor range required")
    candidates = _integer_keys(keys)
    known = {}
    seen = set()
    for cell in graph["cells"]:
        key = tuple(_integer_keys([cell["cell"]])[0])
        state = int(cell["state"])
        if state not in (0, 1, 2) or state != cell["state"]:
            raise ValueError("invalid legal cell state")
        if key in seen:
            raise ValueError("duplicate graph cell")
        seen.add(key)
        if state:
            height = float(cell["height"])
            if not math.isfinite(height):
                raise ValueError("nonfinite known cell height")
            known[key] = (state, height)
    result = np.zeros(len(candidates), bool)
    if len(path) == 1 or not len(candidates):
        return result
    initially_unknown = np.array([tuple(key) not in known for key in candidates], bool)
    # A target with a FREE prefix is necessarily adjacent to a known FREE cell,
    # except for the explicitly omitted origin cell or its immediate neighbor.
    # This exact prefilter avoids rays through the large distant UNKNOWN area.
    frontier = np.zeros(len(candidates), bool)
    if require_prefix:
        for i in np.flatnonzero(initially_unknown):
            x, z = candidates[i]
            frontier[i] = any(known.get(k, (0,))[0] == 1 for k in
                              ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)))
    centers = (candidates.astype(np.float64) + 0.5) * resolution
    tangent = math.tan(math.radians(hfov / 2))
    for pose in path[1:]:
        origin = pose[[1, 3]] / resolution
        origin_cell = tuple(np.floor(origin).astype(np.int64))
        delta = centers - pose[[1, 3]]
        cosine, sine = math.cos(pose[4]), math.sin(pose[4])
        forward = delta[:, 0] * cosine - delta[:, 1] * sine
        side = delta[:, 0] * sine + delta[:, 1] * cosine
        close_to_origin = np.abs(candidates - origin_cell).sum(1) <= 1
        prefix_eligible = frontier | close_to_origin if require_prefix else True
        eligible = (initially_unknown & ~result & prefix_eligible
                    & (forward > 1e-6) & (np.abs(side) <= forward * tangent)
                    & (np.sum(delta * delta, axis=1) <= radius * radius))
        if not require_prefix:
            result[eligible] = True
            continue
        for i in np.flatnonzero(eligible):
            target = tuple(candidates[i])
            prefix = _supercover(origin, candidates[i] + 0.5)
            prefix.discard(origin_cell)
            prefix.discard(target)
            if all(known.get(key, (0, 0))[0] == 1
                   and abs(known[key][1] - pose[2]) <= HEIGHT_TOLERANCE_M for key in prefix):
                result[i] = True
    return result


def visibility(graph, trajectory, keys, sensor):
    """Return conservative first-UNKNOWN frontier opportunities, as bool[N].

    This uses horizontal base-yaw FOV/range and requires a known-FREE prefix.
    It assigns no occupancy labels and is not exact articulated-camera sensing.
    """
    return _visibility(graph, trajectory, keys, sensor, require_prefix=True)


def frustum_opportunity(graph, trajectory, keys, sensor):
    """Return a less restrictive horizontal FOV/range surrogate, as bool[N].

    Rays may pass through UNKNOWN/BLOCKED C-space cells in this separate control.
    This never updates those cells or labels them FREE. It is a superset of
    ``visibility`` under the same surrogate, not an upper bound on real legal
    observations: articulated camera poses and inflated wall witnesses differ.
    """
    return _visibility(graph, trajectory, keys, sensor, require_prefix=False)
