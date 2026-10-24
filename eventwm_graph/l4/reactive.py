"""Nominal tracking and six short native avoidance templates; no beam or mesh.

Risk is a conditional prediction from legal measurements. It never establishes
an actual collision or prohibits nominal execution when avoidance is infeasible.
"""
import math
import numpy as np
from numba import njit
from eventwm_graph.partial.query import wrap, yaw_to
from .geometry import swept_free


def nominal_action(pose, target, query, terminal):
    arrived = np.linalg.norm(pose[:3]-target) <= query.position_tolerance
    desired = query.terminal_orientation if terminal and arrived else yaw_to(pose[:3], target)
    angle = wrap(desired-pose[3])
    return ('left' if angle > 0 else 'right') if abs(angle) > query.orientation_tolerance else 'forward'


@njit(cache=True)
def clearance(pose, people, elapsed):
    result = 100.
    c, s = math.cos(pose[3]), math.sin(pose[3])
    for h in people:
        # Recent, localized evidence can trigger action. Older/unknown people
        # remain uncertain; a widening absence envelope cannot veto progress.
        if h[4] > .8:
            continue
        age = h[4]+elapsed
        dx = h[0]+h[2]*age-pose[0]; dz = h[1]+h[3]*age-pose[2]
        x = c*dx-s*dz; z = s*dx+c*dz
        distance = math.hypot(max(-.646495-x, 0., x-.584202), max(-.242846-z, 0., z-.238225))
        result = min(result, distance-.616-.03-min(.12, .1*age))
    return result


def near_term_risk(pose, action, people, spec, horizon=12):
    if not len(people):
        return False
    # Required heading/terminal turns remain nominal; they are not an excuse
    # for waiting. Test forward exposure along the current heading only.
    if action != 'forward' and horizon != 1:
        return False
    p = np.asarray(pose, dtype=float)
    if clearance(p, people, 0.) <= 0.:
        return True
    for i in range(horizon):
        p = spec.step(p, action)
        if clearance(p, people, (i+1)*spec.max_dt) <= 0.:
            return True
    return False


def choose_maneuver(pose, target, people, geometry, spec, preferred, forecast, times, now):
    best = None; best_score = math.inf; checks = 0
    # Compare predicted risk with continuing the nominal direction. Avoidance
    # is useful when it materially improves clearance; it need not prove a
    # collision-free future under the conservative all-asset human envelope.
    baseline=100.;p=pose.copy()
    for i in range(40):
        delta=target-p[:3];desired=math.atan2(-delta[2],delta[0]);error=wrap(desired-p[3])
        action=('left' if error>0 else 'right') if abs(error)>spec.turn/2+1e-5 else 'forward'
        p=spec.step(p,action);baseline=min(baseline,clearance(p,people,(i+1)*spec.max_dt))
    for side in ('left', 'right'):
        for turns in (6, 10, 14):
            # Shift about a metre laterally, restore heading, then pass before
            # rejoining. An immediate rejoin after a shallow sidestep steers
            # straight back into the same pedestrian (retained dev02 traces).
            advances = int(math.ceil(1.05/(spec.forward*math.sin(turns*spec.turn))))
            other = 'right' if side == 'left' else 'left'
            sequence = [side]*turns+['forward']*advances+[other]*turns+['forward']*6
            p = pose.copy(); minimum = 100.; valid = True
            for i, action in enumerate(sequence):
                b = geometry.predict(p, action, spec); checks += 1
                # Native center support can authorize a best-effort maneuver
                # when body-grid support is unavailable. This is explicitly
                # NOT full-body clearance; actual contact remains authoritative.
                supported=swept_free(p,b,*geometry.kernel_args())
                if not supported and geometry.navigation is not None:
                    supported=geometry.navigation.segment_free(p[:3],b[:3])
                if (not supported or (action=='forward' and np.linalg.norm((b-p)[[0,2]])<.02)):
                    valid = False; break
                for u in (.5, 1.):
                    mid = p+(b-p)*u
                    mid[3] = p[3]+wrap(b[3]-p[3])*u
                    minimum = min(minimum, clearance(mid, people, (i+u)*spec.max_dt))
                p = b
            if not valid or (minimum<=0. and minimum<baseline+.15):
                continue
            # Prefer forward progress, modest clearance, and the previous side.
            score = np.linalg.norm(p[:3]-target)+.018*turns-2.*min(minimum, .5)
            if preferred and preferred != (1 if side == 'left' else 2):
                score += .25
            if len(times) and forecast.size:
                k = min(int(np.searchsorted(times, now+len(sequence)*spec.max_dt)), len(times)-1)
                if now+len(sequence)*spec.max_dt <= times[-1]:
                    distances = np.linalg.norm(forecast[k]-p[[0,2]], axis=1)
                    distances = distances[np.isfinite(distances)]
                    if len(distances):
                        score += .2*float(np.maximum(0., 1.2-distances).mean())
            if score < best_score:
                best = sequence; best_score = score
    return best, checks


def static_escape(pose, target, people, geometry, spec):
    """Two six-action corner nudges, only when current grid support is known."""
    best=None; score=float(np.linalg.norm(pose[:3]-target))-.1; checks=0
    for side in ('left','right'):
        sequence=[side]*4+['forward']*2;p=pose.copy();valid=True
        for i,action in enumerate(sequence):
            b=geometry.predict(p,action,spec);checks+=1
            if not swept_free(p,b,*geometry.kernel_args()) or clearance(b,people,(i+1)*spec.max_dt)<=0.:
                valid=False;break
            p=b
        distance=float(np.linalg.norm(p[:3]-target))
        if valid and distance<score:best=sequence;score=distance
    return best,checks
