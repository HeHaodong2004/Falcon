"""Optimal Reciprocal Collision Avoidance with full robot responsibility.

A direct port of the RVO2 library (van den Berg, Guy, Snape, Lin, Manocha;
``Agent::computeNewVelocity``, ``linearProgram1/2/3``) in the planar frame.
Falcon humans never react to the robot, so the robot takes the *full* velocity
correction ``u`` (half-plane point ``v + u``) instead of the reciprocal
``v + u/2`` -- the standard treatment of non-cooperating agents.  Static
obstacle samples from the legal map become hard constraints that the 3-D
fallback (``linearProgram3``) never relaxes, exactly as RVO2's obstacle lines.

Inputs are legal only: robot pose/measured velocity, constant-velocity
estimates from legal tracks (age-extrapolated, radius inflated by track age),
observed BLOCKED cells, and a preferred velocity from the legal-map route.
"""
from dataclasses import dataclass
import math
import numpy as np

RVO_EPSILON = 1e-5


def det(a, b):
    return a[0]*b[1]-a[1]*b[0]


@dataclass
class Line:
    point: np.ndarray
    direction: np.ndarray


def linear_program1(lines, line_no, radius, opt_velocity, direction_opt):
    line = lines[line_no]
    dot = float(np.dot(line.point, line.direction))
    disc = dot*dot+radius*radius-float(np.dot(line.point, line.point))
    if disc < 0.:
        return None
    sq = math.sqrt(disc)
    t_left, t_right = -dot-sq, -dot+sq
    for i in range(line_no):
        denom = det(line.direction, lines[i].direction)
        numer = det(lines[i].direction, line.point-lines[i].point)
        if abs(denom) <= RVO_EPSILON:
            if numer < 0.:
                return None
            continue
        t = numer/denom
        if denom >= 0.:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return None
    if direction_opt:
        if np.dot(opt_velocity, line.direction) > 0.:
            return line.point+t_right*line.direction
        return line.point+t_left*line.direction
    t = float(np.dot(line.direction, opt_velocity-line.point))
    if t < t_left:
        return line.point+t_left*line.direction
    if t > t_right:
        return line.point+t_right*line.direction
    return line.point+t*line.direction


def linear_program2(lines, radius, opt_velocity, direction_opt):
    if direction_opt:
        result = opt_velocity*radius
    elif np.dot(opt_velocity, opt_velocity) > radius*radius:
        result = opt_velocity/np.linalg.norm(opt_velocity)*radius
    else:
        result = opt_velocity.copy()
    for i in range(len(lines)):
        if det(lines[i].direction, lines[i].point-result) > 0.:
            temp = result
            new = linear_program1(lines, i, radius, opt_velocity, direction_opt)
            if new is None:
                return i, temp
            result = new
    return len(lines), result


def linear_program3(lines, num_obst, begin, radius, result):
    distance = 0.
    for i in range(begin, len(lines)):
        if det(lines[i].direction, lines[i].point-result) > distance:
            proj = list(lines[:num_obst])
            for j in range(num_obst, i):
                d = det(lines[i].direction, lines[j].direction)
                if abs(d) <= RVO_EPSILON:
                    if np.dot(lines[i].direction, lines[j].direction) > 0.:
                        continue
                    point = 0.5*(lines[i].point+lines[j].point)
                else:
                    point = lines[i].point+(det(lines[j].direction, lines[i].point-lines[j].point)/d)*lines[i].direction
                direction = lines[j].direction-lines[i].direction
                direction = direction/np.linalg.norm(direction)
                proj.append(Line(point, direction))
            temp = result
            opt = np.array([-lines[i].direction[1], lines[i].direction[0]])
            fail, new = linear_program2(proj, radius, opt, True)
            result = temp if fail < len(proj) else new
            distance = det(lines[i].direction, lines[i].point-result)
    return result


def orca_line(rel_pos, rel_vel, velocity, combined_radius, inv_tau, inv_dt, responsibility=1.0):
    """ORCA half-plane for one neighbour (RVO2 ``computeNewVelocity`` body)."""
    dist_sq = float(np.dot(rel_pos, rel_pos))
    r_sq = combined_radius*combined_radius
    if dist_sq > r_sq:
        w = rel_vel-inv_tau*rel_pos
        w_len_sq = float(np.dot(w, w))
        dot1 = float(np.dot(w, rel_pos))
        if dot1 < 0. and dot1*dot1 > r_sq*w_len_sq:
            w_len = math.sqrt(w_len_sq)
            unit_w = w/w_len
            direction = np.array([unit_w[1], -unit_w[0]])
            u = (combined_radius*inv_tau-w_len)*unit_w
        else:
            leg = math.sqrt(dist_sq-r_sq)
            if det(rel_pos, w) > 0.:
                direction = np.array([rel_pos[0]*leg-rel_pos[1]*combined_radius,
                                      rel_pos[0]*combined_radius+rel_pos[1]*leg])/dist_sq
            else:
                direction = -np.array([rel_pos[0]*leg+rel_pos[1]*combined_radius,
                                       -rel_pos[0]*combined_radius+rel_pos[1]*leg])/dist_sq
            dot2 = float(np.dot(rel_vel, direction))
            u = dot2*direction-rel_vel
    else:
        w = rel_vel-inv_dt*rel_pos
        w_len = float(np.linalg.norm(w))
        unit_w = w/w_len if w_len > 1e-12 else -rel_pos/max(math.sqrt(dist_sq), 1e-9)
        direction = np.array([unit_w[1], -unit_w[0]])
        u = (combined_radius*inv_dt-w_len)*unit_w
    return Line(velocity+responsibility*u, direction)


@dataclass
class ORCAConfig:
    robot_radius: float = 0.55       # disk proxy of the 1.23x0.49 m body (front 0.584 m)
    human_radius: float = 0.30       # torso proxy (official contact model: 2 cm stick skeleton)
    margin: float = 0.25             # + one native step of duty-cycle tracking error
    time_horizon: float = 1.5        # s, humans
    time_horizon_obst: float = 0.15  # s, observed walls (native robot stops instantly)
    static_radius: float = 0.18      # robot centre to BLOCKED-cell centre (cells = walls + 0.25 m)
    static_range: float = 1.4
    neighbor_range: float = 8.0
    max_neighbors: int = 8
    age_inflation: float = 0.6       # m/s radius growth for stale tracks
    responsibility: float = 1.0      # 1.0 = full (non-reactive humans); 0.5 = reciprocal ORCA


class ORCASolver:
    def __init__(self, config=None):
        self.cfg = config or ORCAConfig()

    def lines(self, P, V, humans, statics, dt):
        cfg = self.cfg
        obst = []
        inv_dt = 1./dt
        inv_to = 1./cfg.time_horizon_obst
        for Q in statics:
            rel = np.asarray(Q)-P
            obst.append(orca_line(rel, V.copy(), V, cfg.static_radius, inv_to, inv_dt, 1.0))
        agents = []
        inv_t = 1./cfg.time_horizon
        near = sorted(humans, key=lambda h: float(np.hypot(*(h.position-P))))
        for h in near[:cfg.max_neighbors]:
            rel = h.position-P
            if float(np.hypot(*rel)) > cfg.neighbor_range:
                continue
            r = cfg.robot_radius+cfg.human_radius+cfg.margin+cfg.age_inflation*min(h.age, 2.)
            agents.append(orca_line(rel, V-h.velocity, V, r, inv_t, inv_dt, cfg.responsibility))
        return obst, agents

    def solve(self, P, V, v_pref, vmax, humans, statics, dt):
        obst, agents = self.lines(P, V, humans, statics, dt)
        lines = obst+agents
        fail, v = linear_program2(lines, vmax, np.asarray(v_pref, dtype=float), False)
        if fail < len(lines):
            v = linear_program3(lines, len(obst), fail, vmax, v)
        return v, lines, len(obst)

    @staticmethod
    def violation(lines, v):
        if not lines:
            return 0.
        return max(det(l.direction, l.point-v) for l in lines)
