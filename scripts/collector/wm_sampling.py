"""Targeted root selection from arrived legal history, never outcome evidence."""
import math

SAMPLER_VERSION = 'eventwm-wm-train-legal-strata-v1'
CATEGORIES = ('zero', 'unseen', 'visible', 'lost', 'visible_lost')


def legal_root_state(history, human_count, config):
    row = history[-1]
    tracks = [t for t in row['human_tracks'] if t['valid']]
    visible = [t for t in tracks if t['visible_now']]
    lost = [t for t in tracks if not t['visible_now']]
    ever = any(t['visible_now'] for h in history for t in h['human_tracks'])
    category = ('zero' if human_count == 0 else 'visible_lost' if visible and lost
                else 'visible' if visible else 'lost' if lost else 'unseen' if not ever
                else 'observed_expired')
    distances = [math.dist(row['robot_position'], t['position']) for t in visible]
    distance = min(distances) if distances else None
    age = max((float(t['observation_age']) for t in lost), default=None)
    near, medium, far = config.get('visible_boundaries_m', [1., 1.5, 3.])
    recent, stale = config.get('lost_boundaries_s', [.5, 1.])
    return dict(category=category, visible_count=len(visible), lost_count=len(lost),
        ever_observed=ever, nearest_legal_visible_distance=distance,
        visible_distance_bucket=(None if distance is None else 'inside_1m' if distance < near
            else 'near' if distance < medium else 'medium' if distance <= far else 'far'),
        oldest_lost_age=age, lost_age_bucket=(None if age is None else 'recent' if age <= recent
            else 'medium' if age <= stale else 'stale'),
        timestamp=row['timestamp'], selection_information='legal_arrived_history_only')


def matches(state, sampling):
    if state['category'] != sampling['target_category']:
        return False
    # A conservative legal-distance cap; hidden current humans/PSC are never read.
    # This does not certify that an unobserved human is outside canonical PSC.
    if state['visible_distance_bucket'] == 'inside_1m' and not sampling.get('allow_visible_inside_1m', False):
        return False
    for target, actual in [('target_visible_bucket', 'visible_distance_bucket'),
                           ('target_lost_bucket', 'lost_age_bucket')]:
        if sampling.get(target) is not None and state[actual] != sampling[target]:
            return False
    return True


def wm_commands(recipe, history):
    """Bounded legal scan/hold prefix; exhausted targets fail without fallback."""
    cfg = recipe['sampling']
    if cfg['target_category'] not in CATEGORIES:
        raise ValueError('Unknown wm_train root category')
    for step in range(cfg['max_prefix_steps'] + 1):
        state = legal_root_state(history, cfg['human_count'], cfg)
        if step >= cfg['min_prefix_steps'] and matches(state, cfg):
            return
        if step == cfg['max_prefix_steps']:
            raise RuntimeError('wm_train_stratum_unavailable: ' + cfg['target_category'])
        target = cfg['target_category']
        # Keep the observed person in view while waiting for a requested distance.
        # After losing a track, hold the view to age it without fabricating history.
        hold = (target in ('zero', 'unseen') or
                target == 'visible' and state['visible_count'] > 0 or
                target == 'lost' and state['category'] == 'lost' or
                target == 'visible_lost' and state['category'] == 'visible_lost')
        yield [0., 0.] if hold else [0., float(cfg.get('scan_angular_velocity', 6.))]
