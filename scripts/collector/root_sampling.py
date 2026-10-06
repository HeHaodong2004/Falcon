"""Bounded prefix recipes. Decisions use only observations already acquired."""

def prefix_commands(recipe, history, natural_motion=None):
    if 'prefix_controls' in recipe:
        yield from recipe['prefix_controls']
        return
    profile = recipe['sampling']['profile']
    if profile == 'wm_train':
        from collector.wm_sampling import wm_commands
        yield from wm_commands(recipe, history)
        return
    if profile == 'natural':
        from collector.natural_sampling import natural_commands
        if natural_motion is None:
            raise ValueError('Natural sampling requires a human-independent static motion factory')
        yield from natural_commands(recipe, natural_motion)
        return
    if profile == 'wait':
        yield from [[0., 0.]] * recipe['sampling']['wait_steps']
        return
    # Native angular command: 6 rad/s integrated at 1/120 s per action.
    # Search at most one revolution; stop based only on current legal detections.
    for _ in range(126):
        if any(t['visible_now'] for t in history[-1]['human_tracks']):
            break
        yield [0., 6.]
    if profile == 'seek_lost':
        yield from [[0., 6.]] * 24
    elif profile == 'seek_move':
        yield from [[1.2, 0.]] * 12  # 0.12 m total nominal translation
    elif profile == 'seek_visible':
        yield from [[0., 0.]] * 12
    else:
        raise ValueError(f'Unknown root profile: {profile}')
