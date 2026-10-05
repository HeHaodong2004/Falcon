"""Bounded prefix recipes. Decisions use only observations already acquired."""

def prefix_commands(recipe, history):
    if 'prefix_controls' in recipe:
        yield from recipe['prefix_controls']
        return
    profile = recipe['sampling']['profile']
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
