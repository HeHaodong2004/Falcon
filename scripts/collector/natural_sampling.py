"""Human-independent discrete root-time selection and static prefix composition.

No simulator, observations, human truth, labels or global RNG are accepted here.
"""
import random
from collector.core import digest

SAMPLER_VERSION='eventwm-natural-static-prefix-v1'


def natural_selection(episode_seed, sampling):
    start=int(sampling['window_start_step']);end=int(sampling['window_end_step'])
    prelude=sampling.get('prelude_controls',[])
    if start < len(prelude)+1 or end < start:
        raise ValueError('Natural window must be nonempty and after the static prelude')
    count=int(sampling.get('prefix_movement_queries',3))
    if count < 1:raise ValueError('Empty prefix candidate set')
    seed=int(sampling['sampler_seed'])
    rng=random.Random(int(digest(dict(episode_seed=int(episode_seed),sampler_seed=seed,version=SAMPLER_VERSION)),16))
    step=rng.randint(start,end)
    return dict(sampler_version=SAMPLER_VERSION,sampling_profile='natural',sampling_profile_family='natural',
        root_sampling_seed=seed,episode_seed=int(episode_seed),window_start_step=start,window_end_step=end,
        selected_root_step=step,prelude_steps=len(prelude),prefix_movement_queries=count,
        selected_prefix_path=rng.randrange(count),prefix_policy='canonical_static_movement_then_wait')


def natural_commands(recipe, motion_factory):
    """Factory sees only static world/robot; selection happens before it is called."""
    sample=natural_selection(recipe['seed'],recipe['sampling'])
    yield from recipe['sampling'].get('prelude_controls',[])
    motion=motion_factory(sample)
    for i in range(sample['selected_root_step']-sample['prelude_steps']):
        yield motion[i] if i < len(motion) else [0.,0.]
