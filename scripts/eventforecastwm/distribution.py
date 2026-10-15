"""Explicit dispatch for the Gaussian and persistent Student scene laws."""
from eventforecastwm.model import trajectory_log_probability as gaussian_log_probability
from eventforecastwm.model import energy_score as gaussian_energy_score


def trajectory_log_probability(f,*args,**kwargs):
    if hasattr(f,'branch_probabilities'):
        from eventforecastwm.performance_v2.regime import trajectory_log_probability as regime
        return regime(f,*args,**kwargs)
    if hasattr(f,'agent_scale_probabilities'):
        from eventforecastwm.model_agent_scale import trajectory_log_probability as agent
        return agent(f,*args,**kwargs)
    if hasattr(f,'scale_values'):
        from eventforecastwm.model_scale_mixture import trajectory_log_probability as mixture
        return mixture(f,*args,**kwargs)
    if hasattr(f,'concentration'):
        from eventforecastwm.model_student import trajectory_log_probability as student
        return student(f,*args,**kwargs)
    if hasattr(f,'decay_rate'):
        from eventforecastwm.model_damped import trajectory_log_probability as damped
        return damped(f,*args,**kwargs)
    return gaussian_log_probability(f,*args,**kwargs)


def energy_score(f,*args,**kwargs):
    if hasattr(f,'agent_scale_probabilities'):
        from eventforecastwm.model_agent_scale import energy_score as agent
        return agent(f,*args,**kwargs)
    if hasattr(f,'scale_values'):
        from eventforecastwm.model_scale_mixture import energy_score as mixture
        return mixture(f,*args,**kwargs)
    if hasattr(f,'concentration'):
        from eventforecastwm.model_student import energy_score as student
        return student(f,*args,**kwargs)
    if hasattr(f,'decay_rate'):
        from eventforecastwm.model_damped import energy_score as damped
        return damped(f,*args,**kwargs)
    return gaussian_energy_score(f,*args,**kwargs)


def enable_evaluation_dispatch():
    # Existing training processes and their pinned source files stay unchanged.
    # This affects only the calling evaluation process, with an exact Gaussian
    # fallback for every existing checkpoint.
    from eventforecastwm import train
    train.trajectory_log_probability=trajectory_log_probability;train.energy_score=energy_score
