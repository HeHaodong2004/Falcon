"""Remove learned future recurrence while preserving encoded beliefs/noise.

This is a controlled mechanism ablation, not a separately calibrated baseline.
No checkpoint is saved because the transition override is experiment-only.
"""
from dataclasses import replace
import sys
from eventforecastwm import benchmark,benchmark_cached
from eventbeliefwm.data import sha256


def replace_transition(model):
    def step(belief):
        return replace(belief,positions=belief.positions+.25*belief.velocities,
            age=belief.age+.25,clock=belief.clock+.25)
    model.core._step=step
    model.recipe['mechanism_ablation']='learned future recurrence replaced by CV; initial belief and noise fixed'
    return model


def main():
    if '--profile' in sys.argv:raise ValueError('Mechanism ablation keeps uncertainty fixed and never saves a model')
    original=benchmark.load_model
    benchmark.load_model=lambda *args,**kwargs:replace_transition(original(*args,**kwargs))
    benchmark.EXPERIMENT_SOURCES['ablate_transition.py']=sha256(__file__)
    benchmark_cached.main()


if __name__=='__main__':main()
