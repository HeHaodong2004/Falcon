"""Matched attention ablation without changing existing training sources."""
from pathlib import Path
from eventbeliefwm.data import sha256
from eventforecastwm import train,train_cached
from eventforecastwm.model_independent import IndependentAttentionWorldModel


def main():
    original=train.fingerprints
    def fingerprints():
        result=original()
        for name in ('model_independent.py','train_independent.py'):result[name]=sha256(Path(__file__).parent/name)
        return result
    train.fingerprints=fingerprints;train.ForecastWorldModel=IndependentAttentionWorldModel;train_cached.main()


if __name__=='__main__':main()
