"""Isolated initial-belief ablation; baseline training sources stay frozen."""
from pathlib import Path
from eventbeliefwm.data import sha256
from eventforecastwm import train,train_cached
from eventforecastwm.model_posterior import PosteriorForecastWorldModel


def main():
    original=train.fingerprints
    def fingerprints():
        result=original()
        for name in ('model_posterior.py','train_posterior.py'):result[name]=sha256(Path(__file__).parent/name)
        return result
    train.fingerprints=fingerprints;train.ForecastWorldModel=PosteriorForecastWorldModel;train_cached.main()


if __name__=='__main__':main()
