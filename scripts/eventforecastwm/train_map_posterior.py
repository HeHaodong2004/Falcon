"""Versioned combined-map/initial-belief experiment on frozen physical targets."""
from pathlib import Path
from eventbeliefwm.data import sha256
from eventforecastwm import train,train_cached
from eventforecastwm.model_map_posterior import MapPosteriorForecastWorldModel


def main():
    original=train.fingerprints
    def fingerprints():
        result=original()
        for name in ('model_map_patch.py','model_posterior.py','model_map_posterior.py','train_map_posterior.py'):
            result[name]=sha256(Path(__file__).parent/name)
        return result
    train.fingerprints=fingerprints;train.ForecastWorldModel=MapPosteriorForecastWorldModel;train_cached.main()


if __name__=='__main__':main()
