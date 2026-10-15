"""Matched physical ablations using the incumbent exact-resume training loop."""
from pathlib import Path
from eventbeliefwm.data import sha256
from eventforecastwm import train, train_cached
from eventforecastwm.performance_v2.physical import PhysicalWorldModel


def main():
    original=train.fingerprints
    def fingerprints():
        result=original()
        here=Path(__file__).parent
        result.update({str(p.relative_to(here.parent)):sha256(p) for p in (here/'physical.py',Path(__file__))})
        for name in ('model_posterior.py','model_independent.py'):
            result[name]=sha256(here.parent/name)
        return result
    train.fingerprints=fingerprints;train.ForecastWorldModel=PhysicalWorldModel
    train_cached.main()


if __name__=='__main__':main()
