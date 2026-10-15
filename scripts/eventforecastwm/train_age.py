"""Versioned age-conditioning ablation; baseline trainer bytes stay frozen."""
import argparse
from pathlib import Path
from eventbeliefwm.data import sha256
from eventforecastwm import train
from eventforecastwm.model_age import AgeAwareForecastWorldModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    original=train.fingerprints
    def fingerprints():
        result=original()
        for name in ('model_age.py','train_age.py'):result[name]=sha256(Path(__file__).parent/name)
        return result
    train.fingerprints=fingerprints
    train.ForecastWorldModel=AgeAwareForecastWorldModel
    train.train(a.config)


if __name__=='__main__':main()
