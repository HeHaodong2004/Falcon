"""Keep active baseline trainer bytes fixed for the acceleration ablation."""
from pathlib import Path
from eventbeliefwm.data import sha256
from eventforecastwm import train,train_cached
from eventforecastwm.model_kinematic import KinematicForecastWorldModel


def main():
    original=train.fingerprints
    def fingerprints():
        result=original()
        for name in ('model_kinematic.py','train_kinematic.py'):result[name]=sha256(Path(__file__).parent/name)
        return result
    train.fingerprints=fingerprints;train.ForecastWorldModel=KinematicForecastWorldModel
    train_cached.main()


if __name__=='__main__':main()
