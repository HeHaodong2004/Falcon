"""Controlled small map encoder ablation on unchanged physical supervision."""
from pathlib import Path
from eventbeliefwm.data import sha256
from eventforecastwm import train,train_cached
from eventforecastwm.model_map_patch import MapPatchForecastWorldModel


def main():
    original=train.fingerprints
    def fingerprints():
        result=original()
        for name in ('model_map_patch.py','train_map_patch.py'):result[name]=sha256(Path(__file__).parent/name)
        return result
    train.fingerprints=fingerprints;train.ForecastWorldModel=MapPatchForecastWorldModel;train_cached.main()


if __name__=='__main__':main()
