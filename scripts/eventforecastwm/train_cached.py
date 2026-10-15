"""Resumable physical training with reusable legal preprocessing sidecars."""
import argparse
from pathlib import Path
from eventbeliefwm.data import read,sha256
from eventforecastwm import train
from eventforecastwm.prepared_cache import prepare


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=read(a.config)
    old_fingerprints=train.fingerprints;old_prepare=train.prepare_roots;old_load=train.load_roots
    names=['prepared_cache.py','train_cached.py']
    if cfg['model'].get('variant')=='age_aware_v1':
        from eventforecastwm.model_age import AgeAwareForecastWorldModel
        train.ForecastWorldModel=AgeAwareForecastWorldModel;names.append('model_age.py')
    if cfg.get('geometry_manifest'):
        from eventforecastwm.geometry_data import attach_geometry
        names.append('geometry_data.py')
        def load(manifest,ids,guard=None):return attach_geometry(old_load(manifest,ids,guard),cfg['geometry_manifest'],guard)
        train.load_roots=load
    def fingerprints():
        result=old_fingerprints()
        for name in names:result[name]=sha256(Path(__file__).parent/name)
        if cfg.get('geometry_manifest'):result['geometry_manifest']=sha256(cfg['geometry_manifest'])
        return result
    def cached_prepare(model,roots,guard=None):
        prepare(model,roots,cfg['manifest'],cfg['prepared_cache'],old_prepare,guard)
    train.fingerprints=fingerprints;train.prepare_roots=cached_prepare
    train.train(a.config)


if __name__=='__main__':main()
