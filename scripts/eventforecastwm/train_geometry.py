"""Controlled map-covariate ablation using the unchanged physical trainer."""
import argparse
from pathlib import Path
from eventbeliefwm.data import read,sha256
from eventforecastwm import train
from eventforecastwm.geometry_data import attach_geometry


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=read(a.config)
    geometry=cfg['geometry_manifest'];original_fingerprint=train.fingerprints;original_loader=train.load_roots
    def fingerprints():
        result=original_fingerprint()
        for name in ('train_geometry.py','geometry_data.py'):
            result[name]=sha256(Path(__file__).parent/name)
        result['geometry_manifest']=sha256(geometry)
        return result
    def load_roots(manifest,ids,guard=None):
        return attach_geometry(original_loader(manifest,ids,guard),geometry,guard)
    train.fingerprints=fingerprints;train.load_roots=load_roots
    train.train(a.config)


if __name__=='__main__':main()
