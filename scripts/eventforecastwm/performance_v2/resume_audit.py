"""Run the actual interrupted GPU trainer check for a v2 physical family."""
import argparse
from functools import partial
from pathlib import Path
import sys

from eventbeliefwm.data import sha256
from eventforecastwm import train, resume_audit
from eventforecastwm.performance_v2.physical import PhysicalWorldModel, VARIANTS


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--variant',required=True,choices=sorted(VARIANTS))
    args,rest=p.parse_known_args()
    train.ForecastWorldModel=partial(PhysicalWorldModel,variant=args.variant)
    original=train.fingerprints
    def fingerprints():
        result=original()
        for name in ('physical.py','resume_audit.py'):
            result['performance_v2/'+name]=sha256(Path(__file__).with_name(name))
        return result
    train.fingerprints=fingerprints;sys.argv=[sys.argv[0],*rest];resume_audit.main()


if __name__=='__main__':main()
