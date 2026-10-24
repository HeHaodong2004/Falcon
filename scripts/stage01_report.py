#!/usr/bin/env python
"""Offline, train-only distribution and performance report."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eventwm_graph.partial.report import summarize

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',action='append',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();r=summarize(a.dataset,a.output)
    print({k:v for k,v in r.items() if k not in ('roots','component_timings')})
