#!/usr/bin/env python
"""Run only after source-bound acceleration and original scientific gates pass."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eventwm_graph.partial.acceleration import run

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    print(json.dumps(run(p.parse_args().config), indent=2))
