"""Reproduce the retained native arm-contact witness using static asset FK."""
import argparse
import json
from pathlib import Path
import numpy as np
import habitat_sim
from eventwm_graph.benchmark.map import load_certified_map
from eventwm_graph.l4.geometry import StaticGrid
from eventwm_graph.l4.precise import attach,update


def main():
    p=argparse.ArgumentParser();p.add_argument('--witness',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    w=json.loads(a.witness.read_text());h=load_certified_map(w['manifest'])
    geometry=StaticGrid.from_navigation(h.navigation,w['reference_route'])
    calibration=attach(geometry,h,w['initial_robot_joints'])
    try:
        update(geometry,calibration,w['actual_robot_joints'],w['forward_count'])
        boxes=geometry.precise.gait_boxes[w['gait_phase']]
        checks={str(margin):geometry.precise.swept(np.asarray(w['before']),np.asarray(w['after']),boxes=boxes,margin=margin)
                for margin in (.005,.015,.025,.04,.06)}
        assert w['native_contact']['static_obstacle_contact']
        assert checks['0.005'] and not checks['0.025']
        a.output.write_text(json.dumps(dict(witness=str(a.witness),accepted_by_margin=checks,passed=True),indent=2))
        print(checks)
    finally:calibration.close()


if __name__=='__main__':main()
