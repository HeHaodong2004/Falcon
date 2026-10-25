"""Minimal actual Stage1+Stage2 World Model Sandbox adapter.

Stage1 runs once per arrived decision, independently of hypothetical Queries.
No planner/L2.5 redesign, new Stage1 training or future human input is involved.
"""
import importlib
import hashlib
import json
from pathlib import Path
import sys
import time
import types
from .api import Stage2Predictor,apply_graph_delta
from .common import ORIGINAL,sha


def load_stage1():
    source=Path((ORIGINAL/'output/research_transition_20261024/inference_source_path.txt').read_text().strip())
    # The immutable Stage1 source is selected explicitly, including its guarded
    # spatial proposal and association implementation identities.
    for name in ('l2_two_stage','l2_final'):
        package='eventwm_graph.'+name
        if any(k.startswith(package+'.') for k in sys.modules):raise RuntimeError('load frozen Stage1 before any other Stage1 modules')
        module=types.ModuleType(package);module.__path__=[str(source/'eventwm_graph'/name)];sys.modules[package]=module
    manifest=json.loads((source/'manifest.json').read_text())
    files=manifest['files']
    if hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()!=manifest['identity'] or manifest['identity']!=source.name:raise ValueError('frozen Stage1 manifest identity mismatch')
    for rel,expected in files.items():
        if not isinstance(expected,str) or not (source/rel).is_file() or sha(source/rel)!=expected:raise ValueError('frozen Stage1 source mismatch: '+rel)
    checkpoint=ORIGINAL/'output/l2_two_stage_20261023/experiments/final_stage1_research_20261024/last.pt'
    if sha(checkpoint)!='99b73751bbf343946893b7f45ca869925fe46f8bccd7661b65845f148560b92f':raise ValueError('frozen Stage1 checkpoint changed')
    cls=importlib.import_module('eventwm_graph.l2_two_stage.conditioned').HistoryConditionedWorldModel
    return cls(checkpoint),source


class WorldModelSandbox:
    """One stream; callers retain the immutable actual observation context."""
    def __init__(self,stage1,stage2):self.stage1=stage1;self.stage2=stage2;self.last_profile=None

    def predict_decision(self,previous_context,observation,*,executed_action,trajectories,arrived_graph=None):
        begin=time.perf_counter()
        people=self.stage1.predict(previous_context,observation,partial_graph=arrived_graph,executed_action=executed_action,horizon=3.,worlds=40)
        human_done=time.perf_counter();actual=self.stage1.last_context
        graph=actual.partial_graph
        # Stage2 consumes the legal immutable graph directly. Its updater
        # privately restores metadata without copying this whole map twice.
        deltas=self.stage2.predict_batch([dict(partial_graph=graph,trajectory=t) for t in trajectories]);map_done=time.perf_counter()
        future=[apply_graph_delta(graph,d) for d in deltas];updated=time.perf_counter()
        self.last_profile=dict(stage1_seconds=human_done-begin,stage2_batch_seconds=map_done-human_done,update_seconds=updated-map_done,total_seconds=updated-begin,
            stage1_calls=1,queries=len(trajectories),worlds=40,knots=64)
        return dict(human_worlds=people,graph_deltas=deltas,hypothetical_graphs=future,actual_context=actual)
