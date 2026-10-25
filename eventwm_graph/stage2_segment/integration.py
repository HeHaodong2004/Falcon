"""Minimal frozen-Stage1/complete-Stage2 Sandbox compatibility adapter."""
import time
from eventwm_graph.stage2_quality.integration import load_stage1


class WorldModelSandbox:
    def __init__(self,stage1,stage2,legacy_snapshot=False):
        self.stage1=stage1;self.stage2=stage2;self.legacy_snapshot=legacy_snapshot;self.last_profile=None
    def predict_decision(self,previous_context,observation,*,executed_action,trajectories,arrived_graph=None):
        begin=time.perf_counter()
        people=self.stage1.predict(previous_context,observation,partial_graph=arrived_graph,executed_action=executed_action,horizon=3.,worlds=40)
        human=time.perf_counter();actual=self.stage1.last_context
        deltas=self.stage2.predict_batch([dict(partial_graph=actual.partial_graph,trajectory=p) for p in trajectories]);predicted=time.perf_counter()
        future=[d['future_partial_graph'] for d in deltas]
        if self.legacy_snapshot:future=[g.to_snapshot() if hasattr(g,'to_snapshot') else g for g in future]
        end=time.perf_counter()
        self.last_profile=dict(stage1_seconds=human-begin,stage2_complete_seconds=predicted-human,snapshot_adapter_seconds=end-predicted,
            total_seconds=end-begin,stage1_calls=1,queries=len(deltas),worlds=40,knots=64)
        return dict(human_worlds=people,graph_deltas=deltas,hypothetical_graphs=future,actual_context=actual)
