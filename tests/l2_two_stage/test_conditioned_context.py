"""Real pre-arrival context, exact historical support and legal proposal basis."""
import numpy as np
import torch
from eventwm_graph.l2_two_stage.context import OnlineContext
from eventwm_graph.l2_two_stage.conditioned import HistoryConditionedWorldModel,history_from_context
from eventwm_graph.l2_two_stage.data import jobs_for,read,selected_branch
from eventwm_graph.l2_two_stage.common import OUT
from pathlib import Path


def test_native_context_updates_association_in_timed_arrival_without_mutation():
    torch.set_num_threads(1)
    folder,_=jobs_for('dev')[0];folder=Path(folder)
    legal=read(folder/'legal_root.json.gz');_,_,branch=selected_branch(folder)
    ctx=OnlineContext.from_raw(legal['history'],legal['partial_graph'],legal['history_actions']).with_association()
    model=HistoryConditionedWorldModel(OUT/'experiments/pilot_k_bounded_unique/best.pt',history_fraction=.5)
    for node in branch[:3]:
        previous=ctx;before=ctx._positive_history.timestamps.copy()
        result=model.predict(ctx,node['observation'],executed_action=node['action'],seed=47)
        ctx=model.last_context
        np.testing.assert_array_equal(previous._positive_history.timestamps,before)
        expected=history_from_context(ctx)
        np.testing.assert_array_equal(ctx._positive_history.positive_xyz,expected.positive_xyz)
        np.testing.assert_array_equal(ctx._positive_history.positive_offsets,expected.positive_offsets)
        assert ctx._association_graph.timestamp==node['observation']['timestamp']
        assert model.last_profile['complete_ms']>=model.last_profile['input_graph_history_ms']
        assert ctx._association_graph.feasible(result.positions[:,0],result.present).all()
        assert result.association.physical_seen.shape==result.present.shape
        person=result.association.event_person[:,result.source_event_indices]
        assert result.present[np.arange(32)[:,None],person].all()
        # Every present physical future stays within the certified speed bound.
        speed=np.linalg.norm(np.diff(result.positions,axis=1),axis=-1)/np.diff(result.times)[None,:,None]
        assert np.max(speed*result.present[:,None])<=3.+1e-10
        # Same sampled human futures can be consumed arbitrarily often without
        # model inference or observation/belief mutation.
        assert not result.positions.flags.writeable
        assert len(ctx.history)==len(previous.history)+1
