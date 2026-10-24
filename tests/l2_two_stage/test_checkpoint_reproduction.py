"""A real completed checkpoint must reproduce its public joint law after reload."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from eventwm_graph.l2_two_stage.common import OUT,sha
from eventwm_graph.l2_two_stage.conditioned import HistoryConditionedWorldModel
from eventwm_graph.l2_two_stage.context import OnlineContext
from eventwm_graph.l2_two_stage.data import jobs_for,read,selected_branch


def test_completed_native_checkpoint_reload_ignores_unrelated_rng_and_context_caches():
    torch.set_num_threads(1)
    checkpoint=Path(os.environ.get('EVENTWM_STAGE1_TEST_CHECKPOINT',str(OUT/'experiments/full_held_o_guided_campaign/best.pt')))
    directory=checkpoint.parent
    complete=json.loads((directory/'complete.json').read_text())
    assert checkpoint.name==complete['selected_checkpoint']
    assert sha(checkpoint)==complete['checkpoint_sha256']
    folder=Path(jobs_for('dev')[0][0]);legal=read(folder/'legal_root.json.gz')
    before=OnlineContext.from_raw(legal['history'],legal['partial_graph'],legal['history_actions']).with_association()
    node=selected_branch(folder)[2][0]
    first=HistoryConditionedWorldModel(checkpoint)
    a=first.predict(before,node['observation'],executed_action=node['action'],seed=731)
    expected_context=first.last_context
    # Restoring learned state and an explicit request seed must suffice. Other
    # callers' NumPy/Torch activity cannot alter this model's sampled world bank.
    numpy_state=np.random.get_state();torch_state=torch.get_rng_state()
    try:
        np.random.seed(991);torch.manual_seed(992)
        np.random.normal(size=1000);torch.randn(1000)
        del first
        second=HistoryConditionedWorldModel(checkpoint)
        b=second.predict(before,node['observation'],executed_action=node['action'],seed=731)
        for key in ('positions','present','weights','times','previously_observed','source_event_indices'):
            np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        for key in ('matching','event_person','physical_seen','row_order','first_edge','auxiliary_log_probability'):
            np.testing.assert_array_equal(getattr(a.association,key),getattr(b.association,key))
        assert dict(a.sampling)==dict(b.sampling)
        assert second.last_context._positive_history.positive_xyz.tobytes()==expected_context._positive_history.positive_xyz.tobytes()
        assert len(before.history)+1==len(second.last_context.history)
        assert a.law.estimate_count_log_probability(int(a.present[0].sum()),seed=733)==b.law.estimate_count_log_probability(int(b.present[0].sum()),seed=733)
    finally:
        np.random.set_state(numpy_state);torch.set_rng_state(torch_state)
