"""Ordered arrived-cell columns preserve scalar facts and immutable ownership."""
import copy
from pathlib import Path
import numpy as np
import pytest
from eventwm_graph.l2_two_stage.context import OnlineContext
from eventwm_graph.l2_two_stage.data import jobs_for,read,selected_branch
from eventwm_graph.l2_two_stage.features import extract
from eventwm_graph.l2_two_stage.graphfacts import OrderedCellFacts
from eventwm_graph.l2_two_stage.types import LegalContext


def actual():
    folder,_=jobs_for('dev')[0];folder=Path(folder)
    legal=read(folder/'legal_root.json.gz');branch=selected_branch(folder)[2]
    legal['partial_graph']['cells'].reverse() # Legal caller order is significant.
    return OnlineContext.from_raw(legal['history'],legal['partial_graph'],legal['history_actions']),branch


def assert_columns(context):
    rows=context.partial_graph['cells'];facts=context.cell_facts
    np.testing.assert_array_equal(facts.states,[r['state'] for r in rows])
    np.testing.assert_array_equal(facts.last_seen,[r['last_seen'] for r in rows])
    assert facts.order==tuple(tuple(r['cell']) for r in rows)
    fast=extract(context);replay=extract(LegalContext(context.history,context.actions,context.partial_graph))
    for field in ('x','graph','global_graph','global_features','origin','velocity','age'):
        assert fast[field].dtype==replay[field].dtype
        assert fast[field].tobytes()==replay[field].tobytes()


def test_native_incremental_columns_order_features_and_old_state_are_exact():
    context,branch=actual();assert_columns(context)
    for node in branch[:8]:
        old=context.cell_facts;before=(old.states.tobytes(),old.last_seen.tobytes(),old.order)
        context=context.ingest(node['observation'],node['action']);assert_columns(context)
        assert before==(old.states.tobytes(),old.last_seen.tobytes(),old.order)
        assert context.processing_ms['cell_facts_update_ms']>=0
    for array in (context.cell_facts.states,context.cell_facts.last_seen):
        with pytest.raises(ValueError):array.setflags(write=True)
    with pytest.raises(TypeError):context.cell_facts.index[(0,0)]=4
    with pytest.raises(ValueError):OrderedCellFacts().validate_certificate()


def test_final_cells_control_duplicate_occupied_and_ego_support_updates():
    context,branch=actual();node=branch[0];ob=copy.deepcopy(node['observation']);clock=ob['timestamp']
    key=(1000001,1000001)
    base=dict(cell=list(key),height=context.partial_graph['height'],timestamp=clock)
    ob['static_updates'] += [dict(base,state=1,source='visible_native_cspace_square_and_clearance_halo'),
        dict(base,state=2,source='observed_static_wall_inflated_025m'),
        dict(base,state=1,source='visible_native_cspace_square_and_clearance_halo'),
        dict(cell=[1000002,1000002],height=base['height'],timestamp=clock,state=1,source='occupied_ego_base_support')]
    result=context.ingest(ob,node['action']);assert_columns(result)
    index=result.cell_facts.index[key]
    assert result.cell_facts.states[index]==2
    assert result.cell_facts.last_seen[index]==clock
    assert (1000002,1000002) not in result.cell_facts.index


def test_internal_successor_rejects_reordered_or_omitted_cell_columns():
    context,_=actual();facts=context.cell_facts
    with pytest.raises(ValueError):facts.update(facts.order[::-1],context.certified_cells,())
    with pytest.raises(ValueError):facts.update(facts.order[:-1],context.certified_cells,())
