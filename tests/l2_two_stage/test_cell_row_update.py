"""Actual-frame and adversarial parity for fused arrived-cell updates."""
import ast
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from eventwm_graph.l2_two_stage import context as context_module
from eventwm_graph.l2_two_stage.common import BASE,OUT
from eventwm_graph.l2_two_stage.context import OnlineContext
from eventwm_graph.l2_two_stage.data import jobs_for,read,selected_branch
from eventwm_graph.l2_two_stage.features import extract
from eventwm_graph.l2_two_stage.conditioned import HistoryConditionedWorldModel


def reference_ingest():
    source=OUT/'frozen_sources/aa0b017fc297407d1f530531b44ecbe2ce00008a741ddb45c92f71f97609b121/eventwm_graph/l2_two_stage/context.py'
    cls=next(node for node in ast.parse(source.read_text()).body if isinstance(node,ast.ClassDef) and node.name=='OnlineContext')
    node=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='ingest')
    scope=dict(vars(context_module));exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    return scope['ingest']


def equivalent(old,new):
    assert old.partial_graph==new.partial_graph
    assert isinstance(new.partial_graph['cells'],tuple)
    assert old.actions==new.actions and old._cell_order==new._cell_order
    assert old.legal_history.revision==new.legal_history.revision
    assert old.legal_history.cells==new.legal_history.cells
    for field in ('states','last_seen'):
        assert getattr(old.cell_facts,field).tobytes()==getattr(new.cell_facts,field).tobytes()
    assert dict(old.cell_facts.index)==dict(new.cell_facts.index)
    a,b=extract(old),extract(new)
    for key in ('x','graph','global_graph','global_features','origin','velocity','age','last_positions','last_times','robotxy'):
        assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape and a[key].tobytes()==b[key].tobytes()


@pytest.fixture(scope='module')
def real_cases():
    torch.set_num_threads(1)
    selections=[('ffccd1e26a606a0c92cbd17fef7b7043fe25f44d26008e6a4be24f3964debc3b',117),
        ('e4445c0e408b3377da67ca62919dcbe3ba68d1554adc876c750dd4653beb2c50',41),
        ('b4436f4b2eb9d2b260123cac159611aa893ab882b2f8c53e13646d7bfd7c806b',10)]
    folders={row['root_id']:Path(folder) for folder,row in jobs_for('dev')}
    for folder,row in jobs_for('train_campaign_20261023'):
        if row['root_id'] in {root for root,_ in selections}:folders[row['root_id']]=Path(folder)
    results=[]
    for root,anchor in selections:
        legal=read(folders[root]/'legal_root.json.gz');legal['partial_graph']['cells'].reverse()
        branch=selected_branch(folders[root])[2]
        context=OnlineContext.from_raw(legal['history'],legal['partial_graph'],legal['history_actions']).with_association()
        for node in branch[:anchor-1]:context=context.ingest(node['observation'],node['action'])
        results.append((context,branch[anchor-1]))
    return results


@pytest.mark.parametrize('case',[0,1,2])
def test_actual_updates_keep_exact_rows_facts_features_and_worlds(real_cases,case):
    context,node=real_cases[case];old_rows=context.partial_graph['cells'];snapshot=tuple(dict(row) for row in old_rows)
    prior_arrays=(context.cell_facts.states.tobytes(),context.cell_facts.last_seen.tobytes())
    old=reference_ingest()(context,node['observation'],node['action'])
    new=context.ingest(node['observation'],node['action']);equivalent(old,new)
    assert snapshot==tuple(dict(row) for row in old_rows)
    assert prior_arrays==(context.cell_facts.states.tobytes(),context.cell_facts.last_seen.tobytes())
    with pytest.raises(ValueError):new.cell_facts.states.setflags(write=True)
    with pytest.raises(ValueError):new.cell_facts.last_seen.setflags(write=True)
    with pytest.raises(TypeError):new.partial_graph['cells'][0]['last_seen']=-1
    model=HistoryConditionedWorldModel(OUT/'experiments/full_refit_o_guided_campaign/last.pt')
    a=model.predict(old,old.history[-1],worlds=40);b=model.predict(new,new.history[-1],worlds=40)
    for field in ('positions','present','weights','times','source_event_indices','previously_observed'):
        assert getattr(a,field).tobytes()==getattr(b,field).tobytes()
    for field in ('matching','event_person','physical_seen','row_order','first_edge','auxiliary_log_probability'):
        assert getattr(a.association,field).tobytes()==getattr(b.association,field).tobytes()


def test_new_indices_follow_final_insert_order_not_raw_first_appearance(real_cases):
    context,node=real_cases[0];ob=copy.deepcopy(node['observation']);clock=ob['timestamp']
    a,b,c=(10000001,10000001),(10000002,10000002),(10000003,10000003)
    base=dict(height=context.partial_graph['height'],timestamp=clock)
    ob['static_updates']=[dict(base,cell=list(a),state=1,source='visible_native_cspace_square_and_clearance_halo'),
        dict(base,cell=list(b),state=1,source='visible_native_cspace_square_and_clearance_halo'),
        dict(base,cell=list(a),state=2,source='observed_static_wall_inflated_025m'),
        dict(base,cell=list(c),state=1,source='occupied_ego_base_support'),
        dict(base,cell=list(a),state=1,source='visible_native_cspace_square_and_clearance_halo')]
    old=reference_ingest()(context,ob,node['action']);new=context.ingest(ob,node['action']);equivalent(old,new)
    assert new._cell_order[-2:]==(b,a) and c not in new.certified_cells
    assert new.certified_cells[a]['state']==2
    ob['static_updates'][0]['height']=9999
    assert new.certified_cells[a]['height']==base['height']


@pytest.mark.parametrize('invalid',[False,True])
def test_empty_or_invalid_frame_preserves_old_rows_but_advances_clock(real_cases,invalid):
    context,node=real_cases[0];ob=copy.deepcopy(node['observation']);ob['static_updates']=[]
    if invalid:ob['observation_valid']=False
    old=reference_ingest()(context,ob,node['action']);new=context.ingest(ob,node['action']);equivalent(old,new)
    assert new._cell_order==context._cell_order and new.partial_graph['cells']==context.partial_graph['cells']
    assert new.partial_graph['timestamp']==ob['timestamp']


def test_changed_height_and_occupied_revisions_invalidate_same_topology(real_cases):
    context,node=real_cases[0];ob=copy.deepcopy(node['observation']);clock=ob['timestamp']
    nodes=context.partial_graph['global_nodes'];assert nodes
    chosen=next(iter(nodes.values()));key=tuple(chosen['cell']);height=context.certified_cells[key]['height']
    for state,newheight,source in ((1,height+10.,'visible_native_cspace_square_and_clearance_halo'),
                                  (2,height,'observed_static_wall_inflated_025m')):
        ob['static_updates']=[dict(cell=list(key),height=newheight,state=state,source=source,timestamp=clock)]
        old=reference_ingest()(context,ob,node['action']);new=context.ingest(ob,node['action']);equivalent(old,new)
        assert chosen['id'] not in new.partial_graph['global_nodes']
        assert new.certified_cells[key]['height']==newheight and new.certified_cells[key]['state']==state


def test_internal_fused_update_rejects_wrong_shape_or_cell_order(real_cases):
    context,_=real_cases[0];facts=context.cell_facts
    for order,rows in ((facts.order[::-1],context.partial_graph['cells']),
                       (facts.order,context.partial_graph['cells'][:-1]),
                       (facts.order,list(context.partial_graph['cells']))):
        with pytest.raises(ValueError):facts.update_with_rows(order,context.certified_cells,(),rows)
