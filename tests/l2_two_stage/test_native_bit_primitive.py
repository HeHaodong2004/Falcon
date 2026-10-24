"""Exact hardware bit primitive against the frozen six-test matcher oracle."""
import ast
from pathlib import Path
import numpy as np
import pytest
import torch
from numba import njit
from eventwm_graph.l2_two_stage import association,guided_association as guided
from eventwm_graph.l2_two_stage.common import OUT
from eventwm_graph.l2_two_stage.context import OnlineContext
from eventwm_graph.l2_two_stage.conditioned import HistoryConditionedWorldModel
from eventwm_graph.l2_two_stage.data import jobs_for,read,selected_branch


@pytest.fixture(scope='module')
def reference():
    source=OUT/'frozen_sources/8bf0c8025608fab6f22d8b1f0abd0c54e32200600df05fade29f4dcd7b139bc7/eventwm_graph/l2_two_stage/association.py'
    tree=ast.parse(source.read_text());scope=dict(vars(association))
    for name in ('_trailing_bit','_first_unexcluded','_bitset_augment','_bitset_lineages'):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);node.decorator_list=[]
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        scope[name]=njit(cache=False)(scope[name])
    return scope


def test_uint64_lowest_bit_zero_highbit_and_random_values_match(reference):
    old=reference['_trailing_bit'];new=association._trailing_bit
    values=np.r_[np.array([0,np.iinfo(np.uint64).max],np.uint64),
        np.array([1<<i for i in range(64)],np.uint64),np.random.default_rng(917).bit_generator.random_raw(100000)]
    for value in values:assert old(value)==new(value)
    assert new(np.uint64(0))==63 and new(np.uint64(1)<<np.uint64(63))==63


def test_exhaustive_small_words_and_full_word_boundaries_keep_candidate_order(reference):
    old=reference['_first_unexcluded'];new=association._first_unexcluded
    for bits in range(64):
        for excluded in range(64):
            a=np.array([bits],np.uint64);b=np.array([excluded],np.uint64)
            for first in (-1,0,1,2,3,4,5,6,63,64):assert old(a,b,first)==new(a,b,first)
    rng=np.random.default_rng(29)
    for words in range(10):
        for _ in range(40):
            bits=rng.bit_generator.random_raw(words);excluded=rng.bit_generator.random_raw(words)
            for first in (-1,0,31,32,63,64,65,127,128,64*words):assert old(bits,excluded,first)==new(bits,excluded,first)


@pytest.fixture(scope='module')
def actual_contexts():
    torch.set_num_threads(1)
    selections=[('ffccd1e26a606a0c92cbd17fef7b7043fe25f44d26008e6a4be24f3964debc3b',117),
        ('e4445c0e408b3377da67ca62919dcbe3ba68d1554adc876c750dd4653beb2c50',41),
        ('b4436f4b2eb9d2b260123cac159611aa893ab882b2f8c53e13646d7bfd7c806b',10)]
    folders={row['root_id']:Path(folder) for folder,row in jobs_for('dev')}
    for folder,row in jobs_for('train_campaign_20261023'):
        if row['root_id'] in {root for root,_ in selections}:folders[row['root_id']]=Path(folder)
    results=[]
    for root,anchor in selections:
        legal=read(folders[root]/'legal_root.json.gz');branch=selected_branch(folders[root])[2]
        context=OnlineContext.from_raw(legal['history'],legal['partial_graph'],legal['history_actions']).with_association()
        for node in branch[:anchor-1]:context=context.ingest(node['observation'],node['action'])
        results.append((context,branch[anchor-1]))
    return results


@pytest.mark.parametrize('case',[0,1,2])
@pytest.mark.parametrize('worlds',[32,40])
def test_actual_worlds_auxiliary_probabilities_and_ancestry_are_bitwise(reference,actual_contexts,case,worlds,monkeypatch):
    context,node=actual_contexts[case];candidate=guided._bitset_lineages
    model=HistoryConditionedWorldModel(OUT/'experiments/full_refit_o_guided_campaign/last.pt')
    monkeypatch.setattr(guided,'_bitset_lineages',reference['_bitset_lineages'])
    old=model.predict(context,node['observation'],executed_action=node['action'],worlds=worlds,seed=731015)
    monkeypatch.setattr(guided,'_bitset_lineages',candidate)
    new=model.predict(context,node['observation'],executed_action=node['action'],worlds=worlds,seed=731015)
    for field in ('positions','present','times','weights','source_event_indices','previously_observed'):
        assert getattr(old,field).tobytes()==getattr(new,field).tobytes()
    for field in ('matching','event_person','physical_seen','row_order','first_edge','auxiliary_log_probability'):
        assert getattr(old.association,field).tobytes()==getattr(new.association,field).tobytes()
    assert old.association.kernel==new.association.kernel and dict(old.sampling)==dict(new.sampling)
