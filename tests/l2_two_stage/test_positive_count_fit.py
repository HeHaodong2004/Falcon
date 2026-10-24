"""Root/anchor weighting, fold/provenance guards and exact restart identity."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eventwm_graph.l2_two_stage import fit_positive_count as trainer
from eventwm_graph.l2_two_stage.association import AssociationGraph
from eventwm_graph.l2_two_stage.common import OUT,sha
from eventwm_graph.l2_two_stage.data import load_records
from eventwm_graph.l2_two_stage.future_population_targets import attach_targets
from eventwm_graph.l2_two_stage.last_event_metadata import attach_metadata
from eventwm_graph.l2_two_stage.positive_history import attach_histories


def toy(monkeypatch):
    monkeypatch.setattr(trainer,'root_validation',lambda root:root.startswith('v'))
    # One fitting and one held Root are wholly H0. A different Root loses all
    # current detections after an actual positive history and remains active.
    structure={'a':[0,1,2],'b':[1,1],'c':[0,0,0,0],'va':[0,1,1],'vb':[0,0]}
    records=[];graphs={}
    for root,counts in structure.items():
        for anchor,count in enumerate(counts):
            r=dict(root_id=root,scene_id='scene'+root,anchor_index=anchor,timestamp=float(anchor),
                weight=999.,current_positive=0,
                positive_history=SimpleNamespace(positive_offsets=np.array([0,count])))
            records.append(r)
            graph=SimpleNamespace(detections=count,minimum_current_count=int(count>0),validate_certificate=lambda:None)
            graphs[root,anchor]=graph
    return records,graphs,trainer.prepare_schedule(records,graphs)


def test_active_root_cycling_weights_equal_original_all_root_anchor_objective(monkeypatch):
    records,graphs,plan=toy(monkeypatch)
    assert len(plan.all_fit)==3 and len(plan.active_fit)==2 and plan.cycle==2
    assert len(plan.all_validation)==2 and len(plan.validation)==2
    expected=sum(sum((r['anchor_index']+1) for r in rows if graphs[root,r['anchor_index']].detections)/len(rows)
                 for root,rows in plan.all_fit.items())/len(plan.all_fit)
    values=[];seen={root:set() for root in plan.active_fit}
    for epoch in range(1,plan.cycle+1):
        chosen=plan.epoch(epoch)
        assert {r['root_id'] for r in chosen}==set(plan.active_fit)
        assert all(graphs[r['root_id'],r['anchor_index']].minimum_current_count>0 for r in chosen)
        values.append(np.mean([r['weight']*(r['anchor_index']+1) for r in chosen]))
        for r in chosen:
            root=r['root_id'];seen[root].add(r['anchor_index'])
            assert r['weight']==pytest.approx((2/3)*len(plan.active_fit[root])/len(plan.all_fit[root]))
    assert np.mean(values)==pytest.approx(expected,abs=1e-15)
    assert all(seen[root]=={r['anchor_index'] for r in rows} for root,rows in plan.active_fit.items())
    assert all(r['weight']==999. for r in records) # No caller-record mutation.
    assert all(r['weight']==1/3 for r in plan.validation)
    # Validation denominator includes the wholly-H0 held Root. Its omitted
    # correction objective is constant, not a silently removed fitting Root.
    assert sum(r['weight'] for r in plan.validation)/len(plan.all_validation)==pytest.approx(1/3)


def test_modes_share_exact_order_weights_and_restartable_anchor_cycle(monkeypatch):
    _,_,plan=toy(monkeypatch)
    def order(seed,start,stop):
        rng=np.random.RandomState(seed);rows=[]
        for epoch in range(1,stop+1):
            chosen=plan.epoch(epoch);permutation=rng.permutation(len(chosen))
            if epoch>=start:rows.append([(chosen[i]['root_id'],chosen[i]['anchor_index'],chosen[i]['weight']) for i in permutation])
        return rows
    assert order(413,1,64)==order(413,1,64)
    assert order(413,19,64)==order(413,1,64)[18:]
    assert plan.identities()['inactive_fit_roots']==1


@pytest.mark.parametrize('change',['duplicate','clock','scene','history_graph','minimum'])
def test_invalid_or_noncausal_prepared_support_is_rejected(monkeypatch,change):
    records,graphs,_=toy(monkeypatch);records=copy.deepcopy(records)
    if change=='duplicate':records.append(copy.deepcopy(records[0]))
    elif change=='clock':records[1]['timestamp']=records[0]['timestamp']
    elif change=='scene':records[1]['scene_id']='different'
    elif change=='history_graph':records[1]['positive_history'].positive_offsets[-1]=7
    else:graphs['a',1].minimum_current_count=0
    with pytest.raises(ValueError):trainer.prepare_schedule(records,graphs)


@pytest.mark.parametrize('change',[dict(mode='oracle'),dict(split='dev'),dict(sidecar='dev'),dict(epochs=63),
    dict(seed=True),dict(batch_size=24),dict(learning_rate=.002),dict(worlds=4),dict(validation_worlds=8),dict(full_fit=1)])
def test_predeclared_configuration_is_not_an_implicit_tuning_interface(change):
    settings=dict(mode='pooled',split='train_pilot_native_v2',sidecar='train_pilot_conditioned')
    settings.update(change)
    with pytest.raises(ValueError):trainer.validate_settings(**settings)


def test_exact_resume_rejects_model_data_optimizer_and_partial_epoch_changes():
    config=dict(epochs=64,name='new',mode='pooled',source_identity='frozen',model_config=dict(width=48),
                data_manifest_sha256='prepared',learning_rate=.001,iid_worlds=8,validation_iid_worlds=32,
                environment=dict(numpy='fixed',torch='fixed'),analytic_current_cover=False)
    previous=dict(config=dict(config,name='old'),epoch=12,resume_exact=True)
    trainer.verify_resume(previous,config)
    for field,value in [('mode','intercept'),('source_identity','changed'),('data_manifest_sha256','other'),
                         ('learning_rate',.002),('iid_worlds',4),('model_config',dict(width=24)),
                         ('environment',dict(numpy='changed',torch='fixed')),('analytic_current_cover',True)]:
        bad=copy.deepcopy(previous);bad['config'][field]=value
        with pytest.raises(ValueError,match='resume identity'):trainer.verify_resume(bad,config)
    for epoch,exact in [(64,True),(12,False),(-1,True),(True,True)]:
        with pytest.raises(ValueError):trainer.verify_resume(dict(previous,epoch=epoch,resume_exact=exact),config)


def test_historical_resume_uses_preserved_best_epoch_not_a_future_best_alias(tmp_path):
    learning=[dict(epoch=1,validation_score=.1),dict(epoch=2,validation_score=.7),dict(epoch=3,validation_score=.8)]
    selected=tmp_path/'best_epoch002.pt';torch.save(dict(epoch=2),selected)
    torch.save(dict(epoch=17),tmp_path/'best.pt')
    assert trainer.historical_best_path(tmp_path,learning,2)==selected
    assert trainer.historical_best_path(tmp_path,learning[:1],2) is None
    torch.save(dict(epoch=3),selected)
    with pytest.raises(ValueError,match='wrong epoch'):trainer.historical_best_path(tmp_path,learning,2)
    selected.unlink()
    with pytest.raises(ValueError,match='immutable historical'):trainer.historical_best_path(tmp_path,learning,2)


def test_importance_weights_retain_float64_in_the_double_precision_proper_objective(monkeypatch):
    monkeypatch.setattr(trainer,'make_batch',lambda rows,device:dict(weight=torch.ones(len(rows))))
    rows=[dict(weight=4/9),dict(weight=2/3)]
    batch=trainer.weighted_batch(rows,device='cpu')
    assert batch['weight'].dtype==torch.float64
    assert batch['weight'].numpy().tobytes()==np.array([4/9,2/3],np.float64).tobytes()
    for weight in (0.,-1.,float('nan')):
        with pytest.raises(ValueError):trainer.weighted_batch([dict(weight=weight)],device='cpu')


@pytest.fixture(scope='module')
def actual_plan():
    records=load_records('train_pilot_native_v2')
    audits=dict(history=attach_histories(records,'train_pilot_conditioned'),events=attach_metadata(records,'train_pilot_conditioned'),
                targets=attach_targets(records,'train_campaign_20261023'))
    graphs={};previous={}
    for r in sorted(records,key=lambda r:(r['root_id'],r['timestamp'],r['anchor_index'])):
        root=r['root_id'];h=r['positive_history']
        g=previous[root].append(h) if root in previous else AssociationGraph.from_history(h)
        graphs[root,r['anchor_index']]=g;previous[root]=g
    return records,audits,trainer.prepare_schedule(records,graphs)


def test_actual_complete_T_prepared_fold_and_joins_bind_every_root(actual_plan):
    _,audits,plan=actual_plan
    checkpoint=OUT/'experiments/pilot_t_empty_history_affine/best.pt'
    saved=torch.load(checkpoint,map_location='cpu')
    prepared=sha(OUT/'prepared/train_pilot_native_v2/manifest.json')
    trainer.validate_base_provenance(saved,prepared,audits,plan,'train_pilot_native_v2','train_pilot_conditioned')
    identity=plan.identities()
    assert identity['fit_roots']==1657 and identity['fit_anchors']==5065
    assert identity['validation_roots']==391 and identity['validation_anchors']==1214
    assert identity['inactive_fit_roots']>0 and identity['active_fit_anchors']<identity['fit_anchors']
    assert identity['minimum_selection_epoch']<=64
    for field in ('data_manifest_sha256','fitting_Root_ids_sha256','fit_anchors'):
        tampered=copy.deepcopy(saved);tampered['config'][field]='changed'
        with pytest.raises(ValueError):trainer.validate_base_provenance(tampered,prepared,audits,plan,'train_pilot_native_v2','train_pilot_conditioned')
    changed=copy.deepcopy(audits);changed['history']['manifest_sha256']='changed'
    with pytest.raises(ValueError):trainer.validate_base_provenance(saved,prepared,changed,plan,'train_pilot_native_v2','train_pilot_conditioned')


@pytest.mark.parametrize('mode',['pooled','intercept'])
def test_actual_T_initialization_copies_every_base_tensor_and_trains_only_two_new_tensors(mode):
    torch.set_num_threads(1)
    saved=torch.load(OUT/'experiments/pilot_t_empty_history_affine/best.pt',map_location='cpu')
    model,config=trainer.initialize_model(saved,mode,device='cpu')
    assert config['positive_history_count_affine']==mode
    assert len(saved['model'])==19
    assert tuple(name for name,p in model.named_parameters() if p.requires_grad)==trainer.HEAD
    for name,value in saved['model'].items():assert torch.equal(model.state_dict()[name],value)
    assert torch.count_nonzero(model.positive_count_affine.weight)==0
    assert torch.count_nonzero(model.positive_count_affine.bias)==0
    bad=copy.deepcopy(saved);bad['model'].pop('empty_count_affine.bias')
    with pytest.raises(ValueError,match='exactly the two'):trainer.initialize_model(bad,mode,device='cpu')
