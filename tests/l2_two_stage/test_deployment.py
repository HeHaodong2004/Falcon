"""Startup provenance checks cannot confer acceptance or alter inference."""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from eventwm_graph.l2_two_stage import deployment
from eventwm_graph.l2_two_stage.common import OUT


PREPARED='1'*64
CAMPAIGN='2'*64
SOURCE='3'*64


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    """Use real loader guards and hashes, replacing only neural construction.

    Tiny synthetic bundles are acceptance-guard counterexamples, not accepted
    research artifacts. No forecasting or collection is performed by this suite.
    """
    source=tmp_path/'runtime_source'
    files={}
    for name,module in (('conditioned',deployment.conditioned),('deployment',deployment)):
        relative='eventwm_graph/l2_two_stage/'+name+'.py'
        target=source/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(module.__file__).read_bytes())
        files[relative]=sha(target)
    (source/'manifest.json').write_text(json.dumps(dict(identity=SOURCE,files=files)))
    monkeypatch.setattr(deployment.conditioned,'__file__',str(source/'eventwm_graph/l2_two_stage/conditioned.py'))
    monkeypatch.setattr(deployment.torch,'get_num_threads',lambda:1)
    constructed=[]

    class Network:
        def requires_grad_(self,value):self.requires_grad=value

    class Model:
        def __init__(self,checkpoint):
            constructed.append(Path(checkpoint));self.network=Network();self.calls=[]
        def predict(self,*args,**kwargs):
            self.calls.append((args,kwargs));return 'complete-output-sentinel'

    monkeypatch.setattr(deployment.conditioned,'HistoryConditionedWorldModel',Model)
    return constructed


def bundle(tmp_path,checkpoint_value=None,checkpoint_path=None,prepared=PREPARED):
    target=tmp_path/'bundle';target.mkdir()
    checkpoint=target/'checkpoint.pt'
    if checkpoint_path is None:
        torch.save(dict(config=dict(data_manifest_sha256=PREPARED)) if checkpoint_value is None else checkpoint_value,checkpoint)
    else:checkpoint.write_bytes(Path(checkpoint_path).read_bytes())
    config=deployment.Stage1InferenceConfiguration(sha(checkpoint),SOURCE,prepared)
    config_path=target/'INFERENCE_CONFIG.json'
    config_path.write_text(json.dumps(config.to_dict()))
    gate=dict(decision='STAGE1_PASS',gates={k:dict(status='PASS') for k in 'ABCD'},
        data=dict(manifest_sha256=CAMPAIGN),
        selected_configuration=dict(inference_config_sha256=sha(config_path),
            checkpoint_sha256=config.checkpoint_sha256,source_identity=SOURCE,
            worlds=40,training_manifest_sha256=prepared))
    (target/'STAGE1_GATE.json').write_text(json.dumps(gate))
    return target,config,gate


def write_gate(target,gate):
    (target/'STAGE1_GATE.json').write_text(json.dumps(gate))


def test_accepted_bundle_binds_prepared_manifest_and_keeps_campaign_distinct(tmp_path,runtime,monkeypatch):
    target,config,gate=bundle(tmp_path)
    assert gate['data']['manifest_sha256']!=config.training_manifest_sha256
    predictor=deployment.load_accepted_stage1(target)
    assert runtime==[target/'checkpoint.pt']
    assert predictor._model.network.requires_grad is False
    # All metadata loading is startup-only; every call still requests the full
    # declared40-world output from the unchanged model.
    monkeypatch.setattr(deployment.torch,'load',lambda *a,**k:pytest.fail('checkpoint reread during prediction'))
    assert predictor.predict('old-context','new-frame',seed=731)=='complete-output-sentinel'
    assert predictor._model.calls==[(('old-context','new-frame'),dict(partial_graph=None,
        horizon=3.,worlds=40,seed=731,actions=(),executed_action=None))]


@pytest.mark.parametrize('metadata',[{},dict(config=None),dict(config={}),
    dict(config=dict(data_manifest_sha256=None)),dict(config=dict(data_manifest_sha256=7)),
    dict(config=dict(data_manifest_sha256='a'*63)),dict(config=dict(data_manifest_sha256='A'*64))])
def test_missing_or_malformed_checkpoint_training_identity_fails_before_model(tmp_path,runtime,metadata):
    target,config,_=bundle(tmp_path,checkpoint_value=metadata)
    with pytest.raises(ValueError,match='prepared-training manifest SHA256'):
        deployment.ConfiguredStage1Predictor(target/'checkpoint.pt',config)
    assert runtime==[]


def test_self_consistent_config_and_gate_cannot_substitute_campaign_for_training(tmp_path,runtime):
    target,_,_=bundle(tmp_path,prepared=CAMPAIGN)
    with pytest.raises(ValueError,match='configured prepared-training manifest'):
        deployment.load_accepted_stage1(target)
    assert runtime==[]


@pytest.mark.parametrize('value',[None,CAMPAIGN,'4'*64,{},'a'*63])
def test_gate_requires_its_own_exact_prepared_manifest_binding(tmp_path,runtime,value):
    target,_,gate=bundle(tmp_path)
    if value is None:gate['selected_configuration'].pop('training_manifest_sha256')
    else:gate['selected_configuration']['training_manifest_sha256']=value
    # Neither an ambiguous campaign field nor a separate unselected field can
    # replace the checkpoint-specific accepted-configuration binding.
    gate['data']['training_manifest_sha256']=PREPARED
    write_gate(target,gate)
    with pytest.raises(ValueError,match='accepted gate does not bind'):
        deployment.load_accepted_stage1(target)
    assert runtime==[]


@pytest.mark.parametrize('failed_gate',list('ABCD')+[None])
def test_matching_provenance_never_bypasses_acceptance(tmp_path,runtime,failed_gate):
    target,_,gate=bundle(tmp_path)
    if failed_gate is None:gate['decision']='STAGE1_REVISE'
    else:gate['gates'][failed_gate]['status']='INCONCLUSIVE'
    write_gate(target,gate)
    with pytest.raises(ValueError,match='not accepted'):
        deployment.load_accepted_stage1(target)
    assert runtime==[]


def test_real_checkpoint_prepared_manifest_matches_actual_manifest_bytes(tmp_path,runtime):
    checkpoint=Path(os.environ.get('EVENTWM_STAGE1_TEST_CHECKPOINT',
        str(OUT/'experiments/pilot_s_count_likelihood32/best.pt')))
    assert checkpoint.is_file(), 'real research checkpoint is required for this provenance test'
    saved=torch.load(checkpoint,map_location='cpu')
    prepared=sha(OUT/'prepared'/saved['config']['split']/'manifest.json')
    assert prepared==saved['config']['data_manifest_sha256']
    target,config,_=bundle(tmp_path,checkpoint_path=checkpoint,prepared=prepared)
    predictor=deployment.load_accepted_stage1(target)
    assert predictor.configuration.training_manifest_sha256==prepared
    assert runtime==[target/'checkpoint.pt']


def test_checkpoint_replacement_remains_rejected_before_metadata_load(tmp_path,runtime):
    target,config,_=bundle(tmp_path)
    torch.save(dict(config=dict(data_manifest_sha256=PREPARED),tampered=True),target/'checkpoint.pt')
    with pytest.raises(ValueError,match='configured artifact'):
        deployment.ConfiguredStage1Predictor(target/'checkpoint.pt',config)
    assert runtime==[]
