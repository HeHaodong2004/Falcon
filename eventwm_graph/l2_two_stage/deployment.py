"""Configuration-bound Stage1 inference and an acceptance-checked release loader.

The core research predictor retains its historical sampling default. This small
adapter always constructs the complete bank declared by its validated config.
It adds no forecast, association, geometry, observation or planning algorithm.
"""
from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path
import sys

import torch
from eventwm_graph.l2_two_stage import conditioned


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _is_sha256(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _checkpoint_training_manifest(checkpoint):
    """Read the prepared-training identity, not the sealed campaign identity.

    Training writes the SHA256 of its prepared manifest to
    ``checkpoint['config']['data_manifest_sha256']``. The source campaign and
    prepared manifest describe different artifacts and must not be substituted.
    This metadata check belongs to construction, never the prediction path.
    """
    saved=torch.load(checkpoint,map_location='cpu')
    config=saved.get('config') if isinstance(saved,dict) else None
    identity=config.get('data_manifest_sha256') if isinstance(config,dict) else None
    if not _is_sha256(identity):
        raise ValueError('checkpoint requires a prepared-training manifest SHA256')
    return identity


@dataclass(frozen=True)
class Stage1InferenceConfiguration:
    checkpoint_sha256: str
    source_identity: str
    # SHA256 of the prepared manifest used by training; not the source campaign.
    training_manifest_sha256: str
    worlds: int = 40
    time_samples: int = 64
    max_horizon_seconds: float = 3.0
    cpu_threads: int = 1
    version: str = 'stage1_full_population_inference_1'

    def __post_init__(self):
        for value in (self.checkpoint_sha256,self.source_identity,self.training_manifest_sha256):
            if not _is_sha256(value):
                raise ValueError('configuration requires complete lowercase SHA256 identities')
        if (type(self.worlds) is not int or self.worlds!=40 or
                type(self.time_samples) is not int or self.time_samples!=64 or
                type(self.cpu_threads) is not int or self.cpu_threads!=1 or
                type(self.max_horizon_seconds) not in (int,float) or self.max_horizon_seconds!=3. or
                self.version!='stage1_full_population_inference_1'):
            raise ValueError('this inference configuration requires40worlds/64times/3seconds/CPU1')

    def to_dict(self):return asdict(self)

    @classmethod
    def from_file(cls,path):
        value=json.loads(Path(path).read_text())
        if not isinstance(value,dict):raise ValueError('inference configuration must be an object')
        try:return cls(**value)
        except TypeError as error:raise ValueError('unknown or missing inference configuration fields') from error


class ConfiguredStage1Predictor:
    """An explicit40-world public API; construction alone is not acceptance."""

    def __init__(self,checkpoint,configuration):
        if not isinstance(configuration,Stage1InferenceConfiguration):
            raise TypeError('a versioned Stage1InferenceConfiguration is required')
        if torch.get_num_threads()!=configuration.cpu_threads:
            raise ValueError('initialize the inference process with one Torch CPU thread')
        if _sha(checkpoint)!=configuration.checkpoint_sha256:
            raise ValueError('checkpoint differs from the configured artifact')
        if _checkpoint_training_manifest(checkpoint)!=configuration.training_manifest_sha256:
            raise ValueError('checkpoint differs from the configured prepared-training manifest')
        package=Path(conditioned.__file__).resolve().parent
        source=package.parents[1];manifest_path=source/'manifest.json'
        if not manifest_path.is_file():raise ValueError('load the configured immutable runtime source')
        manifest=json.loads(manifest_path.read_text())
        if manifest['identity']!=configuration.source_identity:
            raise ValueError('runtime differs from the configured executable source')
        for relative,expected in manifest['files'].items():
            if _sha(source/relative)!=expected:raise ValueError('changed immutable source: '+relative)
        # Catch an already imported module from a different tree as well as a
        # changed file on disk. The optional adapter itself is artifact-bound.
        for name,module in tuple(sys.modules.items()):
            path=getattr(module,'__file__',None)
            if not name.startswith('eventwm_graph.l2_two_stage.') or not path:continue
            relative='eventwm_graph/l2_two_stage/'+name.rsplit('.',1)[-1]+'.py'
            if relative in manifest['files'] and _sha(path)!=manifest['files'][relative]:
                raise ValueError('mixed runtime module: '+name)
        self.configuration=configuration
        self._model=conditioned.HistoryConditionedWorldModel(checkpoint)
        self._model.network.requires_grad_(False)

    def predict(self,history,obs,partial_graph=None,horizon=3.,seed=None,actions=(),executed_action=None):
        if torch.get_num_threads()!=self.configuration.cpu_threads:
            raise ValueError('configured inference requires one Torch CPU thread')
        return self._model.predict(history,obs,partial_graph=partial_graph,horizon=horizon,
            worlds=self.configuration.worlds,seed=seed,actions=actions,executed_action=executed_action)

    @property
    def last_context(self):return self._model.last_context

    @property
    def last_profile(self):return self._model.last_profile


def load_accepted_stage1(bundle):
    """Load only an accepted bundle, preserving its exact source and bank size.

    Put ``bundle/runtime_source`` and the bundled API module on PYTHONPATH before
    importing this adapter. Startup certification/hashing is a one-time cost.

    ``selected_configuration.training_manifest_sha256`` must bind the prepared
    training manifest. ``data.manifest_sha256`` can identify the sealed source
    campaign and is deliberately not an alternative for that required binding.
    """
    bundle=Path(bundle);config_path=bundle/'INFERENCE_CONFIG.json'
    config=Stage1InferenceConfiguration.from_file(config_path)
    gate=json.loads((bundle/'STAGE1_GATE.json').read_text())
    if gate.get('decision')!='STAGE1_PASS' or any(gate.get('gates',{}).get(key,{}).get('status')!='PASS' for key in 'ABCD'):
        raise ValueError('Stage1 release is not accepted')
    selected=gate.get('selected_configuration',{})
    if (selected.get('inference_config_sha256')!=_sha(config_path) or
            selected.get('checkpoint_sha256')!=config.checkpoint_sha256 or
            selected.get('source_identity')!=config.source_identity or selected.get('worlds')!=config.worlds or
            selected.get('training_manifest_sha256')!=config.training_manifest_sha256):
        raise ValueError('accepted gate does not bind this inference configuration')
    return ConfiguredStage1Predictor(bundle/'checkpoint.pt',config)
