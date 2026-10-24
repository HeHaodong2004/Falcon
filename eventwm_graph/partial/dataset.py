"""Explicit input/target reader for future graph-conditioned learning.

This module builds examples and never trains a model. Inference callers use
``online_input`` directly; privileged targets are opened only by this offline
supervised reader. No sibling branch's future enters the chosen example.
"""
from pathlib import Path
from ..config import read_json
from ..stage1.storage import read_gzip
from .storage import online_input,check_legal,validate_root
from .map import PartialGraph

READER_VERSION='partial_graph_supervised_example_1'


def sampling_metadata(folder):
    """Offline sampling metadata; never attach this to ``online_input``.

    The importance ratio corrects the node mixture before admission only.
    It does not correct legal-history rejection or deterministic query selection.
    Export the campaign's sampling directory together with its immutable roots.
    """
    folder=Path(folder)
    manifest=read_json(folder/'manifest.json')
    receipt=folder.parents[2]/'sampling'/(manifest['root_id']+'.json')
    if not receipt.exists():return None
    value=read_json(receipt)
    if value['root_id']!=manifest['root_id'] or value['seed']!=manifest['seed'] or value['scene']!=manifest['scene_id']:
        raise ValueError('Sampling receipt does not identify this root')
    return value


def supervised_example(folder,query_index=0,verify=True):
    folder=Path(folder)
    if verify:validate_root(folder)
    inputs=online_input(folder,query_index)
    outcome=read_gzip(folder/'outcomes.json.gz')[query_index]
    nodes=read_gzip(folder/'future_observations.json.gz')
    labels=read_gzip(folder/'privileged_step_labels.json.gz')
    selected=[nodes[i] for i in outcome['node_ids']]
    # Observation/delta targets remain legal for their own future decision
    # times. Ground-truth proximity/contact supervision is separate again.
    future=[{'action':n['action'],'observation':n['observation'],'graph_delta':n['graph_delta']} for n in selected]
    graph=PartialGraph.restore(inputs['partial_graph'])
    for frame in future:graph.update(frame['observation'])
    graph.rebuild_global(outcome['successor']['robot_pose'][:3])
    successor_graph=graph.snapshot()
    check_legal(inputs);check_legal(future);check_legal(outcome['successor'])
    return {'reader_version':READER_VERSION,'inputs':inputs,
            'targets':{'outcome':outcome['summary'],'future_observations':future,
                       'successor_partial_graph':successor_graph,
                       'successor_belief':outcome['successor'],
                       'privileged_step_labels':[labels[i] for i in outcome['node_ids']]}}


def iter_roots(dataset,split='train'):
    if split not in ('train','dev','confirmation'):raise ValueError('Explicit valid scene split required')
    for path in sorted(Path(dataset).glob('roots/*/*/manifest.json')):
        if path.parent.name.startswith('.'):continue  # Unpublished or failed atomic staging.
        metadata=read_json(path)
        if metadata.get('sealed') and metadata['split']==split:yield path.parent
