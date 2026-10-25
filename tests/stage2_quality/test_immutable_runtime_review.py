"""Independent immutable-graph and actual frozen Stage1 arrival regressions."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MappingProxyType
from collections.abc import Mapping
import unittest
import numpy as np
import torch
from eventwm_graph.partial.map import PartialGraph
from eventwm_graph.stage2.api import apply_graph_delta as reference_apply
from eventwm_graph.stage2_quality.api import Stage2Predictor
from eventwm_graph.stage2_quality.common import BASE,ORIGINAL,SCHEMA
from eventwm_graph.stage2_quality.model import GraphDeltaNet
from eventwm_graph.stage2_quality.update import FastPartialGraph,apply_graph_delta


def immutable(value):
    if isinstance(value,Mapping):return MappingProxyType({key:immutable(v) for key,v in value.items()})
    if isinstance(value,(tuple,list,np.ndarray)):return tuple(immutable(v) for v in value)
    return value


def canonical_thaw(value):
    if isinstance(value,Mapping):return {key:canonical_thaw(v) for key,v in value.items()}
    if isinstance(value,(tuple,list)):return [canonical_thaw(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    return value


def graph():
    g=PartialGraph(1.,0.)
    updates=[dict(cell=[x,z],state=1,height=0.,timestamp=1.,source='measured') for x in range(-4,5) for z in range(-4,5)]
    g.update(dict(timestamp=1.,robot_pose=[0.,0.,0.,0.],static_updates=updates))
    out=g.snapshot();out['global_nodes']={'a':dict(id='a',cell=[-2,0],position=[-1.5,0.,.5],kind='junction',created_at=1.),'b':dict(id='b',cell=[2,0],position=[2.5,0.,.5],kind='junction',created_at=1.)}
    out['global_edges']=[dict(source='a',target='b',path=[[-1.5,0.,.5],[2.5,0.,.5]],length=4.,revision=1,certificate='known_free_cell_supercover')]
    return out


def real_arrival_fixture(checkpoint):
    # Read a single training manifest member, not the161MB manifest or protected data.
    manifest=ORIGINAL/'output/l2_two_stage_20261023/final_collection_20261024/manifest.json'
    with manifest.open() as stream:prefix=stream.read(65536)
    start=prefix.index('[',prefix.index('"rows"'))+1
    row,_=json.JSONDecoder().raw_decode(prefix[start:].lstrip())
    assert row['split']==row['source_split']=='train'
    folder=Path(row['folder'])
    for name in ('legal_root.json.gz','queries.json.gz','outcomes.json.gz','future_observations.json.gz'):
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==row['files'][name]
    from eventwm_graph.stage2_quality.integration import load_stage1,WorldModelSandbox
    model,source=load_stage1()
    from eventwm_graph.l2_two_stage.context import OnlineContext
    from eventwm_graph.l2_two_stage.data import read,selected_branch
    legal=read(folder/'legal_root.json.gz');_,_,branch=selected_branch(folder)
    assert branch
    previous=OnlineContext.from_raw(legal['history'],legal['partial_graph'],legal['history_actions']).with_association()
    obs=branch[0]['observation'];path=np.asarray([[node['observation']['timestamp'],*node['observation']['robot_pose']] for node in branch],np.float64)
    predictor=Stage2Predictor(checkpoint);sandbox=WorldModelSandbox(model,predictor)
    result=sandbox.predict_decision(previous,obs,executed_action=branch[0]['action'],trajectories=[path])
    actual=result['actual_context'];initial=canonical_thaw(actual.partial_graph)
    assert isinstance(actual.partial_graph,Mapping)
    assert not isinstance(actual.partial_graph,dict)
    assert actual.history[-1]['timestamp']==obs['timestamp']>previous.history[-1]['timestamp']
    assert len(actual._positive_history.timestamps)==len(previous._positive_history.timestamps)+1
    assert result['human_worlds'].positions.shape[:2]==(40,64)
    canonical_delta=predictor.predict(partial_graph=initial,trajectory=path);delta=result['graph_deltas'][0]
    for key in ('candidate_cells','probabilities','states','heights','unknown_cells'):np.testing.assert_array_equal(delta[key],canonical_delta[key])
    assert delta['static_updates']==canonical_delta['static_updates']
    assert result['hypothetical_graphs'][0]==reference_apply(initial,delta)
    assert canonical_thaw(actual.partial_graph)==initial
    return dict(root_id=row['root_id'],queries=1,stage1_calls=sandbox.last_profile['stage1_calls'],worlds=40,knots=64,source_stage1=str(source),known_cells=len(initial['cells']),candidate_cells=len(delta['states']),fixture_passed=True,timing_gate_evaluated=False)


class ImmutableRuntimeReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.checkpoint=Path(cls.temp.name)/'synthetic.pt';torch.manual_seed(419)
        model=GraphDeltaNet()
        torch.save(dict(schema=SCHEMA,mode='both',model_config=dict(width=48),model=model.state_dict(),config={}),cls.checkpoint)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_immutable_graph_restore_predict_update_matches_canonical_thaw(self):
        mutable=graph();sealed=immutable(mutable);before=canonical_thaw(sealed)
        self.assertEqual(FastPartialGraph.restore(sealed).snapshot(),PartialGraph.restore(before).snapshot())
        model=Stage2Predictor(self.checkpoint)
        for path in ([[1.,0.,0.,0.,0.]],[[1.,0.,0.,0.,0.],[2.,.7,0.,.2,.9]]):
            a=model.predict(partial_graph=sealed,trajectory=path);b=model.predict(partial_graph=before,trajectory=path)
            for key in ('candidate_cells','probabilities','states','heights','unknown_cells'):np.testing.assert_array_equal(a[key],b[key])
            self.assertEqual(a['static_updates'],b['static_updates'])
            self.assertEqual(apply_graph_delta(sealed,a),reference_apply(before,b))
            self.assertEqual(canonical_thaw(sealed),before)
        batch=model.predict_batch([dict(partial_graph=sealed,trajectory=[[1.,0.,0.,0.,0.],[2.,0.,0.,0.,.4]])]*2)
        np.testing.assert_array_equal(batch[0]['probabilities'],batch[1]['probabilities'])
        self.assertEqual(apply_graph_delta(sealed,batch[0]),reference_apply(before,batch[0]))

    def test_real_frozen_stage1_arrival_immutable_graph(self):
        original=ORIGINAL/'output/l2_two_stage_20261023/experiments/final_stage1_research_20261024/last.pt'
        if not original.exists():self.skipTest('Original frozen Stage1 checkpoint/data are not locally available')
        code='''import os,sys,types,runpy,json
from pathlib import Path
base=Path(sys.argv[1]);namespace=types.ModuleType('eventwm_graph');namespace.__path__=[str(base/'eventwm_graph')];sys.modules['eventwm_graph']=namespace
module=runpy.run_path(str(base/'tests/stage2_quality/test_immutable_runtime_review.py'),run_name='independent_fixture')
print(json.dumps(module['real_arrival_fixture'](Path(sys.argv[2]))))
'''
        env=dict(os.environ,EVENTWM_STAGE2_RESEARCH_BASE=str(BASE),EVENTWM_STAGE2_BASE=str(ORIGINAL),EVENTWM_TWO_STAGE_BASE=str(ORIGINAL),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',NUMBA_CACHE_DIR=str(BASE/'output/reviewer_b_numba_cache'))
        run=subprocess.run([sys.executable,'-c',code,str(BASE),str(self.checkpoint)],cwd='/tmp',env=env,text=True,capture_output=True)
        self.assertEqual(run.returncode,0,run.stdout+'\n'+run.stderr)
        result=json.loads(run.stdout.strip().splitlines()[-1]);self.assertTrue(result['fixture_passed']);self.assertEqual(result['stage1_calls'],1)
        out=BASE/'docs/l2/stage2_quality_runtime_20261024/reviews/reviewer_b/IMMUTABLE_REAL_ARRIVAL_FIXTURE.json'
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':unittest.main()
