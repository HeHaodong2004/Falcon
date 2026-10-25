"""Independent Reviewer B: runtime optimization semantic differentials.

No model source edits or private labels. Synthetic checkpoints use temporary
files and are unrelated to frozen research weights.
"""
import copy
import hashlib
import importlib
import importlib.util
import json
import sys
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from eventwm_graph.partial.map import PartialGraph
from eventwm_graph.stage2.api import apply_graph_delta as reference_apply,graph_identity
from eventwm_graph.stage2.common import SCHEMA as GRAPH_SCHEMA
from eventwm_graph.stage2.features import SceneFeatures
from eventwm_graph.stage2_quality.api import Stage2Predictor
from eventwm_graph.stage2_quality.common import SCHEMA
from eventwm_graph.stage2_quality.features import Features,PreparedGraph
from eventwm_graph.stage2_quality.model import GraphDeltaNet
from eventwm_graph.stage2_quality.update import FastPartialGraph,apply_graph_delta


def synthetic_graph(resolution=1.):
    g=PartialGraph(resolution,0.)
    updates=[dict(cell=[x,z],state=1 if (x,z)!=(2,2) else 2,height=0.,timestamp=1.,source='measured')
             for x in range(-4,5) for z in range(-4,5)]
    g.update(dict(timestamp=1.,robot_pose=[0.,0.,0.,0.],static_updates=updates))
    graph=g.snapshot()
    graph['global_nodes']={'a':dict(id='a',cell=[-2,0],position=[-1.5,0.,.5],kind='junction',created_at=1.),
                           'b':dict(id='b',cell=[2,0],position=[2.5,0.,.5],kind='junction',created_at=1.)}
    graph['global_edges']=[dict(source='a',target='b',path=[[-1.5,0.,.5],[2.5,0.,.5]],length=4.,revision=1,certificate='known_free_cell_supercover')]
    return graph


def make_delta(graph,updates=None,timestamp=2.):
    return dict(schema=GRAPH_SCHEMA,predicted=True,base_graph_identity=graph_identity(graph),timestamp=timestamp,
       robot_pose=[.8,0.,0.,.2],checkpoint_sha256='independent-synthetic',static_updates=updates or [])


class RuntimeSemanticReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.directory=Path(cls.temp.name)
        torch.manual_seed(901)
        model=GraphDeltaNet();cls.model=model.eval()
        for mode in ('baseline','trajectory','map','both'):
            path=cls.directory/(mode+'.pt')
            torch.save(dict(schema=SCHEMA,mode=mode,model_config=dict(width=48),model=model.state_dict(),config={}),path)
        cls.paths={mode:cls.directory/(mode+'.pt') for mode in ('baseline','trajectory','map','both')}

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def assert_prediction_equal(self,a,b):
        for name in ('candidate_cells','probabilities','states','heights','unknown_cells'):
            np.testing.assert_array_equal(a[name],b[name],err_msg=name)
        self.assertEqual(a['static_updates'],b['static_updates'])
        self.assertEqual(a['base_graph_identity'],b['base_graph_identity'])

    def test_compiled_supercover_randomized_and_boundary(self):
        rng=np.random.default_rng(902);cases=0
        for index in range(6):
            resolution=(.1,.25,1.)[index%3]
            reference=PartialGraph(resolution,0.)
            for x in range(-12,13):
                for z in range(-12,13):
                    reference.cells[(x,z)]=dict(state=1 if rng.random()>.035 else 2,height=0. if rng.random()>.05 else .3,source='test')
            reference.ego_support=np.array([.001,0.,.001]);fast=FastPartialGraph.restore(reference.snapshot())
            for j in range(500):
                a=rng.uniform(-8*resolution,8*resolution,3);b=rng.uniform(-8*resolution,8*resolution,3);a[1]=b[1]=0.
                if j%4==0:a=np.array([rng.integers(-8,9)*resolution,0.,rng.integers(-8,9)*resolution])
                if j%7==0:b=a.copy()
                if j%11==0:b[1]=.22 if j%22 else np.nextafter(.22,np.inf)
                self.assertEqual(reference.segment_free(a,b),fast.segment_free(a,b),(index,j,a,b));cases+=1
            for eps in (-2e-10,-1e-10,-.5e-10,0.,.5e-10,1e-10,2e-10):
                a=np.array([0.+eps,0.,0.]);b=np.array([resolution*3+eps,0.,resolution*3])
                self.assertEqual(reference.segment_free(a,b),fast.segment_free(a,b),(index,eps));cases+=1
        self.assertEqual(cases,3042)

    def test_ego_exception_and_empty_graph(self):
        for ego in (None,np.array([0.,0.,0.])):
            reference=PartialGraph(.25,0.);reference.ego_support=ego;fast=FastPartialGraph.restore(reference.snapshot())
            for length in (0.,.5e-8,1e-8,2e-8,.5e-7,1e-7,2e-7):
                for axis in range(3):
                    a=np.zeros(3);b=a.copy();b[axis]=length
                    self.assertEqual(reference.segment_free(a,b),fast.segment_free(a,b),(ego,length,axis))

    def test_entire_update_and_metadata_differential(self):
        rng=np.random.default_rng(903)
        for trial in range(80):
            graph=synthetic_graph(1. if trial%2 else .25);graph['height']=float(rng.normal(scale=.1))
            # Exercise outside old support, free and blocked revelations, heights,
            # elapsed visitation, edge preservation and invalidation semantics.
            keys=rng.choice(np.arange(5,13),size=4,replace=False)
            updates=[dict(cell=[int(x),int(z)],state=int(rng.integers(1,3)),height=float(rng.uniform(-.2,.2)),timestamp=2.,source='predicted_stage2_research') for x,z in zip(keys,keys[::-1])]
            if trial%4==0:updates=[]
            delta=make_delta(graph,updates);before=copy.deepcopy(graph);dbefore=copy.deepcopy(delta)
            a=reference_apply(graph,delta);b=apply_graph_delta(graph,delta)
            self.assertEqual(a,b,trial);self.assertEqual(graph,before);self.assertEqual(delta,dbefore)

    def test_graph_update_rejections_match(self):
        graph=synthetic_graph();row=dict(cell=[8,0],state=1,height=0.,timestamp=2.,source='predicted_stage2_research')
        valid=make_delta(graph,[row]);bad=[]
        for field,value in [('predicted',False),('schema','wrong'),('base_graph_identity','wrong'),('timestamp',.5)]:
            d=copy.deepcopy(valid);d[field]=value;bad.append(d)
        for field,value in [('cell',[0,0]),('state',0),('height',float('nan')),('timestamp',1.)]:
            d=copy.deepcopy(valid);d['static_updates'][0][field]=value;bad.append(d)
        duplicate=copy.deepcopy(valid);duplicate['static_updates']*=2;bad.append(duplicate)
        zero=copy.deepcopy(valid);zero['timestamp']=1.;zero['static_updates'][0]['timestamp']=1.;bad.append(zero)
        for delta in bad:
            for apply in (reference_apply,apply_graph_delta):
                with self.assertRaises(ValueError):apply(graph,delta)
        empty=make_delta(graph,timestamp=1.)
        self.assertEqual(reference_apply(graph,empty),apply_graph_delta(graph,empty))

    def test_cache_invalidates_every_static_input_dependency(self):
        graph=synthetic_graph();predictor=Stage2Predictor(self.paths['both']);prepared=predictor.prepare_graph(graph)
        self.assertIs(prepared,predictor.prepare_graph(copy.deepcopy(graph)))
        changes=[lambda g:g['cells'][0].update(cell=[-5,-4]),lambda g:g['cells'][0].update(state=2),lambda g:g['cells'][0].update(height=.1),
          lambda g:g.update(resolution=.5),lambda g:g.update(height=.2),lambda g:g.update(clearance=.3),lambda g:g.update(timestamp=1.5),lambda g:g.update(revision=2)]
        for change in changes:
            changed=copy.deepcopy(graph);change(changed)
            new=predictor.prepare_graph(changed);self.assertIsNot(new,prepared)
            np.testing.assert_array_equal(prepared.keys,np.array([c['cell'] for c in graph['cells']],np.int32))
        # Caller mutation after parsing cannot change owned array state.
        changed=copy.deepcopy(graph);owned=PreparedGraph(changed);before=owned.keys.copy();changed['cells'][0]['cell'][0]-=100
        np.testing.assert_array_equal(owned.keys,before)

    def test_metadata_cache_reuse_uses_current_caller_graph(self):
        graph=synthetic_graph();path=np.array([[1.,0.,0.,0.,0.],[2.,.4,0.,0.,.2]])
        predictor=Stage2Predictor(self.paths['both']);original=predictor.predict(partial_graph=graph,trajectory=path);prepared=predictor.prepare_graph(graph)
        changed=copy.deepcopy(graph);changed['visited'].append(dict(position=[.5,0.,0.],timestamp=1.))
        changed['global_edges']=[];changed['global_nodes']['a']['kind']='historical_decision';changed['cells'][0]['source']='metadata_only_test'
        changed_before=copy.deepcopy(changed);actual=predictor.predict(partial_graph=changed,trajectory=path)
        self.assertIs(prepared,predictor.prepare_graph(changed));self.assert_prediction_equal(original,actual)
        self.assertEqual(apply_graph_delta(changed,actual),reference_apply(changed,actual))
        self.assertEqual(changed,changed_before)
        out=apply_graph_delta(changed,actual)
        cells={tuple(c['cell']):c for c in out['cells']}
        self.assertEqual(cells[tuple(changed['cells'][0]['cell'])]['source'],'metadata_only_test')

    def test_shared_batch_query_isolation_and_zero_duration(self):
        graph=synthetic_graph();before=copy.deepcopy(graph)
        paths=[np.array([[1.,0.,0.,0.,0.]]),np.array([[1.,0.,0.,0.,0.],[2.,.3,0.,0.,.8]]),np.array([[1.,0.,0.,0.,0.],[1.2,0.,0.,0.,-1.],[1.5,0.,0.,0.,1.],[3.,.4,0.,.3,0.]])]
        for mode in ('baseline','trajectory','map','both'):
            p=Stage2Predictor(self.paths[mode]);items=[dict(partial_graph=graph,trajectory=path) for path in paths]
            batch=p.predict_batch(items)
            for item,out in zip(items,batch):self.assert_prediction_equal(p.predict(**item),out)
            self.assertEqual(batch[0]['static_updates'],[]);self.assertTrue(np.all(batch[0]['probabilities'][:,0]==1.))
            self.assertEqual(graph,before)
            self.assertEqual(p.predict_batch([]),[])

    def test_cached_schema_rejection(self):
        graph=synthetic_graph();p=Stage2Predictor(self.paths['both']);p.prepare_graph(graph)
        for value in ('wrong',None):
            bad=copy.deepcopy(graph);bad['schema']=value
            with self.assertRaises(ValueError):p.prepare_graph(bad)
            with self.assertRaises(ValueError):p.predict(partial_graph=bad,trajectory=[[1.,0.,0.,0.,0.]])

    def test_execution_optimizations_match_preserved_source_exactly(self):
        from eventwm_graph.stage2_quality.common import BASE
        snapshot=BASE/'output/stage2_quality_runtime_20261024/source_snapshots/fadf731998f8dd7318bcfcc56daad5c2c269fef8e14fcf82b5d8cf578060536f'
        if not snapshot.exists():self.skipTest('Preserved preoptimization research source is not locally available')
        identity=json.loads((snapshot/'manifest.json').read_text())
        for name,expected in identity['files'].items():self.assertEqual(hashlib.sha256((snapshot/name).read_bytes()).hexdigest(),expected)
        name='stage2_quality_independent_preoptimization'
        spec=importlib.util.spec_from_file_location(name,snapshot/'eventwm_graph/stage2_quality/__init__.py',submodule_search_locations=[str(snapshot/'eventwm_graph/stage2_quality')])
        package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
        old_api=importlib.import_module(name+'.api');old_features=importlib.import_module(name+'.features')
        graph=synthetic_graph(.25)
        paths=[np.array([[1.,0.,0.,0.,0.]]),np.array([[1.,0.,0.,0.,0.],[2.,.2,0.,0.,1.],[3.,.4,0.,.2,-1.]])]
        for mode in ('baseline','trajectory','map','both'):
            new=Stage2Predictor(self.paths[mode]);old=old_api.Stage2Predictor(self.paths[mode])
            for path in paths:
                a=Features(graph,path);b=old_features.Features(graph,path);ka=a.candidates(mode);kb=b.candidates(mode)
                np.testing.assert_array_equal(ka,kb)
                xa,ha=a.transform(ka,mode);xb,hb=b.transform(kb,mode)
                np.testing.assert_array_equal(xa,xb);np.testing.assert_array_equal(ha,hb)
                # Exercise the clipped-neighbor fallback as well as cached-domain fast indexing.
                border=np.array([a.lo,a.lo+a.shape-1,a.lo+1],np.int32)
                ya,za=a.transform(border,mode);yb,zb=b.transform(border,mode)
                np.testing.assert_array_equal(ya,yb);np.testing.assert_array_equal(za,zb)
                self.assert_prediction_equal(new.predict(partial_graph=graph,trajectory=path),old.predict(partial_graph=graph,trajectory=path))

    def test_candidate_domain_exact_on_float_boundaries(self):
        for resolution in (.02,.1,.25,1.):
            # Bounded selected feature keys plus exact candidate generation on
            # practical resolutions keeps this regression under modest memory.
            if resolution<.1:continue
            graph=synthetic_graph(resolution);path=np.array([[1.,0.,0.,0.,0.],[2.,1e-8,0.,1e-8,.5],[3.,.3,0.,-.2,-.5]])
            old=SceneFeatures(graph,path);reference=old.candidates()
            for mode in ('trajectory','both'):
                new=Features(graph,path);np.testing.assert_array_equal(reference,new.candidates(mode))


if __name__=='__main__':unittest.main()
