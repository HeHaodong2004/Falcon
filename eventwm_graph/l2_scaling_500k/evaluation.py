"""Frozen evaluator orchestration; exact baseline cases and deployed inference."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .common import BASE,OUT,OLD,PRIOR,sha,dump

def setup(arm):
    root=OUT/('evaluation_runtime_'+arm);root.mkdir(exist_ok=False)
    for p in (BASE/'eventwm_graph').rglob('*.py'):
        if 'l2_scaling_500k' in p.parts:continue
        target=root/p.relative_to(BASE);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    archived=BASE/'docs/l2/intermittent_wm_20261022/artifacts/verified_execution/sources/eventwm_graph/l2_intermittent/evaluate.py'
    assert sha(archived)=='a278381a264b564eb51683469c89d307a41268805292ac856cec611434f3c184'
    evaluation_path=root/'eventwm_graph/l2_intermittent/evaluate.py'
    shutil.copyfile(archived,evaluation_path)
    if arm!='original':
        # Execute the requested checkpoint only. Unchanged comparator outputs
        # already exist in the committed release. No score, case or RNG changes.
        tree=ast.parse(archived.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        changes=[]
        for node in ast.walk(run):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='models' for t in node.targets):
                names=('D_StopMove','B_MemoryCV') if arm=='original_full' else ('D_StopMove',)
                keep=[i for i,k in enumerate(node.value.keys) if isinstance(k,ast.Constant) and k.value in names]
                assert len(keep)==len(names)
                node.value.keys=[node.value.keys[i] for i in keep];node.value.values=[node.value.values[i] for i in keep];changes.append('catalog_control_only')
        class Prune(ast.NodeTransformer):
            def visit_Assign(self,node):
                if any(isinstance(t,ast.Name) and t.id=='previous' for t in node.targets):changes.append('unused_previous_load');return None
                if any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='active' and isinstance(t.slice,ast.Constant) and t.slice.value=='A_CV' for t in node.targets):changes.append('unused_A_state');return None
                return self.generic_visit(node)
            def visit_Try(self,node):
                if node.body and isinstance(node.body[0],ast.Expr) and 'emit(score(previous,' in ast.unparse(node.body[0]):changes.append('unused_C_root');return None
                if len(node.body)>1 and 'emit(score(previous,' in ast.unparse(node.body[1]):changes.append('unused_C_mask');return None
                return self.generic_visit(node)
            def visit_If(self,node):
                if ast.unparse(node.test)=="'B_MemoryCV' in active":changes.append('unused_C_native');return None
                return self.generic_visit(node)
        Prune().visit(run);ast.fix_missing_locations(tree)
        assert sorted(changes)==sorted(['catalog_control_only','unused_previous_load','unused_A_state','unused_C_root','unused_C_mask','unused_C_native']),changes
        text=ast.unparse(tree)+'\n';compile(text,str(evaluation_path),'exec')
        original=ast.parse(archived.read_text())
        for name in ('score','energy','exact_ids','extend_ancestry','lineage_slot','mask_advance','paired_ci'):
            a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
            assert ast.dump(a)==ast.dump(b),name
        evaluation_path.write_text(text)
        dump(root/'evaluation_orchestration.json',dict(frozen_evaluator_sha256=sha(archived),changes=changes,
            scoring_ancestry_mask_and_statistics_ast_identical=True,semantics='Skip unrelated comparator arms; original_full retains D and fixed MemoryCV. Exact D-only equivalence is tested against the full frozen evaluator and committed original records.'))
        physical=root/'eventwm_graph/l2_intermittent/physical.py';tree=ast.parse(physical.read_text())
        physical_run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        for node in ast.walk(physical_run):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='models' for t in node.targets):
                i=next(i for i,k in enumerate(node.value.keys) if k.value=='D_StopMove')
                node.value.keys=[node.value.keys[i]];node.value.values=[node.value.values[i]]
        class PhysicalPrune(ast.NodeTransformer):
            def visit_Assign(self,node):
                if any(isinstance(t,ast.Name) and t.id=='previous' for t in node.targets):return None
                return self.generic_visit(node)
            def visit_For(self,node):
                if ast.unparse(node.iter)=="list(models) + ['C_Previous']":node.iter=ast.parse('list(models)',mode='eval').body
                return self.generic_visit(node)
        PhysicalPrune().visit(physical_run);ast.fix_missing_locations(tree)
        physical.write_text(ast.unparse(tree)+'\n')
    (root/'docs').symlink_to(BASE/'docs',target_is_directory=True)
    (root/'output').mkdir();legacy=root/'output/l2_graph_wm_120k_20261021';legacy.mkdir()
    for name in ('snapshot','dev.npy','motion.json','population_prior.json','training'):
        (legacy/name).symlink_to(OLD/name,target_is_directory=(OLD/name).is_dir())
    target=OUT/('evaluation_arm_'+arm);target.mkdir(exist_ok=False)
    for name in ('training','motion.json'):(target/name).symlink_to((PRIOR if arm.startswith('original') else OUT)/name,target_is_directory=name=='training')
    (root/'output/l2_intermittent_wm_20261022').symlink_to(target,target_is_directory=True)
    dump(root/'source_manifest.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*.py')})
    return root,target

def execute(arm,start,stop,limit):
    root=OUT/('evaluation_runtime_'+arm);target=OUT/('evaluation_arm_'+arm)
    if not root.exists():setup(arm)
    manifest=json.loads((root/'source_manifest.json').read_text())
    assert all(sha(root/p)==h for p,h in manifest.items())
    paths=[target/'training/checkpoint.pt',target/'training/epoch0.pt',target/'training/training.json',target/'motion.json',
        OLD/'dev.npy',OLD/'snapshot/manifest.json',OLD/'motion.json',OLD/'population_prior.json',OLD/'training/checkpoint.pt',
        BASE/'docs/l2/graph_wm_120k_20261021/artifacts/population_prior.json',
        BASE/'docs/l2/graph_wm_120k_20261021/artifacts/checkpoints/frozen_observation_fit.json']
    before={str(p):sha(p) for p in paths};started=time.perf_counter()
    env=os.environ.copy();env['PYTHONPATH']=str(root)
    args=[sys.executable,'-m','eventwm_graph.l2_intermittent.evaluate','--start',str(start),'--stop',str(stop)]
    if limit:args+=['--limit',str(limit)]
    with (target/('run_%04d_%04d_%d_%d.log'%(start,stop,limit,time.time_ns()))).open('x') as log:
        subprocess.run(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert all(sha(root/p)==h for p,h in manifest.items())
    assert before=={str(p):sha(p) for p in paths}
    name='evaluation_smoke' if limit else ('evaluation_shard_%04d_%04d'%(start,stop) if start or stop!=2110 else 'evaluation')
    dump(target/name/'execution_guard.json',dict(inputs_before=before,inputs_stable=True,sources_stable=True,
        source_manifest_sha256=sha(root/'source_manifest.json'),wrapper_sha256=sha(Path(__file__)),seconds=time.perf_counter()-started))

def physical(arm):
    root=OUT/('evaluation_runtime_'+arm);target=OUT/('evaluation_arm_'+arm)
    manifest=json.loads((root/'source_manifest.json').read_text())
    assert all(sha(root/p)==h for p,h in manifest.items())
    env=os.environ.copy();env['PYTHONPATH']=str(root)
    before={str(p):sha(p) for p in (target/'training/checkpoint.pt',target/'motion.json',OLD/'dev.npy',OLD/'snapshot/manifest.json')}
    with (target/'physical.log').open('x') as log:
        subprocess.run([sys.executable,'-m','eventwm_graph.l2_intermittent.physical'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert all(sha(root/p)==h for p,h in manifest.items()) and all(sha(Path(p))==h for p,h in before.items())
    dump(target/'physical_execution_guard.json',dict(sources_stable=True,inputs_stable=True,inputs_before=before))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['setup','run','physical']);p.add_argument('--arm',choices=['original','original_focused','original_full','new'],default='new');p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=2110);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    if a.command=='setup':setup(a.arm)
    elif a.command=='run':execute(a.arm,a.start,a.stop,a.limit)
    else:physical(a.arm)
