import hashlib,json,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OLD=Path('/home/haodong/workspace/projects/eventwm-falcon-stage2-segment-20261024')
PREP=OLD/'output/stage2_segment_20261024/prepared'
OUT=ROOT/'output/stage2_hf_20261025'
CKPT=ROOT/'docs/l2/stage2_segment_20261024/artifacts/segment_ray104.pt'
CKPT_SHA='1ba6844d8b7eb336c46b41125eefae16e413c4ba408da9b0a1dc47ebcef66bfd'
SEED=2526
THRESHOLDS=[.01,.02,.05,.10,.15,.20,.30,.40,.50,.60,.70,.80,.90]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def read(p):return json.loads(Path(p).read_text())
def stable(s):return hashlib.sha256(('stage2-hf2526:'+str(s)).encode()).hexdigest()
def source():
 files=[* (ROOT/'eventwm_graph/stage2_hf').glob('*.py'),ROOT/'eventwm_graph/stage2_segment/data.py',ROOT/'eventwm_graph/partial/map.py',ROOT/'scripts/stage2_segment_run.py']
 blobs=OUT/'sources/blobs';blobs.mkdir(parents=True,exist_ok=True);mapping={}
 for p in sorted(files):
  data=p.read_bytes();digest=hashlib.sha256(data).hexdigest();mapping[str(p.relative_to(ROOT))]=digest;target=blobs/digest
  if not target.exists():target.write_bytes(data);target.chmod(0o444)
 return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=mapping,source_blob_directory=str(blobs))
def freeze():
 from eventwm_graph.stage2.data import source_rows
 if (OUT/'protocol.json').exists():raise FileExistsError('protocol already frozen')
 train,tm=source_rows('train');dev,dm=source_rows('dev');old=read(OLD/'output/stage2_segment_20261024/protocol.json')
 opened=read(OLD/'output/stage2_diagnostic_20261025/final_confirmation_metadata/selected_manifest.json')['rows']
 scene_order=sorted({r['scene_id']for r in dev},key=stable);selection=set(scene_order[:70]);validation=set(scene_order[70:])
 pilot=[]
 for scene in scene_order[:16]:pilot.extend(sorted([r for r in dev if r['scene_id']==scene],key=lambda r:stable(r['root_id']))[:2])
 assert len(train)==105838 and len(dev)==2110
 assert not({r['scene_id']for r in train}&{r['scene_id']for r in dev})
 assert not({r['root_id']for r in opened}&{r['root_id']for r in train+dev})
 save(OUT/'protocol.json',dict(schema='stage2_high_fidelity_v1',seed=SEED,source=source(),train_manifest_sha256=sha(tm),dev_manifest_sha256=sha(dm),baseline_checkpoint_sha256=sha(CKPT),
  original_segmentation_protocol_sha256=sha(OLD/'output/stage2_segment_20261024/protocol.json'),
  train_roots=len(train),dev_roots=len(dev),pilot_fit=old['pilot_fit'],pilot_fitting_holdout=old['pilot_hold'],
  pilot_selection_roots=[r['root_id']for r in pilot],selection_scenes=sorted(selection),diagnostic_validation_scenes=sorted(validation),
  opened_protected_roots=[r['root_id']for r in opened],opened_protected_scenes=sorted({r['scene_id']for r in opened}),
  domain='Unchanged initiallyUNKNOWN union11m closed discs around all saved supplied poses; add every missedGTpositive as certainUNKNOWN miss. No hard geometry pruning.',
  labels='Exact original mixed windows and legalDAG replay; knowncorrections separate; no privileged geometry; no interpolated sensing labels.',
  pilot_models=['old architecture improved objective/sampling','factorized lightweight twohead','compact spatial twohead and optional hybrid'],
  pilot_budget='At most6initial lightweight configurations and2spatial variants; up to12epochs on2048Roots; one justified refinement perpath. Fullfit selected family once, up to6epochs with devselected early stopping; preserve all epochs/negative outcomes.',
  operating_point='Each model selects its own threshold on70development scenes, primary labelawareIoU; falseFREE and emptybehavior explicit constraints/tradeoffs, no test tuning.',
  selection='Prefer strictIoU improvement over oldthreshold.20 with pairedscene evidence, inspect proper scores/FREEerror/empty performance; speedPareto at matchedcomplete outputs. Never claim .45 reached below actual.',
  probability_metrics='Full candidate domain, multiclassBrier=sum3, logclip1e-9; Root/scene/window hierarchy unchanged; supporting AP/AUROC and calibration.',
  runtime='Identical oldsingleinputs and32oldmultiRootgroups; K1/4/6 distinctpath prefixes, mapcold/retained, normalGC, completefuturegraph primary; softonly separately; oldintegratedactualgate<=100ms retained.',
  uncertainty='2000pairedscene bootstrap seed2526; nestedRoot/windows kept together; protected confirmation only after model/threshold/source freeze, never opened222Root reuse.',
  no_stage1_training=True,no_layer3=True,no_collection=True,no_push=True))
 assert sha(CKPT)==CKPT_SHA
 print('Frozen protocol',OUT/'protocol.json',flush=True)
if __name__=='__main__':freeze()
