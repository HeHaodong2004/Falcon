"""Metadata-only availability/identity check for final, inference-only confirmation.

Reads immutable SQLite sealed-job metadata and selected manifest.json files only.
Never opens legal observations, Query/outcome payloads, or privileged payloads.
"""
from collections import Counter,defaultdict
import hashlib,json,os,re,sqlite3,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/stage2_diagnostic_20261025/final_confirmation_metadata'
PROJECT=Path('/home/haodong/workspace/projects/eventwm-falcon-l2-two-stage-20261023/output/l2_two_stage_20261023')
FINAL=PROJECT/'final_collection_20261024/manifest.json';JOURNAL=FINAL.with_name('journal.sqlite3');REFERENCE=PROJECT/'snapshot/manifest.json'
OLD=Path('/home/haodong/workspace/projects/eventwm-falcon-l2-partial-20261021/output/l2_graph_wm_120k_20261021/snapshot/manifest.json')
MEMBERS=('legal_root.json.gz','queries.json.gz','outcomes.json.gz','future_observations.json.gz')
CORE=('map.py','observer.py','query.py','metrics.py','execution.py','eventwm_graph/stage1/runtime.py','scripts/collector/observation.py','falcon/additional_action.py','falcon/additional_metric.py')
SEED='stage2-final-confirm2525';FROZEN='82eb2dabb3401528c262d1de384b83b4098808aa'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(name,v):
 with (OUT/name).open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def rank(scene,rid):return hashlib.sha256((SEED+':'+scene+':'+rid).encode()).hexdigest()
def scenehash(scene):return int(hashlib.sha256(('l2-explore-20261021:'+scene).encode()).hexdigest()[:8],16)%100
def main():
 start=time.perf_counter();OUT.mkdir(exist_ok=False);fit=read(FINAL);old=read(OLD);reference=read(REFERENCE)
 fitscenes={r['scene_id']for r in fit['rows']};fitids={r['root_id']for r in fit['rows']};devscenes={r['scene_id']for r in old['rows']if r['split']=='dev'};devids={r['root_id']for r in old['rows']if r['split']=='dev'};opened={r['scene_id']for r in old['protected']if r['exclusion']=='previously_opened_fresh_scene'}
 source_splits=defaultdict(set)
 for r in old['rows']+old['protected']:source_splits[r['scene_id']].add(r['source_split'])
 assert all(len(s)==1 for s in source_splits.values())
 assert fitscenes.isdisjoint(devscenes) and fitids.isdisjoint(devids)
 identity=dict(frozen_development_commit=FROZEN,source_sha256=sha(__file__),final_fitting_manifest_sha256=sha(FINAL),old_boundary_manifest_sha256=sha(OLD),reference_collection_manifest_sha256=sha(REFERENCE),journal_sha256=sha(JOURNAL),journal_recorded_sha256=fit['journal_sha256'])
 assert identity['journal_sha256']==identity['journal_recorded_sha256']
 protocol=dict(schema='stage2_final_confirmation_metadata_only_v1',identity=identity,sqlite='mode=ro&immutable=1; PRAGMA query_only=ON; frozen closed-backup SHA verified',seed=SEED,selection='Per scene, take lexicographically smallest two SHA256(seed+":"+scene+":"+root_id) ranks among sealed journal Roots. Scene eligibility is original l2-explore-20261021 hash>=85, source split train, disjoint fitting/development/opened-fresh scenes. Selection never uses outcomes, payload, prediction, query count, or statistics.',source_split_unknown='For any scene absent from OLD metadata, deterministically select two candidates first and inspect ONLY their manifest.json metadata; exclude non-train/inconsistent scenes without replacement.',legal_payloads_opened=False,privileged_payloads_opened=False,training_authorized=False,production_changed=False,final_exposure='Later inference-only payload access requires a separately frozen final protocol after all development choices are frozen. Confirmation roots remain excluded from training/tuning regardless of result.',read_only_originals=True)
 dump('metadata_protocol.json',protocol)
 grouped=defaultdict(list);counts=Counter();allscenes=set();allrootids=set()
 c=sqlite3.connect('file:'+str(JOURNAL)+'?mode=ro&immutable=1',uri=True);c.execute('PRAGMA query_only=ON');assert c.execute('PRAGMA query_only').fetchone()[0]==1
 for seed,scene,text in c.execute("SELECT seed,scene,result FROM jobs WHERE state='sealed' ORDER BY seed"):
  result=json.loads(text);rid=result['root_id'];folder=result['folder']
  assert rid not in allrootids;allrootids.add(rid);allscenes.add(scene);counts['sealed_roots']+=1
  # Journal contains other statistics: they are neither retained nor consulted.
  known=next(iter(source_splits[scene]))if scene in source_splits else None
  reason=('source_non_train'if known is not None and known!='train'else'previously_opened_fresh_scene'if scene in opened else'fitting_scene'if scene in fitscenes else'development_scene'if scene in devscenes else'original_scene_hash_below85'if scenehash(scene)<85 else None)
  if reason:counts[reason]+=1;continue
  grouped[scene].append(dict(root_id=rid,scene_id=scene,seed=seed,folder=folder,selection_hash=rank(scene,rid),known_source_split=known));counts['candidate_roots']+=1
 c.close();assert counts['sealed_roots']==fit['source_roots']==186089
 chosen={scene:sorted(rs,key=lambda r:(r['selection_hash'],r['root_id']))[:2]for scene,rs in sorted(grouped.items())}
 dump('pre_manifest_selection.json',dict(scene_candidates={s:len(v)for s,v in sorted(grouped.items())},selected_before_source_manifest_reads=chosen,source_journal_sha256=identity['journal_sha256'],selection_metadata_only=True))
 selected=[];excluded=[];failure=[];selected_scenes=[]
 for scene,rs in chosen.items():
  checked=[];scene_reasons=[]
  for r in rs:
   folder=Path(r['folder']);mp=folder/'manifest.json'
   try:
    m=read(mp)
    assert m['sealed'] is True and m['schema']=='observation_limited_query_conditioned_root_1'
    assert m['root_id']==m['future_id']==r['root_id'] and m['scene_id']==scene and m['independent_futures']==1
    if m['split']!='train':scene_reasons.append('manifest_source_non_train:'+m['split']);continue
    assert r['known_source_split']in(None,'train')
    assert m['queries']==len(m['query_ids'])
    assert m['collection_config']==reference['collection_config']
    assert all(m['version'][k]==reference['core_reference'][k]for k in CORE)
    files={name:m['files'][name]for name in MEMBERS}
    assert all(isinstance(v,str)and re.fullmatch('[0-9a-f]{64}',v)for v in files.values())
    present={name:(folder/name).is_file()and not(folder/name).is_symlink()for name in MEMBERS}
    assert all(present.values())
    assert r['root_id']not in fitids|devids
    checked.append(dict(root_id=r['root_id'],scene_id=scene,seed=r['seed'],folder=str(folder),split='protected_research_holdout',source_split='train',analysis_use='final_confirmation_inference_only',selection_hash=r['selection_hash'],scene_hash_percent=scenehash(scene),sealed=True,schema=m['schema'],independent_futures=1,queries=m['queries'],files={**files,'manifest.json':sha(mp)},legal_members_exist=present,version_core={k:m['version'][k]for k in CORE},collection_config_sha256=hashlib.sha256(json.dumps(m['collection_config'],sort_keys=True).encode()).hexdigest(),manifest_metadata_only=True,payload_bytes_read=0))
   except Exception as e:failure.append(dict(root_id=r['root_id'],scene_id=scene,error=repr(e)));scene_reasons.append('manifest_validation_failure')
  if scene_reasons:
   excluded.append(dict(scene_id=scene,reasons=scene_reasons,selected_roots=[r['root_id']for r in rs],no_resampling=True));continue
  selected.extend(checked);selected_scenes.append(scene)
 assert {r['scene_id']for r in selected}.isdisjoint(fitscenes|devscenes|opened)
 assert all(scenehash(r['scene_id'])>=85 for r in selected)
 assert identity['journal_sha256']==sha(JOURNAL) and identity['final_fitting_manifest_sha256']==sha(FINAL) and identity['old_boundary_manifest_sha256']==sha(OLD)
 result=dict(schema='stage2_final_confirmation_metadata_availability_v1',status='AVAILABLE'if selected else'UNAVAILABLE',identity=identity,sealed_roots=counts['sealed_roots'],sealed_scenes=len(allscenes),fitting_roots=len(fit['rows']),fitting_scenes=len(fitscenes),development_scenes=len(devscenes),previously_opened_fresh_scenes=len(opened),metadata_source_scenes_in_old=len(source_splits),candidate_scenes_before_manifest_validation=len(grouped),candidate_roots_before_manifest_validation=counts['candidate_roots'],eligible_confirmation_scenes=len(selected_scenes),selected_roots=len(selected),selected_query_count_metadata_only=sum(r['queries']for r in selected),roots_per_scene=dict(Counter(Counter(r['scene_id']for r in selected).values())),exclusion_counts=dict(counts),excluded_selected_scenes=excluded,failures=failure,selected_scene_ids=selected_scenes,protected_boundary_preserved=True,protected_payloads_opened=False,privileged_payloads_opened=False,originals_written=False,separate_unseal='No filesystem unseal or collector control needed: original sealed manifest metadata and member existence are readable. Existing source_rows intentionally rejects protected payloads, so any later authorized inference needs a separate frozen read-only final-confirmation adapter; do not change source_rows or fitting boundaries.',sqlite_initial_attempt_note='Plain mode=ro on frozen WAL-mode DB raised attempt-to-write-readonly on metadata access. Retried mode=ro&immutable=1 with query_only after verifying frozen journal SHA; no original file modified.',elapsed_seconds=time.perf_counter()-start)
 dump('selected_manifest.json',dict(schema='protected_final_confirmation_metadata_selection_v1',protocol_sha256=sha(OUT/'metadata_protocol.json'),preselection_sha256=sha(OUT/'pre_manifest_selection.json'),rows=selected,protected_boundary_preserved=True,read_only_inference_only=True,payloads_opened=False))
 dump('availability.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
