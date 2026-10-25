"""Read-only verification of the compact diagnostic release and frozen sources.

No model imports, source-data payload reads, fitting or experiment reruns.
Original reports retain original output-relative links and are excluded from
reader-facing link validation deliberately.
"""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/l2/stage2_diagnostic_20261025'
OUT=ROOT/'output/stage2_diagnostic_20261025'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def main():
    manifest=read(DOC/'artifacts/artifact_manifest.json')
    for rel,value in manifest['files'].items():
        path=DOC/rel
        assert path.is_file(),rel
        assert path.stat().st_size==value['bytes'],rel
        assert sha(path)==value['sha256'],rel
    tracked=set(manifest['files'])
    actual={str(p.relative_to(DOC)) for p in DOC.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    assert tracked==actual,dict(unlisted=sorted(actual-tracked),missing=sorted(tracked-actual))
    links=[]
    for p in DOC.rglob('*.md'):
        if 'original_reports' in p.parts:continue
        for link in re.findall(r'\[[^\]]*\]\(([^\)\n]+)\)',p.read_text()):
            if link.startswith(('http:','https:','mailto:','#')):continue
            target=link.split('#',1)[0].strip('<>')
            if not target:continue
            resolved=(p.parent/target).resolve()
            assert resolved.exists(),(str(p.relative_to(DOC)),link)
            links.append((str(p.relative_to(DOC)),link))
    final=read(OUT/'final_confirmation/protocol.json')
    assert sha(ROOT/'output/stage2_segment_20261024/training/final/last.pt')==final['checkpoint_sha256']==manifest['checkpoint_sha256']
    for rel,expected in final['source'].items():assert sha(ROOT/rel)==expected,rel
    for rel,expected in final['frozen_development_sources'].items():assert sha(OUT/rel)==expected,rel
    assert sha(OUT/'final_confirmation_metadata/selected_manifest.json')==final['selected_manifest_sha256']
    summary=read(OUT/'final_confirmation/summary.json')
    proof=read(OUT/'final_confirmation/independent_review/proof.json')
    assert sha(OUT/'final_confirmation/summary.json')==proof['summary_sha256']
    assert summary['roots']==222 and summary['scenes']==111 and summary['segments']==4567
    selected=read(OUT/'final_confirmation_metadata/selected_manifest.json')['rows']
    expected_ids={r['root_id'] for r in selected}
    archived=[]
    with gzip.open(DOC/'artifacts/final_confirmation/roots.jsonl.gz','rt') as f:
        for line in f:
            record=json.loads(line);rid=record['root_id'];archived.append(rid)
            assert record==read(OUT/'final_confirmation/roots'/(rid+'.json')),rid
    assert len(archived)==222 and set(archived)==expected_ids
    old_refs={
        'research/stage2-segment-20261024':'28d4e924b2a850f25ff8fc2bc30029175cbc0301',
        'research/stage2-quality-runtime-20261024':'dfde432225071fe9a89fdf97db0396a2cc06a6ec'}
    for ref,expected in old_refs.items():
        assert subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip()==expected,ref
    assert not subprocess.check_output(['git','diff','82eb2dab','--','eventwm_graph'],cwd=ROOT,text=True).strip()
    print(json.dumps(dict(status='PASS',artifact_files=len(tracked),reader_facing_local_links=len(links),
        original_report_links='Excluded intentionally: byte-exact output-relative reports retained for audit hashes',
        frozen_final_source_files=len(final['source']),frozen_development_artifacts=3,
        final_derived_root_records=222,original_branch_refs_unchanged=True,production_source_unchanged=True,
        checkpoint_sha256=manifest['checkpoint_sha256'],neural_inference=False,raw_protected_payload_reads=False),indent=2))

if __name__=='__main__':main()
