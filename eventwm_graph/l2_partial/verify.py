"""Read-only checksum and independent-split verification of the frozen snapshot."""
import argparse,collections,hashlib,json,pathlib,time

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def run(snapshot,output,full=False):
    root=pathlib.Path(snapshot);m=json.loads((root/'manifest.json').read_text());scenes=collections.defaultdict(set);roots=set();queries=set();files=0;started=time.perf_counter()
    for r in m['rows']:
        assert r['root_id'] not in roots;roots.add(r['root_id']);scenes[r['split']].add(r['scene_id'])
        for q in r['query_ids']:
            # Query identity is scoped to the root even if a compiler reuses IDs.
            key=(r['root_id'],q);assert key not in queries;queries.add(key)
        if full:
            for name,want in r['files'].items():
                p=root/'roots'/r['root_id']/name
                assert sha(p)==want, str(p)
                assert p.stat().st_mode&0o222==0, str(p)
                files+=1
    for a in scenes:
        for b in scenes:
            if a!=b:assert not scenes[a]&scenes[b]
    assert sha(root/'journal.sqlite3')==m['journal_sha256']
    result={'roots':len(roots),'root_queries':len(queries),'split_scenes':{k:len(v) for k,v in scenes.items()},'scene_overlap':0,'files_verified':files,'full':full,'seconds':time.perf_counter()-started,'manifest_sha256':sha(root/'manifest.json'),'source_revision':m['source_revision'],'status':'PASS'}
    pathlib.Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('snapshot');p.add_argument('output');p.add_argument('--full',action='store_true');run(**vars(p.parse_args()))
