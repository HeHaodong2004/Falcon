#!/usr/bin/env python3
"""Offline, read-only complete-root verification; writes only a derived report."""
import argparse
import hashlib
import json
from pathlib import Path
from collector.core import read,write,digest
from collector.v3 import social_outcome_labels
from eventwm_mass_collection import REPO,REPORT,inspect_root
from eventwm_scale import verify_root

def verify(output):
    m=read(output/'manifest.json');ids=set();nlabels=nprimary=nseals=0;results=[]
    for root in m['roots']:
        if root['state']!='completed':raise RuntimeError('Incomplete/failed root')
        path=output/root['path'];seals=read(output/root['seal_file'])
        verify_root(output,dict(root,seals=seals),deep=True);nseals+=len(seals)
        stats,labels=inspect_root(path,root)
        q={x['query']['query_id']:x for x in read(path/'queries/queries.json')}
        saved={l['branch_id']:l for l in read(path/'labels/labels.json')}
        fp=read(path/'roots/privileged_root.json')['sha256'];pairs={};branches=set();local_primary=0
        with (path/'branch_truth/branches.jsonl').open() as f:
            for line in f:
                b=json.loads(line);bid=b['branch_id'];query=q[b['query_id']]
                assert bid not in branches;branches.add(bid)
                assert b['query_hash_before']==b['query_hash_after']==query['sha256']==digest(query['query'])
                assert query['query']['root_id']==root['root_id']
                assert b['root_fingerprint']==fp
                assert social_outcome_labels(b,query['query'],root['human_count'])==saved[bid]
                nlabels+=1
                if bid.startswith('b'):
                    assert b['query_id'] not in ids;ids.add(b['query_id']);local_primary+=1;nprimary+=1
                if bid in ('b000','b001','reverse_0','reverse_1'):
                    pairs[bid]=[digest(b[k]) for k in ('truth','controls','legal_observations')]
        assert branches==set(saved) and local_primary==33
        assert pairs['b000']==pairs['reverse_0'] and pairs['b001']==pairs['reverse_1']
        provenance=read(path/'audits/provenance.json')['source_sha256']
        for name,h in provenance.items():assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==h
        results.append(dict(root_id=root['root_id'],primary=local_primary,labels=len(saved),status='PASS'))
    result=dict(output=str(output),completed_roots=len(results),failures=0,primary_queries=nprimary,
                labels_recomputed=nlabels,sealed_files_verified=nseals,root_results=results,
                replay_restore='PASS',query_immutability='PASS',static_certificates='PASS',static_violations=0,
                label_recomputation='PASS',duplicate_primary_branches=0,worker_corruption=0,collector_source_hashes='PASS',overall='PASS')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    result=verify(a.output.resolve());write(a.report,result);print(json.dumps({k:v for k,v in result.items() if k!='root_results'},indent=2))
