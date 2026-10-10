"""Explicit, backed-up source transition; preserves all completed root records.

Default is read-only. --apply changes source provenance only, while stopped.
The unchanged runner subsequently uses its existing interrupted-attempt retry law.
"""
import argparse, copy, fcntl, hashlib, json, shutil, sqlite3, time
import sys
from collections import Counter
from pathlib import Path
from control import C, DATA, SCIENCE, RELEASE_REPO, metadata, reject_active, source_check, require, read, git, canonical_digest, durable

OLD='58b4b0a17c90c5c7de420834b8ef62c44194eee7'
CHANGED={'scripts/collector/static_habitat.py','scripts/collector/static_feasibility.py'}

def transition(m,roots,new_release,new_commit,output):
    require(m['frozen_commit']==OLD,'This transition accepts only the diagnosed previous epoch')
    require(set(m['source_hashes'])==set(new_release['implementation_hashes']),'Implementation inventory changed')
    changed={p for p,h in m['source_hashes'].items() if new_release['implementation_hashes'][p]!=h}
    require(changed==CHANGED,'Unexpected scientific source changes: '+repr(changed))
    require(m['pipeline_settings']==C['pipeline_settings'],'Saved collection settings differ')
    require(m['master_seed']==new_release['configuration']['master_seed'],'Master seed differs')
    require(m['protected_scientific_hashes']==new_release['protected_scientific_hashes'],'Protected reference differs')
    new=copy.deepcopy(m); updates=[]
    for r in roots:
        if r['state']=='completed':continue
        retry=r['state']=='failed' and len(r['attempts'])<m['max_attempts'] and r['attempts'][-1].get('reason') in ('interrupted_before_commit','root_timeout','worker_timeout')
        if r['state'] not in ('pending','preparing','running') and not retry:continue
        path=r['attempts'][-1].get('path') if r['attempts'] else None
        validated=path is not None and (output/path/'VALIDATED.json').exists()
        if validated:continue  # Publish already validated old-epoch work with old provenance.
        after=copy.deepcopy(r)
        after.setdefault('source_epoch_history',[]).append(dict(commit=r['source_commit'],attempt_numbers=[a['number'] for a in r['attempts']]))
        after['source_commit']=new_commit
        updates.append((r,after))
    new['production_source_epochs']=copy.deepcopy(m.get('production_source_epochs',[]))
    new['production_source_epochs'].append(dict(commit=new_commit,source_hashes=new_release['implementation_hashes'],
        pipeline_release=canonical_digest(new_release),transition='diagnosed native float static certification repair; completed roots immutable',
        first_sequence=m['next_sequence'],resumed_root_ids=[r['root_id'] for r,_ in updates],time=time.time()))
    new.update(frozen_commit=new_commit,source_hashes=new_release['implementation_hashes'],pipeline_release=canonical_digest(new_release),state='stopped_recovery_ready')
    new['checker']=new_release.get('recovery',{}).get('checker_version',m.get('checker'))
    return new,updates

def completed_digest(db):
    h=hashlib.sha256()
    for rid,value in db.execute('SELECT id,value FROM roots ORDER BY ordinal'):
        if json.loads(value)['state']=='completed':h.update(rid.encode()+b'\0'+value.encode()+b'\0')
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args()
    reject_active()
    require(git(SCIENCE,'rev-parse','HEAD')==C['scientific_commit'],'New source pin differs')
    release=source_check();m,progress,roots=metadata(roots=True)
    old_release=read(Path(C['parent_scientific_checkout'])/'configs/eventwm_pipeline_release.json')
    require(m['source_hashes']==old_release['implementation_hashes'] and m['pipeline_release']==canonical_digest(old_release),'Old checkpoint release differs')
    new,updates=transition(m,roots,release,C['scientific_commit'],DATA)
    summary=dict(old_commit=OLD,new_commit=C['scientific_commit'],retained_new=progress['primary_queries'],
                 completed_roots=sum(r['state']=='completed' for r in roots),resumed_roots=len(updates),
                 validated_old_roots=sum(r['state']=='running' and (DATA/r['attempts'][-1]['path']/'VALIDATED.json').exists() for r in roots),
                 failed_root_retried=any(r['root_id']=='v3m_e7c6987aa64211d2f63c' for r,_ in updates),settings=m['pipeline_settings'])
    require(summary['failed_root_retried'],'Failing root must take the existing retry path')
    if not a.apply:print(json.dumps(dict(summary,dry_run=True),indent=2));return
    backup=DATA/'recovery_epochs'/C['scientific_commit']
    require(not backup.exists(),'Recovery backup already exists; inspect it instead of repeating the transition')
    with (DATA/'.lock').open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);reject_active_except_lock(m)
        backup.mkdir(parents=True)
        db=sqlite3.connect(str(DATA/'pipeline.sqlite3'));dest=sqlite3.connect(str(backup/'pipeline_before.sqlite3'))
        try:
            db.backup(dest);require(dest.execute('PRAGMA quick_check').fetchone()[0]=='ok','Backup check failed');dest.close()
            shutil.copy2(DATA/'manifest.json',backup/'manifest_before.json')
            # Finish old-epoch validated work using its ORIGINAL checker and
            # publication implementation. The new checker never reinterprets
            # a v2.2 certificate. No native job or completed root is replayed.
            sys.path[:0]=[str(Path(C['parent_scientific_checkout'])/'scripts'),str(Path(C['parent_scientific_checkout']))]
            from eventwm_pipeline_runner import publish_one
            from eventwm_pipeline_io import StateStore
            import eventwm_mass_collection as mass
            for r in roots:
                if r['state']=='running' and (DATA/r['attempts'][-1]['path']/'VALIDATED.json').exists():
                    mass.inspect_root(DATA/r['attempts'][-1]['path'],r)
            store=StateStore(DATA)
            try:
                saved=store.load()
                # Prime incremental accounting without rewriting historical rows.
                store.progress=Counter(progress)
                for r in saved['roots']:
                    contribution=Counter({'proposals':1,'state_'+r['state']:1})
                    if r['state']=='completed':
                        contribution.update(primary_queries=r['stats']['primary_queries'],completed_roots=1)
                        contribution['role_'+r['role']]=r['stats']['primary_queries']
                        contribution['completed_root_bytes']=r['stats']['bytes']
                    store.contributions[r['root_id']]=contribution
                for r in saved['roots']:
                    if r['state']=='running' and (DATA/r['attempts'][-1]['path']/'VALIDATED.json').exists():
                        r.update(publish_one(DATA,r));store.touch(r);store.flush(saved)
                store.export(saved,force=True)
            finally:store.close()
            m,progress,roots=metadata(roots=True)
            new,updates=transition(m,roots,release,C['scientific_commit'],DATA)
            summary['retained_new_after_old_publication']=progress['primary_queries']
            durable(backup/'transition.json',dict(summary,before_manifest=m,root_updates=[dict(before=b,after=x) for b,x in updates]))
            before=completed_digest(db)
            db.execute('PRAGMA synchronous=FULL')
            with db:
                require(json.loads(db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()[0])==m,'Checkpoint changed since inspection')
                db.execute("UPDATE meta SET value=? WHERE key='manifest'",(json.dumps(new,separators=(',',':')),))
                for b,x in updates:
                    actual=json.loads(db.execute('SELECT value FROM roots WHERE id=?',(b['root_id'],)).fetchone()[0])
                    require(actual==b,'Root changed during transition')
                    db.execute('UPDATE roots SET value=? WHERE id=?',(json.dumps(x,separators=(',',':')),b['root_id']))
                require(completed_digest(db)==before,'Completed roots changed')
            exported=copy.deepcopy(new)
            exported['scenes']=json.loads(db.execute("SELECT value FROM meta WHERE key='scenes'").fetchone()[0])
            exported['roots']=[json.loads(v) for v, in db.execute('SELECT value FROM roots ORDER BY ordinal')]
            durable(DATA/'manifest.json',exported)
            durable(backup/'APPLIED.json',dict(summary,completed_records_sha256=before,backup=str(backup),applied_at=time.time()))
        finally:db.close()
    print(json.dumps(dict(summary,applied=True,backup=str(backup)),indent=2))

def reject_active_except_lock(m):
    from control import identity,session_exists
    require(not identity(m),'Collector became active')
    require(not session_exists(C['session']),'Collector session became active')

if __name__=='__main__':main()
