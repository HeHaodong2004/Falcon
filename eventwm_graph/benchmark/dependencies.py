"""Append-only revocation evidence and indexed dataset quarantine.

Samples are identified through their exact map/edge dependencies; no unrelated
shard is deleted. Snapshot builders must consult this index before sealing.
"""
import json
import sqlite3
from pathlib import Path
import time


class DependencyIndex:
    def __init__(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path,timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS samples(sample_id TEXT PRIMARY KEY,map_id TEXT NOT NULL,shard TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS dependencies(sample_id TEXT NOT NULL,map_id TEXT NOT NULL,edge_id TEXT NOT NULL,
            PRIMARY KEY(sample_id,map_id,edge_id));
          CREATE INDEX IF NOT EXISTS edge_samples ON dependencies(map_id,edge_id);
          CREATE TABLE IF NOT EXISTS revocations(map_id TEXT NOT NULL,edge_id TEXT NOT NULL,evidence TEXT NOT NULL,created REAL NOT NULL,
            PRIMARY KEY(map_id,edge_id));
          CREATE TABLE IF NOT EXISTS root_revocations(root_id TEXT PRIMARY KEY,evidence TEXT NOT NULL,created REAL NOT NULL);
        ''')

    def register(self,sample_id,map_id,edge_ids,shard):
        with self.db:
            previous=self.db.execute('SELECT map_id,shard FROM samples WHERE sample_id=?',(sample_id,)).fetchone()
            if previous and previous!=(map_id,str(shard)):raise ValueError('Sample identity reused')
            if previous:
                old={e[0] for e in self.db.execute('SELECT edge_id FROM dependencies WHERE sample_id=?',(sample_id,))}
                if old!=set(edge_ids):raise ValueError('Sample edge dependencies changed')
            self.db.execute('INSERT OR IGNORE INTO samples VALUES(?,?,?)',(sample_id,map_id,str(shard)))
            self.db.executemany('INSERT OR IGNORE INTO dependencies VALUES(?,?,?)',[(sample_id,map_id,e) for e in edge_ids])

    def revoke(self,map_id,edge_id,evidence):
        if not evidence:raise ValueError('Revocation requires evidence')
        with self.db:self.db.execute('INSERT OR IGNORE INTO revocations VALUES(?,?,?,?)',
            (map_id,edge_id,json.dumps(evidence,sort_keys=True),time.time()))
        return self.quarantined()

    def quarantined(self):
        return [{'sample_id':s,'map_id':m,'edge_id':e,'shard':p} for s,m,e,p in self.db.execute('''
          SELECT DISTINCT s.sample_id,s.map_id,r.edge_id,s.shard FROM samples s
          JOIN revocations r ON s.map_id=r.map_id
          LEFT JOIN dependencies d ON d.sample_id=s.sample_id AND d.edge_id=r.edge_id
          WHERE r.edge_id='*' OR d.edge_id IS NOT NULL
          UNION
          SELECT s.sample_id,s.map_id,'@root',s.shard FROM samples s
          JOIN root_revocations r ON substr(s.sample_id,1,length(r.root_id)+1)=r.root_id||':'
          ORDER BY 1,3''')]

    def revoke_root(self,root_id,evidence):
        """Quarantine every branch sharing an invalid root; preserve other data."""
        if not evidence:raise ValueError('Revocation requires evidence')
        with self.db:self.db.execute('INSERT OR IGNORE INTO root_revocations VALUES(?,?,?)',
            (root_id,json.dumps(evidence,sort_keys=True),time.time()))
        return self.quarantined()

    def assert_allowed(self,map_id,edge_ids):
        blocked={r[0] for r in self.db.execute('SELECT edge_id FROM revocations WHERE map_id=?',(map_id,))}
        if '*' in blocked or blocked.intersection(edge_ids):raise ValueError('Route contains revoked connections')

    def close(self):self.db.close()
