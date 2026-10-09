"""Incremental durable bookkeeping and isolated preparation for pipeline v1.

SQLite is authoritative while running. manifest.json is a compatible, periodic
export. A single coordinator owns the DB; native workers never open it.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from collector.core import read
from eventwm_validate_staged import durable_json


def encode(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


class StateStore:
    def __init__(self, output):
        self.output = Path(output)
        self.db = sqlite3.connect(str(self.output/'pipeline.sqlite3'))
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS roots (id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE NOT NULL, value TEXT NOT NULL)')
        self.pending = {}
        self.last_export = 0.
        self.metrics = Counter()
        self.contributions = {}
        self.progress = Counter()

    def touch(self, root):
        self.pending[root['root_id']] = root

    def flush(self, manifest, all_roots=False):
        began = time.perf_counter()
        roots = manifest['roots'] if all_roots else list(self.pending.values())
        meta = {k:v for k,v in manifest.items() if k not in ('roots', 'scenes')}
        # One FULL transaction reserves the proposal AND advances its sequence.
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', ('manifest', encode(meta)))
            if all_roots:
                self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', ('scenes', encode(manifest['scenes'])))
            if all_roots:
                self.contributions.clear()
                self.progress.clear()
            for root in roots:
                rid = root['root_id']
                contribution = Counter({'proposals':1, 'state_'+root['state']:1})
                if root['state']=='completed':
                    q=root['stats']['primary_queries']
                    contribution.update(primary_queries=q, completed_roots=1)
                    contribution['role_'+root['role']]=q
                    contribution['completed_root_bytes']=root['stats']['bytes']
                self.progress.subtract(self.contributions.get(rid, {}))
                self.progress.update(contribution)
                self.contributions[rid]=contribution
                self.db.execute('INSERT OR REPLACE INTO roots VALUES (?,?,?)',
                                (root['root_id'], root['_pipeline_ordinal'], encode(root)))
            self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', ('progress', encode(dict(self.progress))))
        self.pending.clear()
        self.metrics['transactions'] += 1
        self.metrics['root_records_written'] += len(roots)
        self.metrics['flush_seconds'] += time.perf_counter()-began

    def load(self):
        row = self.db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        value['scenes'] = json.loads(self.db.execute("SELECT value FROM meta WHERE key='scenes'").fetchone()[0])
        value['roots'] = [json.loads(r[0]) for r in self.db.execute('SELECT value FROM roots ORDER BY ordinal')]
        return value

    def export(self, manifest, force=False):
        if not force and time.monotonic()-self.last_export < 300:
            return
        import eventwm_mass_collection as mass
        began = time.perf_counter()
        from collector.storage import write_compact_json
        write_compact_json(self.output/'manifest.json', manifest)
        durable_json(self.output/'aggregate.json', mass.aggregate(manifest))
        self.last_export = time.monotonic()
        self.metrics['export_seconds'] += time.perf_counter()-began

    def close(self):
        self.db.close()


def immutable_json(path, value):
    """Retry a pre-dispatch write without silently replacing existing evidence."""
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError('Existing preparation artifact differs: '+str(path))
        return
    # Match collector.core.write byte-for-byte for scientific hash comparisons.
    from collector.core import write
    write(path, value)
    with path.open('rb') as stream:
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def prepare_episode(output, scene, spec, covered):
    """Runs in a spawn process; same episode_for implementation and RNG inputs."""
    import eventwm_mass_collection as mass
    started = time.perf_counter()
    output = Path(output)
    if hashlib.sha256(Path(scene['scene']['navmesh']).read_bytes()).hexdigest() != scene['navmesh_sha256']:
        raise RuntimeError('prepared_navmesh_hash_mismatch')
    try:
        sampler=spec.get('recipe',{}).get('episode_sampler')
        if sampler:
            from eventwm_geometry_v4 import episode as geometry_episode
            episode,geometry_audit=geometry_episode(scene,spec['seed'],spec['human_count'],covered,sampler)
        else:
            episode = mass.episode_for(scene, spec['seed'], spec['human_count'], covered)
    except RuntimeError as exc:
        # Only these two exact, intentional sampling failures are skippable.
        if str(exc) not in ('static_episode_unavailable: no connected 4-10m goal',
                            'static_episode_unavailable: no separated human waypoint',
                            'geometry_episode_unavailable: no feasible start in bounded search'):
            raise
        return dict(error=str(exc), seconds=time.perf_counter()-started)
    epfile = output/'catalogue/episodes'/f"{spec['root_id']}.json"
    # Preserve original episode JSON BYTES, not only decoded content.
    from collector.core import write
    if epfile.exists():
        if read(epfile) != episode:
            raise RuntimeError('Prepared episode differs on recovery')
    else:
        write(epfile, episode)
        with epfile.open('rb') as stream:
            os.fsync(stream.fileno())
        fd = os.open(epfile.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    episode_hash = hashlib.sha256(epfile.read_bytes()).hexdigest()
    case = dict(tag=spec['recipe']['case'], scene=spec['scene'], humans=spec['human_count'],
                seed=spec['seed'], episode=str(epfile.relative_to(output/'catalogue')),
                episode_sha256=episode_hash)
    immutable_json(output/'catalogue/cases'/f"{spec['root_id']}.json", case)
    immutable_json(output/'recipes'/f"{spec['root_id']}.json", spec['recipe'])
    return dict(error=None, start_region=mass.region(episode['start_position']),
                episode_sha256=episode_hash, seconds=time.perf_counter()-started)


def install_indexed_reads(runtime, controlled, current, output):
    """Adapt only catalogue lookups; native science calls and payloads unchanged."""
    output = Path(output)
    catalogue = (output/'catalogue').resolve()
    scenes = {s['name']:s for s in read(catalogue/'scenes.json')}

    def wrap(original):
        def indexed(path):
            p = Path(path)
            if p.name == 'replay.json' and p.parent.name == 'roots':
                recipe = original(path)
                case = read(catalogue/'cases'/f"{recipe['root_id']}.json")
                current['job'] = dict(root_id=recipe['root_id'], scene=case['scene'])
                return recipe
            if p.name in ('cases.json', 'scenes.json') and p.parent.resolve() == catalogue:
                job = current['job']
                if p.name == 'cases.json':
                    case = read(catalogue/'cases'/f"{job['root_id']}.json")
                    if case['scene'] != job['scene']:
                        raise RuntimeError('Indexed catalogue identity mismatch')
                    return [case]
                return [scenes[job['scene']]]
            return original(path)
        return indexed
    runtime.read = wrap(runtime.read)
    controlled.read = wrap(controlled.read)


def reserve_attempt(output, root, session_dir, persist):
    """Reservation reaches durable storage before an attempt directory exists.

    A crash before dispatch resumes the SAME reservation. Once dispatch is
    persisted, restart uses the existing interrupted-execution retry policy.
    """
    output = Path(output)
    if read(output/'recipes'/f"{root['root_id']}.json") != root['recipe']:
        raise RuntimeError('Frozen proposal recipe mismatch')
    epfile = output/'catalogue/episodes'/f"{root['root_id']}.json"
    if hashlib.sha256(epfile.read_bytes()).hexdigest() != root['episode_sha256']:
        raise RuntimeError('Frozen proposal episode mismatch')
    if not root['attempts'] or root['attempts'][-1]['state'] != 'reserved':
        number = len(root['attempts'])+1
        attempt = dict(number=number, path=f"roots/{root['root_id']}/attempt_{number:02d}",
                       state='reserved', started=time.time(),
                       log=str(Path(session_dir)/f"{root['root_id']}_{number}.jsonl"))
        root['attempts'].append(attempt)
        root.update(state='pending', pid=os.getpid(), gpu=None)
        persist(root)
    attempt = root['attempts'][-1]
    path = output/attempt['path']
    (path/'roots').mkdir(parents=True, exist_ok=True)
    immutable_json(path/'roots/replay.json', root['recipe'])
    return path


def read_status(output):
    path = (Path(output)/'pipeline.sqlite3').resolve()
    with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as db:
        data = {k:json.loads(v) for k,v in db.execute("SELECT key,value FROM meta WHERE key IN ('manifest','progress')")}
    m=data['manifest']
    return dict(state=m['state'], runner=m.get('runner'), progress=data.get('progress',{}),
                collection_seconds=m['collection_seconds'], target=m['target_primary_queries'],
                settings=m['pipeline_settings'], source_commit=m['frozen_commit'])
