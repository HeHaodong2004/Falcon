"""Incremental cache of parameter-free, past-only history preprocessing.

Learned GRU states are never cached. Physical targets are never passed to the
normalizer. Payload identities pin raw compact-root bytes, normalizer source,
and the causal velocity-estimation window. Partial preparation survives pauses.
"""
from dataclasses import fields
import fcntl
import os
from pathlib import Path
import time
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.model import PreparedHistory
from eventbeliefwm.experiment import save_json


def prepare(model,roots,manifest,directory,original,guard=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'prepare.lock').open('a') as lock:
        while True:
            if guard:
                guard.check()
                if (guard.directory/'STOP_RESEARCH').exists():raise RuntimeError('Research stopped: preparation stop requested')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(2.)
        return _prepare(model,roots,manifest,directory,original,guard)


def _prepare(model,roots,manifest,directory,original,guard=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    config=dict(version=1,motion_window_s=model.core.config.motion_window_s,
        normalizer_sha256=sha256(Path(__file__).parents[1]/'eventbeliefwm/model.py'),
        adapter_sha256=sha256(Path(__file__).parent/'train.py'),cache_sha256=sha256(__file__))
    identity=directory/'identity.json'
    if identity.exists() and read(identity)!=config:raise ValueError('Prepared history cache identity changed')
    if not identity.exists():save_json(identity,config)
    entries={r['root_id']:r for r in read(manifest)['roots']}
    pending=[];pending_windows=0
    def write_pending():
        if not pending:return
        original(model,pending,guard)
        for root in pending:
            path=directory/(root['root_id']+'.pt')
            values=[{f.name:getattr(w['_prepared'],f.name) for f in fields(PreparedHistory)} for w in root['windows']]
            temp=path.with_suffix(f'.{os.getpid()}.tmp')
            torch.save(dict(source_sha256=entries[root['root_id']]['sha256'],prepared=values),temp)
            os.replace(temp,path)
            save_json(path.with_suffix('.json'),dict(sha256=sha256(path),windows=len(values)))
        pending.clear()
    for root in roots:
        if guard:
            guard.check()
            if (guard.directory/'STOP_RESEARCH').exists():raise RuntimeError('Research stopped: preparation stop requested')
        path=directory/(root['root_id']+'.pt');receipt=path.with_suffix('.json')
        if path.exists() and receipt.exists():
            if sha256(path)!=read(receipt)['sha256']:raise ValueError('Prepared legal feature bytes changed')
            cached=torch.load(path,map_location='cpu')
            if cached['source_sha256']!=entries[root['root_id']]['sha256']:raise ValueError('Prepared root provenance changed')
            if len(cached['prepared'])!=len(root['windows']):raise ValueError('Prepared window identity changed')
            for w,value in zip(root['windows'],cached['prepared']):w['_prepared']=PreparedHistory(**value)
        else:
            pending.append(root);pending_windows+=len(root['windows'])
            if pending_windows>=64:write_pending();pending_windows=0
    write_pending()
