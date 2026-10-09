"""Resolve live process identity; scientific source is never selected by name."""
from pathlib import Path
from control import C, SCIENCE, metadata, identity, git, read, require, stop

def active_repo():
    m, _, _ = metadata()
    require(m is not None and m['frozen_commit'] == C['scientific_commit'], 'Checkpoint source differs')
    live = identity(m)
    repo = Path(live['worktree']) if live else SCIENCE
    require(git(repo, 'rev-parse', 'HEAD') == m['frozen_commit'], 'Source pin differs')
    require(read(repo/'configs/eventwm_pipeline_release.json')['implementation_hashes'] == m['source_hashes'], 'Source manifest differs')
    return repo

def drain():
    stop()
