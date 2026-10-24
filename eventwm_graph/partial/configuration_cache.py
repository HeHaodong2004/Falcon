"""Exact private Hydra composition cache; each caller receives a fresh config.

Only the expensive composition is retained. DictConfig contents, readonly flags,
structured types and unresolved interpolation nodes are preserved by deepcopy.
Source-tree content, environment, working directory, arguments and process bind
each template; changes fail closed instead of silently returning stale state.
The installed Habitat/Falcon source is never modified.
"""
import copy
from contextlib import contextmanager
import inspect
import os
from pathlib import Path
import threading
import time

from ..config import digest, file_digest


CONFIGURATION_CACHE_VERSION = 'exact_hydra_composition_fresh_deepcopy_1'


def configuration_sources():
    import habitat.config.default as habitat_config
    import habitat_baselines.config.default as baseline_config
    roots = {Path(inspect.getfile(module)).resolve().parent
             for module in (habitat_config, baseline_config)}
    paths = sorted({p for root in roots for p in root.rglob('*')
                    if p.is_file() and p.suffix in ('.yaml', '.yml', '.py')})
    return {str(path):file_digest(path) for path in paths}


class ExactConfigurationCache:
    """A bounded per-worker template cache, installed only inside its scope."""
    def __init__(self):
        import habitat_baselines.config.default as config_module
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.module, self.original = config_module, config_module.get_config
        self.signature = inspect.signature(self.original)
        self.sources = configuration_sources()
        self.source_id = digest(self.sources)
        self.templates = {}
        self.active = False
        self.performance = dict(compositions=0, hits=0, composition_seconds=0.,
                                dependency_seconds=0., deepcopy_seconds=0.)

    def _owner(self):
        if os.getpid() != self.pid or threading.get_ident() != self.thread:
            raise RuntimeError('configuration_cache_owner_changed')

    def get_config(self, *args, **kwargs):
        self._owner()
        began = time.perf_counter()
        if configuration_sources() != self.sources:
            raise ValueError('configuration_cache_source_changed')
        self.performance['dependency_seconds'] += time.perf_counter()-began
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        from habitat.config.default import get_full_config_path
        selected_path = get_full_config_path(bound.arguments['config_path'],
                                             bound.arguments['configs_dir'])
        if str(Path(selected_path).resolve()) not in self.sources:
            raise ValueError('configuration_cache_external_config_unsupported')
        # Hydra paths/defaults may use environment or relative files. Both are
        # exact key dependencies, even where this workload currently does not.
        key = digest(dict(arguments=bound.arguments, environment=dict(os.environ),
                          cwd=str(Path.cwd()), source_id=self.source_id))
        if key not in self.templates:
            if len(self.templates) >= 8:
                self.templates.pop(next(iter(self.templates)))
            began = time.perf_counter()
            self.templates[key] = self.original(*args, **kwargs)
            self.performance['composition_seconds'] += time.perf_counter()-began
            self.performance['compositions'] += 1
        else:
            self.performance['hits'] += 1
        began = time.perf_counter()
        result = copy.deepcopy(self.templates[key])
        self.performance['deepcopy_seconds'] += time.perf_counter()-began
        return result

    @contextmanager
    def scope(self):
        self._owner()
        if self.active or self.module.get_config is not self.original:
            raise RuntimeError('overlapping_configuration_cache_scope')
        self.active = True
        self.module.get_config = self.get_config
        try:
            yield self
        finally:
            self.module.get_config = self.original
            self.active = False
