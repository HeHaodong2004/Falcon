"""Opt-in scene lifetime reuse with fresh native episode/task/controller state.

The accepted :class:`NativeEpisode` constructor remains authoritative. A lease
reuses only its private simulator's static scene: Habitat's normal new-episode
reconfigure recreates every articulated body and reloads the certified NavMesh.
Each lease constructs a new Env, task, measures, actions and humanoid controller.
This deliberately avoids relying on incomplete controller ``reset`` methods.

The temporary Env simulator factory replacement is process local, guarded, and
restricted to the owner's thread. This pool is for one sequential worker; it is
not a global or thread-safe simulator pool. No collection is started here.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import threading

from ..config import file_digest
from ..stage1.runtime import NativeEpisode


PERSISTENT_RUNTIME_VERSION = 'fresh_native_episode_retained_static_scene_1'


def _configuration(config):
    from omegaconf import OmegaConf
    result = OmegaConf.to_container(config, resolve=True)
    # NativeEpisode reseeds the retained simulator before its ordinary reset.
    result.pop('seed', None)
    return result


class _Lease:
    def __init__(self, pool, hmap, episode, seed, visual):
        self.pool, self.arguments = pool, (hmap, episode, seed, visual)

    def __enter__(self):
        return self.pool._acquire(*self.arguments)

    def __exit__(self, error_type, error, traceback):
        self.pool._release(error_type is not None)


class SceneEpisodePool:
    """One retained scene per worker; compatible with NativeEpisode factories.

    Use ``with pool(hmap, episode, seed) as runtime`` and finally ``pool.close``.
    Exceptions discard the simulator, preserving a fresh retry. Scene/manifest/
    NavMesh/config mismatches also close it before creating the next runtime.
    Runtime leases never outlive or overlap one another.
    """
    def __init__(self):
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.runtime = None
        self.identity = None
        self.active = False
        self.performance = {'fresh_scene_initializations':0, 'retained_scene_initializations':0}

    def _owner(self):
        if os.getpid() != self.pid or threading.get_ident() != self.thread:
            raise RuntimeError('persistent_runtime_owner_changed')

    @staticmethod
    def _identity(hmap, episode, visual):
        return (str(Path(episode.scene_id).resolve()),
                file_digest(hmap.manifest_path),
                file_digest(episode.info['graph_navmesh']),
                hmap.contract['config_name'], bool(visual))

    def __call__(self, hmap, episode, seed, visual=False):
        self._owner()
        if visual:
            raise ValueError('persistent_runtime_is_nonvisual_only')
        return _Lease(self, hmap, episode, int(seed), visual)

    def _acquire(self, hmap, episode, seed, visual):
        self._owner()
        if self.active:
            raise RuntimeError('overlapping_persistent_runtime_leases')
        identity = self._identity(hmap, episode, visual)
        if self.runtime is not None and identity != self.identity:
            self.close()
        try:
            if self.runtime is None:
                runtime = NativeEpisode(hmap, episode, seed, visual=visual)
                self.performance['fresh_scene_initializations'] += 1
            else:
                sim = self.runtime.sim
                with self._retained_simulator(sim, episode):
                    runtime = NativeEpisode(hmap, episode, seed, visual=visual)
                if runtime.sim is not sim:
                    raise RuntimeError('persistent_runtime_did_not_retain_scene')
                self.performance['retained_scene_initializations'] += 1
            self.runtime, self.identity, self.active = runtime, identity, True
            return runtime
        except BaseException:
            self.close()
            raise

    @contextmanager
    def _retained_simulator(self, sim, episode):
        import habitat.core.env as env_module
        original = env_module.make_sim
        used = False
        expected_configuration = _configuration(sim.habitat_config)

        def retained(id_sim, *, config):
            nonlocal used
            if used or id_sim != 'GraphV2SurfaceSim-v0':
                raise RuntimeError('unexpected_persistent_simulator_factory_call')
            if _configuration(config) != expected_configuration:
                raise ValueError('persistent_simulator_configuration_changed')
            if str(sim._current_scene) != str(episode.scene_id):
                raise ValueError('persistent_simulator_scene_changed')
            used = True
            # Falcon's own new-scene path recreates all articulated bodies and
            # reloads NavMesh after env.seed. HabitatSim detects the unchanged
            # actual stage and retains its static physics/collision geometry.
            sim.prev_scene_id = None
            sim.first_setup = True
            return sim

        env_module.make_sim = retained
        try:
            yield
            if not used:
                raise RuntimeError('persistent_simulator_factory_unused')
        finally:
            env_module.make_sim = original

    def _release(self, failed):
        self._owner()
        if not self.active:
            raise RuntimeError('persistent_runtime_lease_not_active')
        self.active = False
        if failed:
            self.close()

    def close(self):
        self._owner()
        if self.active:
            raise RuntimeError('cannot_close_active_persistent_runtime')
        if self.runtime is not None:
            runtime, self.runtime = self.runtime, None
            self.identity = None
            runtime.close()
