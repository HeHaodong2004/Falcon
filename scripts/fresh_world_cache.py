"""Validated immutable native asset reuse; a NEW Bullet world per job.

Never call reset to obtain freshness.
The NONE transition forces Habitat 0.3.1 createSceneInstance/initPhysicsManager.
Old scene graphs may accumulate; cap jobs/scenes and recycle the native owner.
"""
import gc
import time

class FreshWorldCache:
    def __init__(self, runtime, metrics, max_jobs=40, max_scenes=3):
        import habitat.core.env as env_module
        from habitat_sim.sim import SimulatorBackend
        self.rt,self.metrics=runtime,metrics
        self.retained=None;self.rebuilding=False;self.jobs=0;self.scenes=set()
        self.max_jobs,self.max_scenes=max_jobs,max_scenes
        self.original_make=env_module.make_sim
        original_backend=runtime.habitat_sim.Simulator._config_backend
        def backend(sim,config):
            if self.rebuilding:
                scene=config.sim_cfg.scene_id
                try:
                    config.sim_cfg.scene_id='NONE'
                    SimulatorBackend.reconfigure(sim,config.sim_cfg)
                finally:config.sim_cfg.scene_id=scene
                SimulatorBackend.reconfigure(sim,config.sim_cfg)
                sim._initialized=True
                metrics['fresh_world_rebuilds']+=1
            else:
                original_backend(sim,config)
                metrics['native_backend_constructions']+=1
        runtime.habitat_sim.Simulator._config_backend=backend
        env_module.make_sim=self.make

    def close(self):
        if self.retained is not None:
            self.retained.close();self.retained=None
            self.jobs=0;self.scenes.clear();gc.collect()

    def make(self,*args,**kwargs):
        scene=str(kwargs['config'].scene)
        if self.retained is not None and (self.jobs>=self.max_jobs or (scene not in self.scenes and len(self.scenes)>=self.max_scenes)):
            self.close();self.metrics['native_owner_recycles']+=1
        self.scenes.add(scene);self.jobs+=1
        if self.retained is None:self.retained=self.original_make(*args,**kwargs)
        else:
            sim=self.retained
            for suite in sim._Simulator__sensors:
                for sensor in suite.values():sensor.close()
            for agent in sim.agents:agent.close()
            sim.__dict__.clear()
            self.rebuilding=True
            try:type(sim).__init__(sim,kwargs['config'])
            finally:self.rebuilding=False
        return self.retained

def install_config_cache(metrics):
    """Cache pristine parsed templates, return a separately unpickled copy per call."""
    import pickle
    import habitat_baselines.config.default as config_module
    original=config_module.get_config;cache={}
    def get_config(*args,**kwargs):
        key=pickle.dumps((args,kwargs))
        if key not in cache:
            cfg=original(*args,**kwargs);cache[key]=pickle.dumps(cfg)
            metrics['config_template_misses']+=1
            return cfg
        metrics['config_template_hits']+=1
        return pickle.loads(cache[key])
    config_module.get_config=get_config
