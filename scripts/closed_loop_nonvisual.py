"""Process-local renderer-free adapter from the validated collection contract.

Only rendering changes. Native physics, clocks, contacts, task actions, raycasts,
and the moving Spot jaw-camera calibration node remain active. No pixels are
fabricated; videos are separate privileged top-down evaluation visualizations.
"""
from types import SimpleNamespace


class GeometryNode:
    def __init__(self,node):
        self._sensor_object=SimpleNamespace(node=node)
    def close(self):
        pass


def install():
    import habitat_sim
    import habitat
    from habitat.config import read_write
    name='agent_0_articulated_agent_jaw_depth'
    original_env=habitat.Env.__init__
    def env_init(self,*args,**kwargs):
        cfg=kwargs.get('config',args[0] if args else None)
        with read_write(cfg):
            for agent in cfg.simulator.agents.values():
                agent.sim_sensors={}
        return original_env(self,*args,**kwargs)
    habitat.Env.__init__=env_init
    original_sanitize=habitat_sim.Simulator._sanitize_config
    def sanitize(config):
        original_sanitize(config)
        config.sim_cfg.create_renderer=False
        config.sim_cfg.requires_textures=False
        config.sim_cfg.load_semantic_mesh=False
        assert config.sim_cfg.enable_physics
    habitat_sim.Simulator._sanitize_config=staticmethod(sanitize)
    def attach(sim):
        sensors=sim._Simulator__sensors[sim._default_agent_id]
        if name not in sensors:
            node=sim.agents[sim._default_agent_id].scene_node.create_child()
            sensors[name]=GeometryNode(node)
        assert sim.renderer is None
    original_post=habitat_sim.Simulator.__attrs_post_init__
    def post(self):
        original_post(self);attach(self)
    habitat_sim.Simulator.__attrs_post_init__=post
    original_reconfigure=habitat_sim.Simulator.reconfigure
    def reconfigure(self,*args,**kwargs):
        result=original_reconfigure(self,*args,**kwargs);attach(self);return result
    habitat_sim.Simulator.reconfigure=reconfigure
    original_observations=habitat_sim.Simulator.get_sensor_observations
    def observations(self,*args,**kwargs):
        sensors=self._Simulator__sensors[self._default_agent_id]
        node=sensors.pop(name,None)
        try:
            return original_observations(self,*args,**kwargs)
        finally:
            if node is not None:
                sensors[name]=node
    habitat_sim.Simulator.get_sensor_observations=observations
