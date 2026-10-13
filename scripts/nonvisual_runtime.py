"""Validated structured-observation contract; no visual input is fabricated.

Keep the exact Spot jaw camera's SceneNode transform, FOV, range and native rays.
The node has no visual sensor/render target. Original Manipulator.update moves it.
Image hashes and sensor inventory metadata are explicitly outside this experiment's
structured fingerprint. No physical state, RNG, time, contacts or tracks is omitted.
"""
import copy,hashlib,json
from types import SimpleNamespace

CALIBRATION_NODE='agent_0_articulated_agent_jaw_depth'

def scientific_history(history):
    return [{k:v for k,v in row.items() if k!='sensor_keys'} for row in history]

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def scientific_root(state,history):
    out=copy.deepcopy(state)
    out.pop('image_hash',None)
    out['history_hash']=digest(scientific_history(history))
    return out

def scientific_trace(rows,history):
    # runtime.worker mutates its final recorder row into the fingerprint state.
    # Apply the same visual-only projection there; all other native rows stay exact.
    return [scientific_root(row,history) if 'image_hash' in row and 'history_hash' in row else row for row in rows]

class GeometryNode:
    def __init__(self,node):self._sensor_object=SimpleNamespace(node=node)
    def close(self):pass

def install(rt,mode,metrics):
    assert mode in ['NO_VISUAL','NO_RENDERER','COMBO_NO_RENDERER']
    original_env=rt.habitat.Env.__init__
    def env_init(self,*args,**kwargs):
        from habitat.config import read_write
        cfg=kwargs.get('config',args[0] if args else None)
        # ObservationConfig has already captured the frozen depth-camera calibration.
        with read_write(cfg):
            for agent in cfg.simulator.agents.values():agent.sim_sensors={}
        return original_env(self,*args,**kwargs)
    rt.habitat.Env.__init__=env_init
    original_sanitize=rt.habitat_sim.Simulator._sanitize_config
    def sanitize(config):
        original_sanitize(config)
        config.sim_cfg.create_renderer=(mode=='NO_VISUAL')
        config.sim_cfg.requires_textures=False
        config.sim_cfg.load_semantic_mesh=False
        assert config.sim_cfg.enable_physics,'Physics must remain enabled'
    rt.habitat_sim.Simulator._sanitize_config=staticmethod(sanitize)

    def attach(sim):
        sensors=sim._Simulator__sensors[sim._default_agent_id]
        if CALIBRATION_NODE not in sensors:
            node=sim.agents[sim._default_agent_id].scene_node.create_child()
            sensors[CALIBRATION_NODE]=GeometryNode(node)
        metrics['renderer_present']=float(sim.renderer is not None)
        metrics['visual_sensors']=0
    original_post=rt.habitat_sim.Simulator.__attrs_post_init__
    def post(self):original_post(self);attach(self)
    rt.habitat_sim.Simulator.__attrs_post_init__=post
    original_reconfigure=rt.habitat_sim.Simulator.reconfigure
    def reconfigure(self,*args,**kwargs):
        result=original_reconfigure(self,*args,**kwargs);attach(self);return result
    rt.habitat_sim.Simulator.reconfigure=reconfigure
    original_observations=rt.habitat_sim.Simulator.get_sensor_observations
    def observations(self,*args,**kwargs):
        # This is a pose node, not a renderer observation. Keep it available for
        # unchanged articulation/observer code while excluding it from sensor draw.
        sensors=self._Simulator__sensors[self._default_agent_id]
        node=sensors.pop(CALIBRATION_NODE,None)
        try:return original_observations(self,*args,**kwargs)
        finally:
            if node is not None:sensors[CALIBRATION_NODE]=node
    rt.habitat_sim.Simulator.get_sensor_observations=observations

    def legal(observations,position,yaw,timestamp,previous_command,human_tracks=None):
        from collector.observation import legal_tracks
        return dict(timestamp=float(timestamp),robot_position=list(map(float,position)),
            robot_yaw=float(yaw),previous_command=list(previous_command),
            human_tracks=legal_tracks(human_tracks) if human_tracks is not None else [],
            tracking_available=human_tracks is not None,sensor_keys=[]),{}
    rt.legal_observation=legal
    original_digest=rt.digest
    def structured_digest(value):
        if isinstance(value,list) and value and isinstance(value[0],dict) and 'human_tracks' in value[0] and 'sensor_keys' in value[0]:
            return original_digest(scientific_history(value))
        if isinstance(value,dict) and 'image_hash' in value and 'history_hash' in value and 'robot' in value:
            return original_digest({k:v for k,v in value.items() if k!='image_hash'})
        return original_digest(value)
    rt.digest=structured_digest
    original_write=rt.write
    def write(path,value):
        from pathlib import Path
        if Path(path).name=='image_storage.json':
            value=dict(retained=False,reason='Nonvisual structured observations; geometric jaw calibration and rays retained',fingerprint_version='structured-root-no-images-v1')
        return original_write(path,value)
    rt.write=write
