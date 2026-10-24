"""Frozen official depth/PointGoal Falcon reference; CPU inference, no training."""
import numpy as np


class FalconPolicy:
    def __init__(self,runtime):
        import torch
        from gym import spaces
        from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy
        from habitat.tasks.nav.nav import IntegratedPointGoalGPSAndCompassSensor
        from habitat_baselines.config.default import get_config
        from .native import FALCON_CHECKPOINT
        self.torch=torch
        ck=torch.load(FALCON_CHECKPOINT,map_location='cpu');cfg=ck['config']
        from habitat.config import read_write
        with read_write(cfg):cfg.habitat_baselines.rl.auxiliary_losses={}
        sensor_config=get_config('social_nav_v2/falcon_hm3d.yaml').habitat.task.lab_sensors.agent_0_pointgoal_with_gps_compass
        self.goal=IntegratedPointGoalGPSAndCompassSensor(runtime.sim,sensor_config)
        depth=runtime.obs['agent_0_articulated_agent_jaw_depth']
        obs_space=spaces.Dict({'articulated_agent_jaw_depth':spaces.Box(0.,1.,shape=depth.shape,dtype=np.float32),
                              'pointgoal_with_gps_compass':self.goal.observation_space})
        self.policy=PointNavResNetPolicy.from_config(cfg,obs_space,spaces.Discrete(4),agent_name='agent_0')
        self.policy.load_state_dict(ck[0]['state_dict'],strict=True);self.policy.eval()
        self.hidden=torch.zeros(1,*self.policy.hidden_state_shape)
        self.previous=torch.zeros(1,1,dtype=torch.long);self.mask=torch.zeros(1,1,dtype=torch.bool)

    def act(self,runtime):
        torch=self.torch
        observation={'articulated_agent_jaw_depth':runtime.obs['agent_0_articulated_agent_jaw_depth'],
                     'pointgoal_with_gps_compass':self.goal.get_observation(runtime.obs,runtime.episode)}
        batch={k:torch.as_tensor(v).unsqueeze(0) for k,v in observation.items()}
        with torch.inference_mode():
            data=self.policy.act(batch,self.hidden,self.previous,self.mask,deterministic=False)
        self.hidden=data.rnn_hidden_states;self.previous=data.actions;self.mask.fill_(True)
        return ('stop','forward','left','right')[int(data.actions.item())]
