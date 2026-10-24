"""Resolve and fingerprint the installed task; never infer it from a paper."""
from pathlib import Path
import inspect
from ..config import REPO, digest, file_digest

SCHEMA = 3
CONTRACT = "falcon-kinematic-surface-2"


def installed_contract(config_name="social_nav_v2/falcon_hm3d.yaml"):
    import falcon.additional_action as actions
    import falcon.additional_metric as metrics
    from habitat_baselines.config.default import get_config
    from habitat.tasks.rearrange.actions.actions import BaseVelAction
    from omegaconf import OmegaConf
    c = get_config(config_name)
    sim = c.habitat.simulator
    config = OmegaConf.to_container(c.habitat, resolve=True)
    names = ("discrete_move_forward", "discrete_turn_left", "discrete_turn_right", "discrete_stop")
    selected = {n: config["task"]["actions"]["agent_0_"+n] for n in names}
    if not (sim.kinematic_mode and sim.habitat_sim_v0.allow_sliding and sim.habitat_sim_v0.enable_physics):
        raise ValueError("This adapter requires the audited kinematic/sliding/physics contract")
    if not all(selected[n]["allow_dyn_slide"] for n in names[:3]):
        raise ValueError("Body collision rollback is enabled; a different navigation map is required")
    paths = [Path(inspect.getfile(actions)), Path(inspect.getfile(metrics)), Path(inspect.getfile(BaseVelAction)),
             REPO/"falcon/rearrange_sim_v2.py",
             REPO/"habitat-lab/habitat/core/embodied_task.py",
             REPO/"habitat-lab/habitat/core/env.py",
             REPO/"habitat-lab/habitat/tasks/rearrange/social_nav_v2/socialnav_task.py",
             REPO/"habitat-lab/habitat/articulated_agents/articulated_agent_base.py",
             REPO/"habitat-lab/habitat/articulated_agents/robots/spot_robot.py"]
    sources = {str(p.resolve().relative_to(REPO)): file_digest(p) for p in paths}
    animation = (REPO/selected[names[0]]["leg_animation_checkpoint"]).resolve()
    import habitat_sim
    record = {"schema_version": SCHEMA, "contract": CONTRACT, "sources": sources,
        "config_name": config_name, "actions": selected, "control_frequency": int(sim.ctrl_freq),
        "physics_target_sps": int(c.habitat.task.physics_target_sps),
        "physics_substeps": int(sim.ac_freq_ratio), "step_physics": bool(sim.step_physics),
        "update_articulated_agent": bool(sim.update_articulated_agent),
        "kinematic_mode": True, "allow_sliding": True, "enable_physics": True,
        "base_offset": [0., -.48, 0.], "animation_sha256": file_digest(animation),
        "habitat_sim_version": habitat_sim.__version__,
        "max_episode_steps": int(c.habitat.environment.max_episode_steps),
        "stop_terminates": True, "human_contact_terminates": True,
        "scene_contact_terminates": False,
        "scene_contact_reward_penalty": any(v.get('type')=='SocialNavReward' for v in config['task']['measurements'].values()),
        "configured_measurements": {k:v['type'] for k,v in config['task']['measurements'].items()},
        "wait_action_available": "agent_0_discrete_pause" in config["task"]["actions"],
        "scope": "SE(2) control on native 3D surfaces; benchmark reachability, not physical body safety"}
    record["identity"] = digest(record)
    return record


def audit_installed(output):
    from ..config import write_json
    result = {}
    upstream = Path("/research-user/workspace/baselines/Falcon")
    for corpus in ("hm3d", "mp3d"):
        contract = installed_contract(f"social_nav_v2/falcon_{corpus}.yaml")
        comparison = {n: {"installed": sha, "reference": file_digest(upstream/n) if (upstream/n).exists() else None,
                           "equal": (upstream/n).exists() and file_digest(upstream/n) == sha}
                      for n, sha in contract["sources"].items()}
        result[corpus] = {"contract": contract, "reference_comparison": comparison}
    write_json(output, result)
    return result
