# FALCON benchmark human dynamics: frozen defaults

The targeted pipeline retains the benchmark-native human behavior. No reactive controller is added, no activation switch is enabled, and no speed, goal, spawn or avoidance parameter is changed. The user's final clarification makes **benchmark fidelity**, rather than robot reactivity, the requirement.

## Source and configuration

The audited upstream baseline is `Zeying-Gong/Falcon` main at `d15ef512c4b1451f9daeaf6d4d7ecc702ed57430`. The local human action, simulator, task, structured defaults and `social_nav_v2/falcon_hm3d.yaml` files match that commit byte for byte. `scripts/controlled/run_case.py::build_config` loads that same benchmark configuration. The runtime registry resolves `OracleNavRandCoordAction_Obstacle` to this repository's `falcon/additional_action.py`.

`scripts/eventwm_human_defaults_audit.py` verifies those source bytes and compares all six human action configurations in every completed pilot root's resolved configuration. Missing, changed or added action fields fail the audit. Evidence is saved in `output/eventwm_benchmark_human_audit/defaults_audit.json`.

| Mechanism | Exact implementation and default |
|---|---|
| Human-human avoidance | `falcon/additional_action.py::OracleNavObstacleAction.step`: iterates `range(1, num_articulated_agents)`, excludes self; responds to other humans with geodesic distance <2m and forward dot product >0.5 once previous positions exist |
| Avoidance update | `OracleNavObstacleAction.update_rel_targ_obstacle`: velocity-scaled, distance-weighted repulsion combined with path target; preserve actual implementation rather than substituting an ORCA solver |
| Robot response | Robot is agent 0, outside that avoidance loop |
| Optional robot-distance activation | `OracleNavRandCoordAction_Obstacle.step`: `human_stop_and_walk_to_robot_distance_threshold=-1` disables it; enabling it requires exactly two allocated agents and changes stop/walk speed activation, not the default reciprocal-avoidance behavior |
| Motion and speed | `human_joints`, 17 joints, lin/ang speed 10, forward/turn velocity 1, turn threshold 0.1, distance threshold 0.2; inherited random multiplier Uniform(0.8,1.2) |
| Spawn/goals | Existing episode initialization and native two-goal mechanism remain unchanged; explicit pilot seeds control inherited random draws |
| Other action defaults | `allow_dyn_slide=True`, `allow_back=True`, spawn distance 2m, 200 spawn attempts; all six resolved slots retain activation threshold -1 |

**Humans may react to other humans. Humans do not react to robot agent 0 under the default configuration. The human process is ego-independent with respect to the robot, but not necessarily independent of other humans.** Native robot-human contact can terminate an episode; this is distinct from robot-conditioned human avoidance/replanning.

The paper's “Natural Human Behaviors” paragraph describes humans reacting to each other. It does not establish that the released default controller avoids robot agent 0. Its ORCA wording should not override the actual released action implementation. The official README points to the main configuration; the releases page lists no releases. No official alternative configuration for the reported results was identified in the paper, project page or repository documentation inspected on 2026-10-06. Sources: [paper](https://arxiv.org/html/2409.13244v2), [official repository](https://github.com/Zeying-Gong/Falcon), [releases](https://github.com/Zeying-Gong/Falcon/releases), [pinned action implementation](https://github.com/Zeying-Gong/Falcon/blob/d15ef512c4b1451f9daeaf6d4d7ecc702ed57430/falcon/additional_action.py).

## Interpretation of diagnostics and pilot status

The inherited `human_reactivity` diagnostic compares the first active human trajectory under movement and WAIT queries over their common recorded prefix. Its FAIL means no robot-conditioned trajectory difference was detected. That result is expected under these defaults; the diagnostic is retained without weakening or relabeling it. Source/configuration equivalence is the fidelity gate. A first-human trajectory comparison is not a proof about every human or an empirical test of human-human avoidance.

The targeted pilot therefore **does not need to be marked non-benchmark solely for lacking robot reactivity**, and no replacement pilot with altered humans is warranted. It remains a bounded sampling-validation pilot, not a production-scale run. Existing controlled episode fixtures, scene selection, root sampling and static query protocol define its distribution; matching human controllers does not establish equivalence to the entire Social-HM3D/MP3D benchmark episode distribution.

The binary training view remains `P(native physical human collision | legal history, statically feasible fixed query)`. Saved v3 categories remain unchanged: class 2 positive, classes 0 and 1 negative, with class 1 retained as a hard-negative stratum. Root strata, distance/age coverage, N=8 timing/speed groups, raw branch retention, separate natural holdout and offline 40/30/30 collision/social/safe sampling remain intact.

```bash
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/eventwm_human_defaults_audit.py --dataset datasets/raw/eventwm_social_outcome_v3_wm_train_pilot_01 --output output/eventwm_benchmark_human_audit/defaults_audit.json
```
