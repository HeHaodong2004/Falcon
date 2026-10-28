# Official Falcon rules and feasibility accounting (binding since 2026-10-28)

User decision (2026-10-28): follow the official Falcon benchmark settings and evaluation protocol only; no
additional safety constraints or stricter geometric assumptions; target >= 95% Oracle SR over every episode that
is not rigorously proven infeasible in the original simulator.

## Rules, verified in this repository
| item | official definition | source |
|---|---|---|
| actions | `agent_0_discrete_move_forward` (lin_speed 25), `agent_0_discrete_turn_left/right` (ang_speed +-10), `agent_0_discrete_stop`; `allow_dyn_slide: True` | `habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d.yaml` |
| episode length | `max_episode_steps: 500` | `habitat-lab/habitat/config/benchmark/nav/socialnav_v2/falcon_hm3d_task.yaml` |
| success | `Success`: STOP called and `distance_to_goal < success_distance` (0.2); `end_on_success: True` | `habitat-lab/habitat/tasks/nav/nav.py`, `default_structured_configs.py`, `socialnav_task_detail.yaml` |
| collision | `DidMultiAgentsCollide`: a physics contact point between the robot and any human agent; `HumanCollision` = 1 from the first contact on and sets `task.should_end = True` (the episode ends) | `falcon/additional_metric.py` |
| PSC | share of steps with every human >= 1.0 m from the robot (metric only) | `falcon/additional_metric.py` |
| static contact | not a benchmark rule (no measure, no termination) | -- |

Our earlier "declared" rules -- zero static contact outside a spawn-escape window, robot rectangle >= 0.25 m from
human centres -- are NOT part of the benchmark.  They are kept only as `--rules declared` for reproducing old runs.

## Verdict (`evaluate.py`, `--rules official`, default)
PASS iff the authoritative continuous replay from reset with the unchanged `Env.step` reproduces the Oracle
branch's actions exactly and the official `success` measure is 1.  Human collision, PSC, SPL and static-contact /
proximity counts are reported, never added to the verdict.  The Oracle's branches end only on official human
contact, STOP or the step limit.  Configurations use the official discrete action space (C1 ORCA x native,
C2 DWA x native); the earlier inertia velocity action is outside the official benchmark.

## Feasibility accounting (`tests/querygen/native_accounting.py`)
* **confirmed feasible**: a replay-verified native witness (`witness.py --rules native`, or any declared-rule
  witness, which is stricter) or a replay-verified official success of any run;
* **natively infeasible**: only an exact certificate under the official rules (`native_cert.py`: every native
  action sequence enumerated, merging only bitwise-identical robot states, a branch closed only by official human
  contact) whose closed sequences all reproduce the official human-collision measure in the official
  environment;
* **unresolved**: everything else, **counted in the denominator**.

Declared-rule certificates (start escape, exact-gait progress, goal region) are void: they assumed the static
contact / body-proxy rules.  Headline SR = PASS / (episodes - natively infeasible).

## Status (2026-10-28, updated 12:40)
| suite | episodes | confirmed feasible | natively infeasible | unresolved |
|---|---|---|---|---|
| DEV | 28 | 28 | 0 | 0 |
| first unseen | 40 | 39 | 0 | 1 (nMeXfQU4PMS) |
| second unseen | 60 | 57 | 3 (5Poh4Qz68hd, sKLMLpTHeUy: every first action -> official contact at step 1; AMEM2eWycTq: every sequence by step 6) | 0 |

(Updated 12:40: 2-hour native witness searches found sT4fr6TAbpF -- formerly "goal region unreachable for Spot's
body" under the declared rules -- in 146 steps and Wo6kuutE9i7 in 131 steps.)

All 18 episodes previously reported as "certified infeasible" under the declared rules that have since been
searched natively are natively feasible except the three certified above.

## Re-scored earlier results (headline = PASS / all episodes not proven infeasible)
| run | headline SR |
|---|---|
| v9 DEV C1 K10 / K6 / C2 | 21/28 (75.0%) / 21/28 (75.0%) / 18/28 as run (64.3%) |
| v9 first unseen C1 K10 / K6 / C2 | 28/40 (70.0%) / 25/40 (62.5%) / 25/40 as run (62.5%) |
(These runs searched and judged under the declared rules; official-rule runs of the current system follow.)

## Evaluation plan
Development: DEV + first + second unseen suites (128 episodes, all consumed).  Test: the third unseen suite
(`suites_unseen3.json`, frozen 2026-10-28 07:47, never run), after a declared freeze; its native witnesses and
certificates are computed only after the freeze.
