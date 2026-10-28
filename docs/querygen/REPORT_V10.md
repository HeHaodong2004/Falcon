# v10 report: portfolio Oracle, corrected executor and dynamics, second unseen suite

Frozen system: tag `querygen-freeze-v10` (commit aa09bf88).  Protocol: `FINAL_PROTOCOL_V10.md` (declared before any
run on the second unseen suite).  Results: `output/runs/final_v10/` (test) and `output/runs/final_v10_devset/`
(frozen system on the development set); `output/` -> `/var/tmp/falcon-querygen-claude/runs`.  Rules unchanged:
legal-only generator/planners/execution, privileged Oracle selection only, strict PASS = Oracle success +
bit-identical continuous replay from reset with the unchanged `habitat.Env.step`, official success, zero human
collision, zero body-proxy overlap (robot rectangle >= 0.25 m from human centres), zero static contact (declared
spawn-escape rule) and, new in v10, zero dynamics-model violations.

## 1. What changed from v9
| layer | change | evidence (development set) |
|---|---|---|
| executor | in-place rotations end on the primitive that reaches the target heading (v9 spent one more step on a native hold = a right turn: `rot_left1` was a no-op) | pUneSGJDrvY: tree exhausted after 587 edges in v9, PASS in 39 s |
| executor | goto arrival ends on the arriving step | same defect class |
| executor / dynamics | momentum robots brake to rest before STOP; DWA's blocked fallback brakes at the brake limit (v9: instantaneous stop) | 3 v9 C2 solutions relied on an instantaneous stop (REPORT_V9 section 2) |
| verdict | strict PASS also requires zero dynamics violations (`dynamics.dynamics_violations`) | |
| generator | K=6 reset-state coverage: both route turns at the reset state | TQSiMZJawkS K=6: no root candidate started with the only free primitive (left) |
| generator / planners | goal relaxation: a planning target inside observed BLOCKED cells seeds the field from passable cells within 0.5 m | 13 of 68 development goals lie within 0.1 m of the navmesh edge; LEFTm3JecaC had every goal query 'unreachable' (205 -> 144 steps) |
| Oracle | space-time model radius 0.5 m + cell-sound fallback table (member 0) | 0.6 m declared corridors unreachable that witnesses pass (TEEsavR23oF, iigzG1rtanx) |
| Oracle | **two-member portfolio** under one 12,000-edge budget: weight 3.5 with the 0.5 m model, weight 2 with the v9 0.6 m model | single-order search is heavy-tailed (episodes moved between tens and >12,000 edges with unrelated changes); see section 5 |

TBD: sections 2-6 are filled from the final runs.
