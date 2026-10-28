# v9 report: jointly optimised planner / executor / query generator / Oracle

Frozen system: tag `querygen-freeze-v9` (commit aea47081).  Protocol: `FINAL_PROTOCOL_V9.md` (declared before
any unseen run).  Results: `output/runs/final_v9/` (`/var/tmp/falcon-querygen-claude/runs/final_v9`).  Rules
unchanged: legal-only generator/planners/execution, privileged Oracle selection only, strict PASS = Oracle
success + bit-identical continuous replay from reset with the unchanged `habitat.Env.step`, official success,
zero human collision, zero body-proxy overlap (robot rectangle >= 0.25 m from human centres), zero static
contact (declared spawn-escape rule).

## 1. What changed from the frozen v6/v7b system
| layer | change | evidence |
|---|---|---|
| executor (ORCA) | turn commit + non-holonomic speed governor (no plan/ORCA left-right dithering) | Qpor2mEya8F goal query: ~25 of 58 steps wasted before |
| executor | query options `tight` (outer ring of the legal wall band treated as UNKNOWN), `close` (body-width ORCA proxies, 0.5 s horizon, 0.27 m rectangle check; stateless dynamics only) | Qpor2mEya8F, bzCsHPLDztK, bCPU9suPUw9 first solved with these |
| query | `st_goal`: legal space-time lattice plan (exact native primitives incl. holds, legal map, legal constant-velocity human predictions, growing margin), re-planned every 4 steps | solves the same three independently; ~12 ms per plan |
| generator | mixed mode: conservative queries + explicit squeeze variants (`goal_tc`, `st_goal_tc`); approach trigger by legal route cost-to-go; K=10 quotas from solution usage; manoeuvre mode only near start/goal; creep-first manoeuvre order | LEFTm3JecaC, TEEsavR23oF, SByzJLxpRGn, 3t8DB4Uzvkt solved; defects found by regression runs listed in PROGRESS.md |
| Oracle | heading-aware heuristic; spawn escape map (exact BFS, A* fallback); escape penalty for spawn manoeuvres; exact duplicate skip before generation; exact sibling prefix sharing (cache) | ~2x faster per edge; equivalence tests bit-identical |
| DWA | exact per-call memo (2.8x faster planner) | equivalence test identical |
| witnesses | margin searches, long searches; Oracle solutions count as feasibility proofs | 3t8DB4Uzvkt feasible (400-step witness) |

## 2. DEV (24 feasible of 28; 4 certified infeasible)
| config | feasible SR | strict SR | failures | SPL | min human clearance | Oracle edges P50/P95 | Oracle s P50/P95 |
|---|---|---|---|---|---|---|---|
| v6 frozen (`querygen-freeze-v1`) | 14/24 (58%) | 14/28 | 10 | | | 603 median | |
| v7b K=10 (previous best) | 15/24 (62.5%) | 15/28 | 9 | | | | |
| **v9 C1 ORCA x native K=10** | **21/24 (87.5%)** | 21/28 | 3t8DB4Uzvkt, Nfvxx8J5NCo, YRUkbU5xsYj | 0.866 | 0.281 m | 339 / 2548 | 203 / 1466 |
| **v9 C1 ORCA x native K=6** | **21/24 (87.5%)** | 21/28 | LT9Jq6dN3Ea, Nfvxx8J5NCo, YRUkbU5xsYj | 0.855 | 0.286 m | 217 / 3024 | 207 / 1579 |
| v9 C2 DWA x inertia K=10 | 18/24 as run; **16/24 (66.7%) physically consistent** (see note) | 16/28 | 3t8DB4Uzvkt, LEFTm3JecaC, Nfvxx8J5NCo, SByzJLxpRGn, TEEsavR23oF, YRUkbU5xsYj (+ AWUFxHEyV3T, XNeHsjL6nBB) | 0.871 | 0.268 m | 433 / 3153 | 335 / 2278 |

**C2 physics audit (post-freeze).**  The strict replay checked contacts, not the inertia model itself.
A dynamics checker (`dynamics.dynamics_violations`: speed-level change beyond accel/brake, turning while
moving, STOP while moving) finds violations in 10 C2 solutions: 9 issued STOP while still moving (the
Oracle's goal test) and 3 contain an *instantaneous stop* from the DWA's "blocked" fallback (zero velocity
commanded while the speed level only dropped by the brake limit).  Replaying the recorded query sequences
under the corrected rules (brake to rest before STOP; a blocked DWA brakes) with
`tests/querygen/rebrake_check.py`: the 7 STOP-only cases still pass; the 3 instantaneous-stop cases fail
(AWUFxHEyV3T and XNeHsjL6nBB on DEV, 77mMEyxhs44 unseen).  The physically consistent v9 C2 results
are therefore DEV 16/24.  v10 fixes both defects and the strict verdict now requires zero dynamics
violations.

Union of K=10 and K=6: 22/24; 3t8DB4Uzvkt (K=6 only) and LT9Jq6dN3Ea (K=10 only) sit near the edge of the
6000-edge budget and flip with small search changes.  Generator latency P50 36-42 ms per decision.

## 3. Unseen-scene suite (40 episodes, never used for development)
Feasibility (privileged, computed after the freeze): 25 verified witnesses (1,800 s searches), 1 more
from 9,000 s searches (W16Bm4ysK8v, 108 steps), plus 5 episodes proven feasible by a replay-verified
Oracle solution of some configuration (iePHCSf119p, iigzG1rtanx, mJXqzFtmKg4, Pm6F8kyY3z2, PuKPg4mmafe)
= **31 feasible**; **7 certified infeasible** (sT4fr6TAbpF: goal region unreachable for Spot's body;
D8bT1ambLFc, FgswoxWb3uN, s19Uyn7AWwv, JeFG25nYj2p, VLzqgDo317F: no escape from the spawn; gDDiZeyaVc2:
exact-gait progress search exhausted); **2 unresolved** (kdFEfVoT1WE, nMeXfQU4PMS: every certificate
passes, no witness in 9,000 s at two search weights; a sound space-time relaxation (`stcert.py`) does
not certify them either).

| config | feasible SR | strict SR | feasible failures | SPL | steps P50 | min human clearance (min / P50) | Oracle edges P50/P95 | Oracle s P50/P95 | generator ms P50/P95 |
|---|---|---|---|---|---|---|---|---|---|
| **C1 ORCA x native K=10** | **28/31 (90.3%)** | 28/40 | pUneSGJDrvY, zUG6FL9TYeR, W16Bm4ysK8v | 0.841 | 115 | 0.275 / 0.439 m | 102 / 5475 | 96 / 4993 | 38 / 74 |
| **C1 ORCA x native K=6** | **25/31 (80.6%)** | 25/40 | TQSiMZJawkS, iePHCSf119p, p4ZPcGtk6Ex, pUneSGJDrvY, zUG6FL9TYeR, W16Bm4ysK8v | 0.819 | 103 | 0.251 / 0.425 m | 47 / 2703 | 86 / 2414 | 41 / 81 |
| C2 DWA x inertia K=10 | (running) | | | | | | | | |

The union of K=10 and K=6 is 28/31.

## 4. Remaining DEV failures
* **Nfvxx8J5NCo**: the narrow passage opening lies at the end of the inflated wall band and is unobserved while
  the robot drives past; neither the normal nor an optimistic (UNKNOWN = FREE) legal route enters it; every
  branch reaches the passage junction 10-20 steps after the witness and meets a returning human.  Fails with a
  15,000-edge budget too: an exploration/observability limit of the query space, not budget.
* **YRUkbU5xsYj**: 49-primitive escape from a confined spawn (now found within ~70 edges by the escape map),
  followed by human timing near (-3, 2.8) where the best branches are already doomed; the budget is consumed
  before alternative timings are explored.
* **3t8DB4Uzvkt / LT9Jq6dN3Ea**: budget-edge sensitivity (each solved by one of the two K settings).

## 5. Defects found after the freeze (fixed in v10, not in the v9 numbers)
* **In-place rotations ended one step late.**  The executor noticed "heading reached" at the start of
  the next step and spent that step on a native hold, which is itself an in-place turn starting right:
  `rot_left1` = left+right (a no-op), `rot_left<k>` turned k-1 steps, `rot_right<k>` k+1.  This alone
  explains pUneSGJDrvY (wall-side spawn needing a left arc: no left turn was expressible; the tree was
  exhausted after 587 edges; with the fix it is solved in 39 s).
* **K=6 reset-state coverage.**  At TQSiMZJawkS's spawn only a left turn is free; every K=6 root candidate
  started with a right turn or a forward step.
* **Physics of the C2 robot** (section 2 note): STOP while moving and an instantaneous stop in the DWA
  "blocked" fallback; the strict verdict now checks the dynamics model.
* **goto arrival** also spent one extra hold step (a stray in-place turn for the native robot).
