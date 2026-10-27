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
| v9 C2 DWA x inertia K=10 | (running) | | | | | | |

Union of K=10 and K=6: 22/24; 3t8DB4Uzvkt (K=6 only) and LT9Jq6dN3Ea (K=10 only) sit near the edge of the
6000-edge budget and flip with small search changes.  Generator latency P50 36-42 ms per decision.

## 3. Unseen-scene suite (40 episodes, never used for development)
(filled in when the runs and certificates complete)

## 4. Remaining DEV failures
* **Nfvxx8J5NCo**: the narrow passage opening lies at the end of the inflated wall band and is unobserved while
  the robot drives past; neither the normal nor an optimistic (UNKNOWN = FREE) legal route enters it; every
  branch reaches the passage junction 10-20 steps after the witness and meets a returning human.  Fails with a
  15,000-edge budget too: an exploration/observability limit of the query space, not budget.
* **YRUkbU5xsYj**: 49-primitive escape from a confined spawn (now found within ~70 edges by the escape map),
  followed by human timing near (-3, 2.8) where the best branches are already doomed; the budget is consumed
  before alternative timings are explored.
* **3t8DB4Uzvkt / LT9Jq6dN3Ea**: budget-edge sensitivity (each solved by one of the two K settings).
