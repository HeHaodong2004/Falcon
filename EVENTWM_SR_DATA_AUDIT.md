# EventWM SR production data audit

Scope: 48 completed roots; 1,504 primary branches. Audit/repeat branches and failed-root attempts do not enter SR statistics. No model training, collection, collector edits, or label rewrites were performed.

**Frozen SR semantics:** native human contact OR declared research human-disc contact OR declared static contact gives valid SR=0, regardless of early termination. With no collision, SR=1 requires completed execution and the full fixed query. All other cases are invalid/null, never a third class.

Full horizon means every planned fixed control was issued, plus duration agreement within the collector’s stored 0.008001 s native-clock quantization tolerance. This preserves the existing clock contract. Collision declarations are taken from saved labels and independently recomputed from saved branch truth with the unchanged `build_labels` function.

The previous 174-sample completed-only event index is **not an SR index** and must not be used to count usable SR data. Its exclusion of early collision terminations is inapplicable here.

All five collected scenes and all four sampling profiles contain valid SR labels. Safe outcomes are concentrated in Castle (150/155); the other five are in `hm3d_NBg5UqG3di3`. Mixed-SR roots occur in those two scenes, while all 12 timing/speed-changing path groups are in Castle. Thus valid labels are widespread, but the observed temporal decision signal is concentrated in Castle.

## Counts by data condition

### Scene

| Condition | Total | SR-valid | Safe | Collision | Invalid |
|---|---:|---:|---:|---:|---:|
| apartment_1 | 264 | 264 | 0 | 264 | 0 |
| castle | 330 | 330 | 150 | 180 | 0 |
| hm3d_CFVBbU9Rsyb | 250 | 250 | 0 | 250 | 0 |
| hm3d_GLAQ4DNUx5U | 330 | 330 | 0 | 330 | 0 |
| hm3d_NBg5UqG3di3 | 330 | 330 | 5 | 325 | 0 |

### Human count

| Condition | Total | SR-valid | Safe | Collision | Invalid |
|---|---:|---:|---:|---:|---:|
| 0 | 314 | 314 | 68 | 246 | 0 |
| 1 | 314 | 314 | 27 | 287 | 0 |
| 2 | 248 | 248 | 26 | 222 | 0 |
| 4 | 314 | 314 | 4 | 310 | 0 |
| 6 | 314 | 314 | 30 | 284 | 0 |

### Root sampling profile

| Condition | Total | SR-valid | Safe | Collision | Invalid |
|---|---:|---:|---:|---:|---:|
| seek_lost | 314 | 314 | 16 | 298 | 0 |
| seek_move | 413 | 413 | 26 | 387 | 0 |
| seek_visible | 438 | 438 | 62 | 376 | 0 |
| wait | 339 | 339 | 51 | 288 | 0 |

### Visibility history

| Condition | Total | SR-valid | Safe | Collision | Invalid |
|---|---:|---:|---:|---:|---:|
| no human observed in history | 975 | 975 | 87 | 888 | 0 |
| human observed in history | 529 | 529 | 68 | 461 | 0 |
| visible track at root | 289 | 289 | 53 | 236 | 0 |
| lost track at root | 273 | 273 | 29 | 244 | 0 |

Visibility rows overlap: visible/lost-at-root are subsets of observed-history, and one root can contain both visible and lost tracks. The first two rows partition the dataset.

### Exclusive visibility categories

| Condition | Total | SR-valid | Safe | Collision | Invalid |
|---|---:|---:|---:|---:|---:|
| lost_at_root | 240 | 240 | 15 | 225 | 0 |
| no_human_observed | 975 | 975 | 87 | 888 | 0 |
| visible_and_lost_at_root | 33 | 33 | 14 | 19 | 0 |
| visible_at_root | 256 | 256 | 39 | 217 | 0 |

## Collision evidence and invalid branches

| Native human | Research disc | Static | Branches |
|---|---|---|---:|
| False | False | True | 1169 |
| False | True | False | 19 |
| False | True | True | 0 |
| True | False | False | 117 |
| True | False | True | 0 |
| True | True | False | 44 |
| True | True | True | 0 |

Human union: 180; native union: 161; research-disc union: 63. The seven rows above are mutually exclusive, so all overlaps are explicit.

Invalid breakdown: execution_failure=0, timeout=0, environment_termination=0, collection_error=0, other=0.

## Root and timing/speed learning signal

| Scene | Roots | Mixed SR | All-safe | All-collision | With invalid |
|---|---:|---:|---:|---:|---:|
| apartment_1 | 8 | 0 | 0 | 8 | 0 |
| castle | 10 | 8 | 2 | 0 | 0 |
| hm3d_CFVBbU9Rsyb | 10 | 0 | 0 | 10 | 0 |
| hm3d_GLAQ4DNUx5U | 10 | 0 | 0 | 10 | 0 |
| hm3d_NBg5UqG3di3 | 10 | 5 | 0 | 5 | 0 |

All-safe/all-collision require every candidate to have that valid label. Mixed means both valid labels occur; invalid presence is an independent flag.

Spatial-path groups are scoped to the same root and the exact canonical `spatial_path` SHA256; variants differ in requested/effective wait or speed. 364 groups have at least two valid variants; **12 change SR across variants**. WAIT-only singleton groups are not eligible.

| Scene | Groups with ≥2 valid variants | Groups changing SR |
|---|---:|---:|
| apartment_1 | 64 | 0 |
| castle | 80 | 12 |
| hm3d_CFVBbU9Rsyb | 60 | 0 |
| hm3d_GLAQ4DNUx5U | 80 | 0 |
| hm3d_NBg5UqG3di3 | 80 | 0 |

Examples (all variants share the listed root and exact path):

| Root | Path SHA256 prefix | Variants → SR |
|---|---|---|
| r_38554a3f79aeb025 | 1e7cd5b07139 | wait=0,speed=1 → 0; wait=0.5,speed=1 → 1; wait=1,speed=1 → 1; wait=0,speed=0.5 → 0 |
| r_62a4b3b116251750 | fecd10d85ffa | wait=0,speed=1 → 1; wait=0.5,speed=1 → 0; wait=1,speed=1 → 0; wait=0,speed=0.5 → 0 |
| r_6cf2c44bfa8dd905 | 64db4222b26b | wait=0,speed=1 → 1; wait=0.5,speed=1 → 0; wait=1,speed=1 → 0; wait=0,speed=0.5 → 1 |
| r_7166d348fa64da90 | 184a6271ea08 | wait=0,speed=1 → 1; wait=0.5,speed=1 → 1; wait=1,speed=1 → 0; wait=0,speed=0.5 → 1 |

## Failed roots (separate from branch labels)

52 failed roots, 52 failed attempts. Root-level failures contribute zero rows to primary SR statistics.

| Scene | Human count | Profile | Failure category | Roots |
|---|---:|---|---|---:|
| apartment_1 | 2 | seek_lost | prefix_termination_before_root | 1 |
| apartment_1 | 2 | wait | prefix_termination_before_root | 1 |
| coda | 0 | seek_lost | missing_assets | 1 |
| coda | 0 | wait | missing_assets | 1 |
| coda | 1 | seek_visible | missing_assets | 1 |
| coda | 1 | wait | missing_assets | 1 |
| coda | 2 | seek_move | missing_assets | 1 |
| coda | 2 | seek_visible | missing_assets | 1 |
| coda | 4 | seek_lost | missing_assets | 1 |
| coda | 4 | seek_move | missing_assets | 1 |
| coda | 6 | seek_lost | missing_assets | 1 |
| coda | 6 | wait | missing_assets | 1 |
| mp3d_17DRP5sb8fy | 0 | seek_lost | missing_assets | 1 |
| mp3d_17DRP5sb8fy | 0 | seek_move | missing_assets | 1 |
| mp3d_17DRP5sb8fy | 1 | seek_lost | missing_assets | 1 |
| mp3d_17DRP5sb8fy | 1 | wait | missing_assets | 1 |
| mp3d_17DRP5sb8fy | 2 | seek_visible | missing_assets | 1 |
| mp3d_17DRP5sb8fy | 2 | wait | missing_assets | 1 |
| mp3d_17DRP5sb8fy | 4 | seek_move | missing_assets | 1 |
| mp3d_17DRP5sb8fy | 4 | seek_visible | missing_assets | 1 |
| mp3d_17DRP5sb8fy | 6 | seek_lost | missing_assets | 1 |
| mp3d_17DRP5sb8fy | 6 | seek_move | missing_assets | 1 |
| replicacad_apt_0 | 0 | seek_lost | replicacad_config_initialization | 1 |
| replicacad_apt_0 | 0 | wait | replicacad_config_initialization | 1 |
| replicacad_apt_0 | 1 | seek_visible | replicacad_config_initialization | 1 |
| replicacad_apt_0 | 1 | wait | replicacad_config_initialization | 1 |
| replicacad_apt_0 | 2 | seek_move | replicacad_config_initialization | 1 |
| replicacad_apt_0 | 2 | seek_visible | replicacad_config_initialization | 1 |
| replicacad_apt_0 | 4 | seek_lost | replicacad_config_initialization | 1 |
| replicacad_apt_0 | 4 | seek_move | replicacad_config_initialization | 1 |
| replicacad_apt_0 | 6 | seek_lost | replicacad_config_initialization | 1 |
| replicacad_apt_0 | 6 | wait | replicacad_config_initialization | 1 |
| replicacad_apt_1 | 0 | seek_visible | replicacad_config_initialization | 1 |
| replicacad_apt_1 | 0 | wait | replicacad_config_initialization | 1 |
| replicacad_apt_1 | 1 | seek_move | replicacad_config_initialization | 1 |
| replicacad_apt_1 | 1 | seek_visible | replicacad_config_initialization | 1 |
| replicacad_apt_1 | 2 | seek_lost | replicacad_config_initialization | 1 |
| replicacad_apt_1 | 2 | seek_move | replicacad_config_initialization | 1 |
| replicacad_apt_1 | 4 | seek_lost | replicacad_config_initialization | 1 |
| replicacad_apt_1 | 4 | wait | replicacad_config_initialization | 1 |
| replicacad_apt_1 | 6 | seek_visible | replicacad_config_initialization | 1 |
| replicacad_apt_1 | 6 | wait | replicacad_config_initialization | 1 |
| replicacad_apt_2 | 0 | seek_move | replicacad_config_initialization | 1 |
| replicacad_apt_2 | 0 | seek_visible | replicacad_config_initialization | 1 |
| replicacad_apt_2 | 1 | seek_lost | replicacad_config_initialization | 1 |
| replicacad_apt_2 | 1 | seek_move | replicacad_config_initialization | 1 |
| replicacad_apt_2 | 2 | seek_lost | replicacad_config_initialization | 1 |
| replicacad_apt_2 | 2 | wait | replicacad_config_initialization | 1 |
| replicacad_apt_2 | 4 | seek_visible | replicacad_config_initialization | 1 |
| replicacad_apt_2 | 4 | wait | replicacad_config_initialization | 1 |
| replicacad_apt_2 | 6 | seek_move | replicacad_config_initialization | 1 |
| replicacad_apt_2 | 6 | seek_visible | replicacad_config_initialization | 1 |

Missing assets: MP3D `17DRP5sb8fy.glb` and `coda.glb` fail with `No Stage Attributes exists for requested scene`. ReplicaCAD: unresolved stage/object/URDF assets end with `Stage Template managed object handle `` not found in ManagedContainer, so aborting.` These are grouped by the terminal initialization failure, not double-counted as missing assets. Prefix failures end with `RuntimeError: Prefix terminated before root`; the retained artifacts do not identify a static-versus-human contact cause. Exact log text, path/line references, execution return codes, and per-root profile are in `sr_failed_root_audit.json`.

## Consistency and artifacts

**PASS**: 0 contradictions. Rechecked completion markers and every sealed file for all completed roots, primary/label inventory, global query-ID uniqueness, root/query/branch IDs, fixed-query hashes, labels recomputed from raw truth, durations/control coverage, declared termination evidence, and all SR invariants. Existing collector integrity audits remain PASS (human reactivity is not an integrity gate). No data were silently corrected.

Excluded 144 audit/repeat branches. The full per-root audit contains all 48 roots, including candidate counts and category flags.

Artifacts under `datasets/raw/eventwm_prod_v1/audits/`:

- `sr_branch_audit.csv` / `.json`: all primary branches, metadata, collision evidence, binary SR/validity, path/query descriptors.
- `sr_root_audit.csv` / `.json`: every completed root and its learning signal.
- `sr_path_variant_audit.json`: every path group and exact query variants/outcomes.
- `sr_failed_root_audit.csv` / `.json`: every failed root with exact evidence.
- `sr_summary.json`: counts, breakdowns, provenance, check counts, contradictions.

Reproduce (offline audit only):

```bash
/home/haodong/miniforge3/envs/eventwm-falcon/bin/python scripts/audit_eventwm_sr.py
```

## Final summary

```text
TOTAL PRIMARY BRANCHES: 1504
SR-VALID: 1504
  SAFE (SR=1): 155
  COLLISION (SR=0): 1349
SR-INVALID / UNKNOWN: 0

SR-VALID RATE: 100.00%

COLLISION BREAKDOWN:
  HUMAN: 180 (human only; union=180)
  STATIC: 1169 (static only; union=1169)
  HUMAN+STATIC: 0

INVALID BREAKDOWN: execution_failure=0, timeout=0, environment_termination=0, collection_error=0, other=0

MIXED-SR ROOTS: 13
ALL-SAFE ROOTS: 2
ALL-COLLISION ROOTS: 33
ROOTS WITH INVALID QUERIES: 0

TIMING/SPEED GROUPS WITH DIFFERENT SR: 12 / 364 eligible groups

FAILED ROOT TAXONOMY: prefix_termination_before_root=2, missing_assets=20, replicacad_config_initialization=30

DATA CONSISTENCY: PASS
```
