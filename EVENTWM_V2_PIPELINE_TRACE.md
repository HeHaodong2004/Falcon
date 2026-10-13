# Existing pipeline traced before v2 edits

`collector.runtime.worker`: fresh seeded environment -> unchanged real prefix (`root_sampling.prefix_commands`) -> `LegalRoot` position/yaw/native clocks -> `core.generate_queries` -> eight 1.5 m directional targets -> `runtime.StaticPlanner.path` snap/project + shortest path on an independent 0.25 m-radius, 1.41 m-height static navmesh -> `core.compile_motion` fixed native velocity commands and nominal timed trajectory -> immediate / 0.5 s wait / 1 s wait / half-speed -> explicit WAIT matching longest horizon -> fresh-process worker replay per query -> passive native contact recorder -> `core.build_labels`.

Source of each contract:
- Targets, variants, WAIT, immutable canonical query/hash: `scripts/collector/core.py`, `generate_queries`, `compile_motion`, `Query`.
- Static map/path: `scripts/collector/runtime.py:StaticPlanner`; map creation `scripts/controlled/generate.py` uses radius .25, height 1.41, includes static objects.
- Robot: `SpotRobot` in `habitat-lab/habitat/articulated_agents/robots/spot_robot.py`; articulated legs and arm, base offset [0,-.48,0], navmesh offsets [0,0], [±.25,0]. The .25 navmesh/research radius is NOT a bound on the whole robot collision geometry.
- Execution: runtime's `step` -> unchanged Habitat `BaseVelAction.step/update_base`, `actions.py`. Native command integrated at 1/120 s; navmesh step filter may clip movement; no arbitrary motion replanning inside a fixed query.
- Declared contact: `runtime.Recorder.capture` at native substeps; robot-human object-ID contact; static contact excludes near-vertical ground support below base+.12 m. No contact definitions change in v2.
- Research human-disc: `core.build_labels`, planar center separation <= .25+.30 m. Native-human OR research-disc forms human event.
- Termination: worker prioritizes native human, then static, then environment termination, endpoint execution failure; no branch replay/controller changes required.
- Legal input: `core.load_wm_input` + nested `observation.legal_tracks` allowlists. No future truth, labels or controller state.

V1 omitted a full swept-motion/robot-geometry static precheck. Several non-Castle v1 roots already had static contact at timestamp zero, so target-only checks cannot solve the contract. V2 must check the root pose as well as every fixed variant, reject insufficient candidate sets, and quarantine/stop on any unexpected static contact.
