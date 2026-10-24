"""Regenerate shared scene inputs using the Work sampling algorithm."""
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controlled_matrix import REPO, COUNTS, read, write, digest, safe_name, validate


def generate(args):
    # Native import order is required on the workstation.
    import habitat_sim
    import numpy as np
    from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index

    root = args.output
    if root.exists() and any(root.iterdir()):
        raise ValueError('Generation requires a new/empty output directory')
    scenes = read(args.scene_manifest)
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    cases = []
    for index, scene in enumerate(scenes, 1):
        safe_name(scene['name'])
        scene['index'] = index
        source = REPO / scene['scene_id']
        if source.name.endswith('.scene_instance.json'):
            instance = read(source)
            for obj in instance.get('object_instances', []):
                obj['motion_type'] = 'STATIC'
            dest = root / 'scene_instances' / source.name
            write(dest, instance)
            scene['local_scene'] = str(dest.relative_to(root))
            scene['source_sha256'] = digest(source)
            source = dest
        settings = habitat_sim.SimulatorConfiguration()
        settings.scene_id = str(source.resolve())
        settings.scene_dataset_config_file = scene['scene_dataset']
        settings.create_renderer = False
        settings.enable_physics = True
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = []
        sim = habitat_sim.Simulator(habitat_sim.Configuration(settings, [agent]))
        try:
            nav = habitat_sim.NavMeshSettings()
            nav.set_defaults()
            nav.agent_radius, nav.agent_height = .25, 1.41
            nav.include_static_objects = True
            if not sim.recompute_navmesh(sim.pathfinder, nav):
                raise ValueError(f"Navmesh generation failed for {scene['name']}")
            mesh = root / 'navmeshes' / (scene['name'] + '.navmesh')
            mesh.parent.mkdir(exist_ok=True)
            sim.pathfinder.save_nav_mesh(str(mesh))
            scene['navmesh'] = str(mesh.relative_to(root))
            island = get_largest_island_index(sim.pathfinder, sim, allow_outdoor=False)
            seed = args.seed_base + index
            sim.seed(seed)
            rng = np.random.RandomState(seed)

            def point():
                return sim.pathfinder.get_random_navigable_point(island_index=island)

            def distance(x, y):
                path = habitat_sim.ShortestPath()
                path.requested_start, path.requested_end = x, y
                return path.geodesic_distance if sim.pathfinder.find_path(path) else float('inf')

            for _ in range(30000):
                start, goal = point(), point()
                d = distance(start, goal)
                if abs(float(start[1]-goal[1])) < .25 and 4 < d < 10:
                    break
            else:
                raise ValueError('No connected 4–10 m start/goal pair')

            def sample(avoid):
                for _ in range(30000):
                    q = point()
                    if abs(float(q[1]-start[1])) > .3 or np.linalg.norm((q-start)[[0, 2]]) < 1.2:
                        continue
                    if any(np.linalg.norm((q-x)[[0, 2]]) < .8 for x in avoid):
                        continue
                    if np.isfinite(distance(start, q)):
                        return q
                raise ValueError('No separated human point')

            info = dict(geodesic_distance=float(d), sampling_seed=seed, policy_seed=seed,
                        scene_dataset=scene['scene_dataset'], scene_name=scene['name'], island_index=island)
            chosen = []
            for h in range(6):
                q = sample(chosen)
                chosen.append(q)
                for k in range(3):
                    pos = q if k == 0 else sample([q])
                    info[f'human_{h}_waypoint_{k}_position'] = pos.tolist()
                    info[f'human_{h}_waypoint_{k}_rotation'] = float(rng.uniform(0, 2*np.pi))
            yaw = math.atan2(-(goal[0]-start[0]), -(goal[2]-start[2]))
            for j, humans in enumerate(COUNTS):
                idx = (index-1)*5+j+1
                tag = f"{idx:03}_{scene['name']}_h{humans}"
                ep = dict(episode_id=str(3000+idx), scene_id=scene['scene_id'],
                          start_position=start.tolist(), start_rotation=[0, math.sin(yaw/2), 0, math.cos(yaw/2)],
                          goals=[dict(position=goal.tolist(), radius=.2)], info=dict(info, human_num=humans))
                dest = root / 'episodes' / (tag+'.json')
                write(dest, ep)
                cases.append(dict(index=idx, tag=tag, scene=scene['name'], humans=humans,
                                  seed=seed, episode=str(dest.relative_to(root)),
                                  episode_sha256=digest(dest), initial_geodesic_m=float(d)))
            print('GENERATED', scene['name'], 'seed', seed, flush=True)
        finally:
            sim.close()
    write(root / 'scenes.json', scenes)
    write(root / 'cases.json', cases)
    write(root / 'provenance.json', dict(mode='generated', seed_base=args.seed_base,
                                       scene_manifest_sha256=digest(args.scene_manifest)))
    validate(root)
