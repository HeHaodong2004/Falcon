"""Check the real scene paths referenced by SocialNav episode files."""

import argparse
import gzip
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["hm3d", "mp3d", "both"], default="both")
parser.add_argument("--split", default="val")
parser.add_argument("--scene", help="Content-file stem, e.g. CFVBbU9Rsyb")
parser.add_argument("--list-missing", action="store_true")
args = parser.parse_args()
missing = []

def check_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        missing.append(str(path))
        return False
    return True


for dataset in (["hm3d", "mp3d"] if args.dataset == "both" else [args.dataset]):
    folder = Path("data/datasets/pointnav") / f"social-{dataset}" / args.split
    if not check_file(folder / f"{args.split}.json.gz"):
        continue
    files = sorted((folder / "content").glob(f"{args.scene or '*'}.json.gz"))
    if not files:
        missing.append(str(folder / "content" / f"{args.scene or '*'}.json.gz"))
        continue
    scenes = set()
    episode_count = 0
    for file in files:
        with gzip.open(file, "rt") as stream:
            episodes = json.load(stream)["episodes"]
        episode_count += len(episodes)
        for episode in episodes:
            info = episode.get("info", {})
            assert "human_num" in info, (file, episode["episode_id"])
            for index in range(info["human_num"]):
                assert f"human_{index}_waypoint_0_position" in info, file
            scene = episode["scene_id"]
            prefix = "data/scene_datasets/"
            if scene.startswith(prefix):
                scene = scene[len(prefix):]
            scenes.add(str(Path(prefix) / scene))
    before = len(missing)
    for scene in sorted(scenes):
        check_file(scene)
    print(f"{dataset}/{args.split}: {episode_count} episodes parsed; "
          f"{len(scenes)} scene paths; {len(missing) - before} missing scenes")

check_file("data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf")
check_file("data/robots/spot_data/spot_walking_trajectory.csv")
for human in ["female_0", "female_1", "male_0", "male_1", "female_2", "female_3"]:
    root = Path("data/humanoids/humanoid_data") / human
    check_file(root / f"{human}.urdf")
    check_file(root / f"{human}.glb")
    check_file(root / f"{human}_motion_data_smplx.pkl")
check_file("pretrained_model/falcon_noaux_25.pth")
check_file("pretrained_model/pretrained_habitat3.pth")
if missing:
    print("BLOCKED: missing assets:")
    for path in missing if args.list_missing else missing[:5]:
        print(path)
    if not args.list_missing and len(missing) > 5:
        print(f"... {len(missing) - 5} more; use --list-missing")
    raise SystemExit(2)
print("PASS: selected episodes, scene files, robot/humanoid files and weights exist")
