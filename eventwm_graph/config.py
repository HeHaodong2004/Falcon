"""Versioned, explicit geometry and construction contracts (SI units)."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path

SCHEMA_VERSION = 1
ALGORITHM_VERSION = "native-regions-9"
REPO = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scene_fingerprint(path):
    """Hash scene content and direct stage-config geometry dependencies."""
    path = Path(path).resolve()
    if path.name.endswith(".scene_instance.json"):
        raise ValueError("Scene-instance dependency resolution is not implemented; use a static stage/GLB")
    if path.name.endswith(".stage_config.json"):
        config = read_json(path)
        assets = {}
        for key in ("render_asset", "collision_asset", "semantic_asset"):
            if key in config and config[key] not in ("NONE", ""):
                asset = (path.parent / config[key]).resolve()
                assets[key] = file_digest(asset)
        return digest({"config": file_digest(path), "assets": assets})
    return file_digest(path)


def robot_asset_fingerprint(robot):
    """Full static collision assets and the installed reference-posture code."""
    import xml.etree.ElementTree as ET
    path = (REPO / robot.robot_urdf).resolve()
    tree = ET.parse(path)
    assets = {}
    for mesh in tree.findall(".//collision/geometry/mesh"):
        name = mesh.attrib["filename"]
        assets[name] = file_digest((path.parent / name).resolve())
    pose = REPO / "habitat-lab/habitat/articulated_agents/robots/spot_robot.py"
    return digest({"urdf": file_digest(path), "collision_meshes": assets,
                   "reference_posture_adapter": file_digest(pose) if "hab_spot_arm" in str(path) else robot.geometry_id})


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, value):
    """Atomic replacement; readers never see a partially written artifact."""
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        name = f.name
    os.replace(name, path)


@dataclass(frozen=True)
class RobotConfig:
    nav_radius: float = 0.25
    nav_height: float = 1.41
    max_climb: float = 0.2
    max_slope: float = 45.0
    collision_reference: float = 0.0  # Required: measured, never an implicit threshold.
    geometry_id: str = ""
    robot_outer_radius: float = 0.0
    robot_urdf: str = "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
    safety_margin: float = 0.03
    max_speed: float = 0.7
    max_yaw_rate: float = 1.5

    def __post_init__(self):
        for k in ("nav_radius", "nav_height", "max_climb", "max_slope",
                  "collision_reference", "max_speed", "max_yaw_rate"):
            if not math.isfinite(getattr(self, k)) or getattr(self, k) <= 0:
                raise ValueError(f"{k} must be positive and finite")
        if not math.isfinite(self.safety_margin) or self.safety_margin < 0:
            raise ValueError("Invalid safety margin")

    @property
    def navigation(self):
        return {k: getattr(self, k) for k in
                ("nav_radius", "nav_height", "max_climb", "max_slope")}


@dataclass(frozen=True)
class BuildConfig:
    region_radius: float = 2.5
    surface_max_edge: float = 1.5
    portal_clearance: float = 0.45
    portal_ratio: float = 0.7
    turn_angle: float = 55.0
    floor_band: float = 1.5
    max_surface_triangles: int = 150000
    clearance_search: float = 10.0
    edge_sample_step: float = 0.15

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Construction parameters must be positive and finite")


@dataclass(frozen=True)
class LocalConfig:
    k: float = 1 / 3
    max_cached_cells: int = 2048
    max_patch_cells: int = 50000

    def __post_init__(self):
        if not math.isfinite(self.k) or self.k <= 0:
            raise ValueError("k must be positive and finite")
        if min(self.max_cached_cells, self.max_patch_cells) <= 0:
            raise ValueError("Local cache/patch limits must be positive")


@dataclass(frozen=True)
class FloorContext:
    island: int = -1
    floor_id: int = -1
    max_height_delta: float = 0.65


def load_robot_config(path):
    return RobotConfig(**read_json(path))
