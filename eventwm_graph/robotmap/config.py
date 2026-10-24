"""Robot configuration-space mapping contracts, independent of human scale."""
from dataclasses import asdict, dataclass
import math

ROBOT_SCHEMA = 2
ROBOT_ALGORITHM = "body-roadmap-5"
TRAVERSABLE = "TRAVERSABLE"
NON_TRAVERSABLE = "NON_TRAVERSABLE"
UNKNOWN = "UNKNOWN"
VALIDATION_LEVEL = "native_body_support_sampled_reference"


@dataclass(frozen=True)
class TraversabilityConfig:
    surface_max_edge: float = 0.9
    heading_count: int = 16
    spatial_step: float = 0.01
    angular_step: float = 0.025
    lateral_numerical_tolerance: float = 0.00001
    connection_radius: float = 1.15
    max_neighbors: int = 10
    region_radius: float = 2.0
    spanner_stretch: float = 1.15
    max_sites: int = 15000
    adaptive_sites: bool = True
    support_ray_lift: float = 0.25
    support_quantization_allowance: float = 0.025
    support_probe_half_span: float = 0.05
    support_probe_height_margin: float = 0.02
    endpoint_radius: float = 1.25
    endpoint_neighbors: int = 18
    validation_level: str = VALIDATION_LEVEL

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{key} must be positive and finite")
        if self.heading_count < 4 or self.heading_count % 2:
            raise ValueError("heading_count must be even and at least four")
        if self.validation_level != VALIDATION_LEVEL:
            raise ValueError("Unsupported motion contract")
        if self.spanner_stretch < 1:
            raise ValueError("spanner_stretch must be at least one")


def angle_delta(a, b):
    return math.atan2(math.sin(b-a), math.cos(b-a))


class NoVerifiedRoute(ValueError):
    def __init__(self, reason, status=UNKNOWN, **details):
        self.details = {"reason": reason, "robot_feasibility": status,
                        "executable": False, **details}
        super().__init__(reason)
