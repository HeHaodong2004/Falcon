# Load the 0.3.1 native bindings before modules that import Magnum.
import habitat_sim  # noqa: F401

from . import rearrange_sim_v2
from . import additional_action
from . import additional_sensor
from . import additional_metric
from . import auxiliary_tasks
