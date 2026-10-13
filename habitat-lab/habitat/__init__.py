#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The Habitat-Sim 0.3.1 Conda build must load its native bindings before
# Magnum. Reversing this order causes duplicate plugins and a shutdown abort.
import habitat_sim  # noqa: F401

from habitat.config import get_config, read_write
from habitat.core.agent import Agent
from habitat.core.benchmark import Benchmark
from habitat.core.challenge import Challenge
from habitat.core.dataset import Dataset
from habitat.core.embodied_task import EmbodiedTask, Measure, Measurements
from habitat.core.env import Env, RLEnv
from habitat.core.logging import logger
from habitat.core.registry import registry  # noqa: F401
from habitat.core.simulator import Sensor, SensorSuite, SensorTypes, Simulator
from habitat.core.vector_env import ThreadedVectorEnv, VectorEnv
from habitat.datasets import make_dataset
from habitat.version import VERSION as __version__  # noqa: F401
