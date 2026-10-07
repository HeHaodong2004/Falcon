"""Imports, versions and an actual CUDA operation for the deployed environment."""

import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import habitat
import habitat_baselines
import habitat_sim
import falcon
import torch


print("Python:", sys.version.split()[0], "executable:", sys.executable)
for package in ["torch", "torchvision", "numpy", "habitat-lab", "habitat-baselines",
                "moviepy", "gym", "hydra-core", "omegaconf", "protobuf", "tensorboard"]:
    print(f"{package}: {version(package)}")
print("Habitat-Sim:", habitat_sim.__version__, "Bullet:", habitat_sim.built_with_bullet)
print("Habitat:", habitat.__file__)
print("Baselines:", habitat_baselines.__file__)
print("FALCON:", falcon.__file__)
print("CUDA runtime:", torch.version.cuda, "available:", torch.cuda.is_available(),
      "visible devices:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA access is required; use a host terminal"
for index in range(torch.cuda.device_count()):
    print("GPU", index, torch.cuda.get_device_name(index))
assert (torch.ones(4, device="cuda:0") * 2).sum().item() == 8
print("PASS: imports and CUDA tensor operation")
