Optional acceleration uses only triangles reconstructed from supplied past/current measured ranges. It does not query simulator assets or complete scene geometry. Custom intersections retain the existing Moller calculation and original leaf slab gate; tighter hardware acceleration bounds preserve tested ray/feature outputs.

NVIDIA provides the external [OptiX headers](https://github.com/NVIDIA/optix-dev). The pinned version is7.7.0, commit7b5c4e8608b8b4b601729f6240fc3fd53cb36d23. Build dependencies stay under ignored output/deps, including their licenses; they are not distributed in this branch. This path needs a supported NVIDIA driver, CUDA/NVRTC and g++.

```bash
python scripts/stage2_segment_run.py eventwm_graph.stage2_repair.optix_setup
python scripts/stage2_segment_run.py unittest eventwm_graph.stage2_repair.test_optix_triangle_rays eventwm_graph.stage2_repair.test_optix_triangle_api
python scripts/stage2_segment_run.py eventwm_graph.stage2_repair.range_runtime_evaluate --checkpoint CHECKPOINT --spatial --implementation gpu-candidates --name UNIQUE_RUN --device cuda:1
```

The same API accepts current legal graph, supplied Nx5 trajectories, measured ancestor/current range arrays, and observed track history. The sensor extension is explicit and is not an original G-only comparison. Range acquisition and serialization are excluded from the complete Stage2 timer. Both are separate from map preparation, feature extraction, transfers, decoding, sparse increments and normal hypothetical graph materialization, which are included.

The original63-probe certificate targets and full original output cells remain unchanged. Independent synthetic intersections,40000 observed-mesh rays, and five real complete workloads pass the tested exact comparisons. Those tests are not the full accuracy or latency acceptance result. All goals remain unmet at the report date.
