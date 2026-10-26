"""Explicit implementation selection shared by accuracy and cost runners."""


def predictor_class(spatial=False,proxy=False,radix=False,ray=False,exact_fast=False,device_canvas=False,batched=False,implementation=None):
    if implementation:
        if not spatial or proxy or ray:raise ValueError('explicit spatial acceleration requires original63LOS spatial model')
        if implementation=='optix-leaf':
            from .range_optix import OptixSpatialPredictor
            return OptixSpatialPredictor
        if implementation=='optix-triangle':
            from .range_optix import OptixTriangleSpatialPredictor
            return OptixTriangleSpatialPredictor
        if implementation=='device-pipeline':
            from .device_pipeline import DevicePipelinePredictor
            return DevicePipelinePredictor
        if implementation=='fast-pipeline':
            from .device_pipeline import FastPreparedPipelinePredictor
            return FastPreparedPipelinePredictor
        if implementation=='fast-columns':
            from .device_pipeline import ColumnPreparedPipelinePredictor
            return ColumnPreparedPipelinePredictor
        if implementation=='captured-pipeline':
            from .captured_head import CapturedPipelinePredictor
            return CapturedPipelinePredictor
        if implementation=='fully-device':
            from .gpu_obstacle_distance import FullyDeviceSpatialPredictor
            return FullyDeviceSpatialPredictor
        if implementation=='gpu-candidates':
            from .gpu_candidates import GpuCandidateSpatialPredictor
            return GpuCandidateSpatialPredictor
        raise ValueError('unsupported explicit implementation')
    if ray:
        if proxy or radix or exact_fast or device_canvas or batched:raise ValueError('learned-ray diagnostic excludes measured-geometry variants')
        from .range_ray_api import RayRangePredictor
        return RayRangePredictor
    if device_canvas or batched:
        if not spatial or proxy:raise ValueError('device canvas requires exact spatial model')
        if batched:
            from .range_batched_spatial import BatchedSpatialPredictor
            return BatchedSpatialPredictor
        from .range_device_canvas import DeviceCanvasSpatialPredictor
        return DeviceCanvasSpatialPredictor
    if exact_fast:
        if proxy:raise ValueError('exact acceleration preserves all63LOS probes')
        from .range_exact_api import ExactFastRangePredictor,ExactFastSpatialPredictor
        return ExactFastSpatialPredictor if spatial else ExactFastRangePredictor
    if radix:
        from .range_radix_api import RadixRangePredictor,RadixExactPredictor,RadixSpatialPredictor,RadixExactSpatialPredictor
        return (RadixSpatialPredictor if proxy else RadixExactSpatialPredictor)if spatial else RadixRangePredictor if proxy else RadixExactPredictor
    if proxy:
        from .range_fast_proxy import FastRangePredictor,FastSpatialPredictor
        return FastSpatialPredictor if spatial else FastRangePredictor
    if spatial:
        from .range_spatial_api import SpatialRangePredictor
        return SpatialRangePredictor
    from .range_api import RangeEvidencePredictor
    return RangeEvidencePredictor


def add_exact_options(parser):
    parser.add_argument('--implementation',choices=['optix-leaf','optix-triangle','gpu-candidates','fully-device','device-pipeline','fast-pipeline','fast-columns','captured-pipeline']);parser.add_argument('--exact-fast',action='store_true');parser.add_argument('--device-canvas',action='store_true');parser.add_argument('--batched',action='store_true')
