import cProfile,pstats,time
from eventwm_graph.stage2_segment.api import SegmentPredictor
from eventwm_graph.stage2_segment.benchmark import inputs
from eventwm_graph.stage2_segment.common import OUT
if (OUT/'prepare_cuda_cold.pstats').exists():raise FileExistsError('preserve previous component profile')
m=SegmentPredictor(OUT/'training/pilot_ray/last.pt',execution='optimized',future_format='native',device='cuda:1')
e=inputs('single')['dense'][0]
m.predict(partial_graph=e['graph'],trajectory=e['trajectory'])
p=cProfile.Profile();p.enable()
for i in range(20):m.clear_cache();m.predict(partial_graph=e['graph'],trajectory=e['trajectory'])
p.disable();p.dump_stats(str(OUT/'prepare_cuda_cold.pstats'))
with (OUT/'logs/prepare_cuda_cold_cprofile.log').open('w') as f:pstats.Stats(p,stream=f).strip_dirs().sort_stats('cumtime').print_stats(38)
