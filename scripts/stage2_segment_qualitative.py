"""Fixed examples of recorded legal increments and actual model predictions.

Run only after the sustained service process has exited. Selection is frozen
by benchmark ordering, never by model quality. No hidden geometry is rendered.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LogNorm
from matplotlib.patches import Patch
from eventwm_graph.stage2_segment.common import BASE,OUT,dump,sha,snapshot


def main():
    if not (OUT/'stability_release_dede_v1/COMPLETE.json').exists():
        raise RuntimeError('Do not overlap qualitative model inference with sustained service')
    protocol=OUT/'qualitative_protocol.json';config=json.loads(protocol.read_text())
    assert sha(OUT/'benchmark_inputs/manifest.json')==config['benchmark_input_manifest_sha256']
    from eventwm_graph.stage2_segment.benchmark import inputs,BaselineComplete
    from eventwm_graph.stage2_segment.api import SegmentPredictor
    from eventwm_graph.stage2_segment.data import replay
    from eventwm_graph.stage2.data import source_rows
    from eventwm_graph.stage2_segment.evaluate import score,rates
    from eventwm_graph.stage2_segment.common import CHECKPOINT
    workload=inputs('single');wanted=[(label,workload[label][0]) for label in config['cohorts']]
    rows,source=source_rows('dev');lookup={r['root_id']:r for r in rows};examples={}
    for root_id in {e['root_id'] for _,e in wanted}:
        targets={(e['start_node'],e['end_node']) for _,e in wanted if e['root_id']==root_id}
        for ex in replay(lookup[root_id],'mixed'):
            if 'audit' not in ex and (ex['start_node'],ex['end_node']) in targets:
                examples[(root_id,ex['start_node'],ex['end_node'])]=ex
    models=dict(original=BaselineComplete(CHECKPOINT),final=SegmentPredictor(OUT/'training/final/last.pt',execution='optimized',future_format='native',device='cuda:1'))
    dest=BASE/'docs/l2/stage2_segment_20261024/figures/qualitative_v1';dest.mkdir(exist_ok=False)
    fig,axes=plt.subplots(3,4,figsize=(17,12));records=[]
    palette=ListedColormap(['#ffffff','#dfe5e8','#5a6876','#007f82','#d95d39'])
    for row,(label,frozen) in enumerate(wanted):
        ex=examples[(frozen['root_id'],frozen['start_node'],frozen['end_node'])]
        assert ex['graph']==frozen['graph'];np.testing.assert_array_equal(ex['trajectory'],frozen['trajectory'])
        outputs={name:model.predict(partial_graph=ex['graph'],trajectory=ex['trajectory']) for name,model in models.items()}
        summaries={name:score(ex,delta) for name,delta in outputs.items()}
        current=ex['graph'];res=current['resolution'];path=np.asarray(ex['trajectory'])
        coordinates=[tuple(c['cell']) for c in current['cells']]+list(ex['target'])
        for delta in outputs.values():coordinates += [tuple(u['cell']) for u in delta['static_updates']]
        coordinates+=list(map(tuple,np.floor(path[:,[1,3]]/res).astype(int)))
        xy=np.array(coordinates);pad=max(2,int(np.ceil(.6/res)));lo=xy.min(0)-pad;hi=xy.max(0)+pad+1;shape=hi-lo
        background=np.zeros(tuple(shape),np.uint8)
        for c in current['cells']:
            i,j=np.array(c['cell'])-lo;background[i,j]=c['state']
        panels=[];truth=background.copy()
        for key,(state,height) in ex['target'].items():
            i,j=np.array(key)-lo;truth[i,j]=state+2
        panels.append(truth)
        for name in models:
            image=background.copy()
            for u in outputs[name]['static_updates']:
                i,j=np.array(u['cell'])-lo;image[i,j]=u['state']+2
            panels.append(image)
        extent=(lo[0]*res,hi[0]*res,lo[1]*res,hi[1]*res)
        for col,image in enumerate(panels):axes[row,col].imshow(image.T,origin='lower',extent=extent,cmap=palette,vmin=0,vmax=4,interpolation='nearest')
        ax=axes[row,3];ax.imshow(background.T,origin='lower',extent=extent,cmap=palette,vmin=0,vmax=4,interpolation='nearest')
        probability=np.full(tuple(shape),np.nan,np.float32);delta=outputs['final'];ij=delta['candidate_cells']-lo
        inside=np.all((ij>=0)&(ij<shape),axis=1);at=ij[inside];probability[at[:,0],at[:,1]]=delta['probabilities'][inside,1:].sum(1)
        heat=ax.imshow(probability.T,origin='lower',extent=extent,cmap='magma',norm=LogNorm(vmin=1e-4,vmax=1),interpolation='nearest')
        for col,ax in enumerate(axes[row]):
            ax.plot(path[:,1],path[:,3],'k.-',lw=.7,ms=2);ax.set_aspect('equal');ax.set_xlabel('x (m)')
            if col==0:ax.set_ylabel('z (m)')
            ax.set_title(('Recorded new observations (projected)','Original prediction','Final prediction','Final P(new observation)')[col])
        info=dict(cohort=label,root_id=frozen['root_id'],scene_id=frozen['scene_id'],start_node=frozen['start_node'],end_node=frozen['end_node'],
            duration=float(path[-1,0]-path[0,0]),poses=len(path),new_cells=len(ex['target']),known_corrections_excluded=ex['known_terminal_differences'],
            models={name:dict(**rates(v['confusion']),logloss=v['scores'][0],brier=v['scores'][1]) for name,v in summaries.items()})
        records.append(info);axes[row,0].text(0,-.23,f"{label}: {info['duration']:.3f}s, {len(path)} saved poses; {len(ex['target'])} true new cells",transform=axes[row,0].transAxes,fontsize=9)
    fig.colorbar(heat,ax=axes[:,3],fraction=.025,pad=.02,label='Probability (display clipped below 1e-4)')
    fig.legend(handles=[Patch(color=palette(i),label=label) for i,label in ((1,'Initially known FREE'),(2,'Initially known BLOCKED'),(3,'New FREE'),(4,'New BLOCKED'))],loc='lower center',ncol=4)
    fig.suptitle('First frozen example in each duration cohort; no quality-based selection\nOnly legal map observations. Shared viewport shows all known and revealed cells; scores use the full candidate domain.',fontsize=12)
    fig.subplots_adjust(top=.91,bottom=.08,hspace=.44,wspace=.25)
    fig.savefig(dest/'examples.png',dpi=180,bbox_inches='tight');fig.savefig(dest/'examples.pdf',bbox_inches='tight');plt.close(fig)
    dump(dest/'manifest.json',dict(protocol_sha256=sha(protocol),source=snapshot(),source_manifest_sha256=sha(source),examples=records,
        files={p.name:sha(p) for p in dest.iterdir() if p.suffix in ('.png','.pdf')},
        scope='Fixed illustrative examples; recorded new-observation projection, initially known corrections excluded; not a new model-selection metric'))


if __name__=='__main__':main()
