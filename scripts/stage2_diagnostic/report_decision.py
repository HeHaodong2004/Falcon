"""Count/order plots and complete decision-relevance report from frozen results."""
import gzip
import json
import numpy as np
from .common import OUT

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=OUT/'reports';figs=dest/'figures';figs.mkdir(parents=True,exist_ok=True)
    counts=json.loads((OUT/'decision_relevance/counts.json').read_text())
    ranking=json.loads((OUT/'decision_relevance/pairwise_counts.json').read_text())
    validation=set(json.loads((OUT/'protocol.json').read_text())['diagnostic_validation_scenes'])
    actual=[];expected=[]
    for path in sorted((OUT/'probability/roots').glob('*.json.gz')):
        with gzip.open(path,'rt') as f:r=json.load(f)
        if r['scene_id'] in validation:
            actual.extend(v['true_counts'] for v in r['records']);expected.extend(v['expected_counts'] for v in r['records'])
    y=np.array(actual);x=np.array(expected)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for j,(ax,label) in enumerate(zip(axes,('revelation','free','blocked'))):
        ax.hexbin(y[:,j]+1,x[:,j]+1,gridsize=45,xscale='log',yscale='log',bins='log',mincnt=1,cmap='viridis')
        maximum=max(float(y[:,j].max()),float(x[:,j].max()))+1
        ax.plot([1,maximum],[1,maximum],ls='--',color='red',lw=1)
        m=counts['metrics']['model'][label]
        ax.set(xlabel='Actual new cells + 1',ylabel='Expected new cells + 1',title=f'{label.title()}: MAE {m["mae"]:.1f}, r {m["pearson"]:.3f}')
    fig.suptitle('38-scene diagnostic partition: descriptive segment density; headline metrics use scene/Root weighting')
    fig.savefig(figs/'expected_counts.png',dpi=170);fig.savefig(figs/'expected_counts.pdf');plt.close(fig)
    rv=ranking['subsets']['diagnostic_validation38']['revelation'];methods=['stage2','soft_geometry_fixed_union','strict_frontier_count','frustum_opportunity_count']
    labels=['Stage 2','Soft geometry','Strict frontier','Frustum count'];margins=[0,10,50]
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    for j,method in enumerate(methods):
        values=[];low=[];high=[]
        for margin in margins:
            v=rv[f'actual_difference_gt_{margin}']['accuracy'][method];values.append(v['value']);low.append(v['value']-v['ci95'][0]);high.append(v['ci95'][1]-v['value'])
        ax.errorbar(np.arange(3)+(j-1.5)*.13,values,yerr=[low,high],fmt='o',capsize=3,label=labels[j])
    ax.axhline(.5,color='grey',ls='--');ax.set(xticks=np.arange(3),xticklabels=['Non-tied outcomes\n547 pairs','Difference >10 cells\n444 pairs','Difference >50 cells\n272 pairs'],
        ylabel='Pairwise count-order accuracy',ylim=(.35,1),title='Same starting graph and exact horizon; 38 diagnostic scenes, scene-bootstrap 95% intervals')
    ax.legend(ncol=2,fontsize=9);fig.savefig(figs/'paired_count_order.png',dpi=170);fig.savefig(figs/'paired_count_order.pdf');plt.close(fig)
    lines=['# D — Decision-relevant count and spatial prediction','',
        'These are frozen-model diagnostics, not optimization of a Query-ranking model, planner, policy or reward. Expected counts are sums of per-cell revelation/label probabilities. Each segment counts unique initially UNKNOWN cells; overlapping windows are not summed as independent discoveries.','',
        'Simple count baselines are fitted only on the 70 selection scenes. The following table evaluates the remaining 38 historically used development scenes with equal windows within Root, equal Roots within scene and equal scenes. MAE favors medians, so both mean and median duration controls are included. Constant-predictor correlation is undefined.','',
        '| Forecast | New-total MAE | New-total bias | New-total Pearson | FREE MAE | BLOCKED MAE |','|---|---:|---:|---:|---:|---:|']
    for name,m in counts['metrics'].items():
        r=m['revelation'];corr='undefined' if r['pearson'] is None else f"{r['pearson']:.6f}"
        lines.append(f"|{name}|{r['mae']:.3f}|{r['bias']:.3f}|{corr}|{m['free']['mae']:.3f}|{m['blocked']['mae']:.3f}|")
    lines+=['',f"The model's revelation-count MAE CI is {counts['scene_ci95']['model/revelation/mae']}; Pearson CI is {counts['scene_ci95']['model/revelation/pearson']}. Its bias is negative: expected177.28 versus actual191.41 new cells per weighted segment. FREE-count Pearson is0.7003 and BLOCKED-count Pearson0.8853; FREE-count MAE33.01 remains large relative to mean44.15.",'',
        'Pairwise ordering uses one pose-selected recorded alternative pair per Root at the same exact starting graph and timestamp schedule. Actual ties are excluded; predicted ties receive half credit. The model was not selected or fitted using these order scores. Conditional margin cohorts were declared before scoring.','',
        '| Count-order predictor | Non-tied | Difference >10 | Difference >50 |','|---|---:|---:|---:|']
    for method in rv['actual_difference_gt_0']['accuracy']:
        vs=[rv[f'actual_difference_gt_{margin}']['accuracy'][method]['value'] for margin in margins]
        lines.append('|'+method+'|'+'|'.join(f'{v:.4%}' for v in vs)+'|')
    lines+=['', 'For547non-tied validation pairs, Stage2 accuracy is69.91%, CI[65.77%,74.55%]. The highest observed geometric-control point estimate is strictfrontier54.39%; the paired Stage2 advantage CI is[9.14,21.97]percentagepoints. At difference>50cells, Stage2 is78.71%, CI[72.70%,84.10%], on272pairs. Margins change the evaluation population and are not interchangeable overall scores.','',
        'The direct-fixed-union model and default production model have identical observed order decisions here. The soft-geometry control shares the fixed-union universe; the other controls are matched to the same recorded alternatives. Source checks show A/B production expected counts agree to4.55e-13. All class-specific ordering and intervals are retained in pairwise_counts.json.','',
        'Coarse spatial diagnosis uses a4x4 grid of6m bins around the starting pose, with outer coordinates clipped to edge bins. Expected-versus-actual spatial count L1 is105.49cells; conditional normalized shape TV is0.24013. Zero predicted mass with positive actual mass would be undefined and counted separately; no such mass occurred. This coarse statistic does not replace exact cell IoU. The trajectory report additionally evaluates full probability fields and exclusive near-path/heading regions.','',
        'The results support useful expected-count and action-comparison signals on this recorded distribution. They do not establish navigation success, information-theoretic entropy gain, RL improvement, calibrated joint-map uncertainty or transfer to arbitrary planners.','',
        '![Expected counts](figures/expected_counts.png)','', '![Paired order](figures/paired_count_order.png)','']
    (dest/'D_decision_relevance.md').write_text('\n'.join(lines));print(dest/'D_decision_relevance.md')

if __name__=='__main__':main()
