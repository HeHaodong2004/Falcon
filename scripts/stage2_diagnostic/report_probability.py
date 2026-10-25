"""Standalone probability/threshold research figures and complete CSV tables."""
import csv
import json
from pathlib import Path
import numpy as np
from .common import OUT,THRESHOLDS,EDGES

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((OUT/'probability/summary.json').read_text());head=json.loads((OUT/'probability/conditional_head.json').read_text())
    dest=OUT/'reports';dest.mkdir(parents=True,exist_ok=True);figs=dest/'figures';figs.mkdir(exist_ok=True)
    rows=[]
    for rule,r in summary['rules'].items():
        row={'rule':rule}
        for k,v in r.items():
            if isinstance(v,dict):row.update({k+'_'+m:x for m,x in v.items()})
            else:row[k]=v
        row.update({'pooled_C_'+a+b:summary['pooled_confusions'][rule][i][j] for i,a in enumerate('UFB') for j,b in enumerate('UFB')})
        rows.append(row)
    with open(dest/'thresholds.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    rules=list(summary['rules']);rs=[summary['rules'][r] for r in rules[1:]];base=summary['rules']['argmax']
    fig,ax=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    ax[0].plot(THRESHOLDS,[r['label_aware_delta_iou'] for r in rs],'o-',label='Label-aware IoU')
    ax[0].plot(THRESHOLDS,[r['revelation_iou'] for r in rs],'o-',label='Binary revelation IoU')
    ax[0].axhline(base['label_aware_delta_iou'],color='C0',ls='--',label='Argmax label-aware IoU')
    ax[0].set(xlabel='Revelation threshold',ylabel='IoU',title='Same checkpoint, different hard rule');ax[0].legend(fontsize=8)
    ax[1].plot(THRESHOLDS,[r['revelation_precision'] for r in rs],'o-',label='Revelation precision')
    ax[1].plot(THRESHOLDS,[r['revelation_recall'] for r in rs],'o-',label='Revelation recall')
    ax[1].plot(THRESHOLDS,[r['terminal_label_agreement_on_true_revelations'] for r in rs],'o-',label='Correct-label recall')
    ax[1].set(xlabel='Revelation threshold',ylabel='Fraction',title='Recall comes with false revelations');ax[1].legend(fontsize=8)
    ax[2].stackplot(THRESHOLDS,[r['free']['precision'] for r in rs],
        [r['false_free_unknown_fraction'] for r in rs],[r['false_free_blocked_fraction'] for r in rs],
        labels=['Actually newly FREE','Actually remains UNKNOWN','Actually newly BLOCKED'],colors=['#75b980','#b8bdc8','#dc7771'])
    ax[2].set(xlabel='Revelation threshold',ylabel='Fraction of predicted FREE',ylim=(0,1),title='FREE updates are hypothetical');ax[2].legend(fontsize=8,loc='lower left')
    fig.savefig(figs/'threshold_tradeoff.png',dpi=170);fig.savefig(figs/'threshold_tradeoff.pdf');plt.close(fig)
    with np.load(OUT/'probability/scene_arrays.npz') as z:h=z['histograms'].mean(0)
    fig,ax=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for j,label in enumerate(('UNKNOWN truth','FREE truth','BLOCKED truth')):
        values=h[3,j];ax[0].plot(EDGES[1:],np.cumsum(values)/values.sum(),label=label)
    ax[0].set(xscale='log',xlim=(1e-6,1),xlabel='Predicted P(revelation)',ylabel='Conditional cumulative mass',title='Probability distributions by true label');ax[0].legend(fontsize=8)
    positive=h[3,1:].sum(0)[::-1];negative=h[3,0][::-1];tp=np.cumsum(positive);fp=np.cumsum(negative)
    precision=tp/np.maximum(tp+fp,1e-30);recall=tp/tp[-1]
    ax[1].plot(recall,precision,label=f"Model AP={summary['probability']['revelation_rank']['auprc_ap']:.4f}")
    ax[1].axhline(summary['probability']['revelation_rank']['prevalence'],ls='--',color='grey',label='Weighted prevalence')
    ax[1].set(xlabel='Revelation recall',ylabel='Revelation precision',ylim=(0,1),title='Hierarchically weighted cell-level PR');ax[1].legend(fontsize=8)
    bins=summary['probability']['revelation_calibration']['bins'];x=[b['probability'] for b in bins];y=[b['frequency'] for b in bins]
    ax[2].plot([0,1],[0,1],ls='--',color='grey');ax[2].plot(x,y,'o-');ax[2].set(xlabel='Mean predicted P(revelation)',ylabel='Observed revelation frequency',xlim=(0,1),ylim=(0,1),title='20-bin revelation reliability')
    fig.savefig(figs/'probability_diagnostics.png',dpi=170);fig.savefig(figs/'probability_diagnostics.pdf');plt.close(fig)
    best=summary['rules']['threshold_0.20'];rank=summary['probability']['revelation_rank'];val=summary['diagnostic_scene_validation'];v=val['metrics']['rules'];ci=summary['paired_scene_ci95']['threshold_0.20']
    text=['# A — Probability and decision-rule diagnosis','',
        'Checkpoint unchanged. All47,128windows/2,110Roots/108scenes exactly reproduce the saved original argmax confusion counts and target/domain counts. No protected payloads or future observation inputs.','',
        f"Weighted cell-normalized revelation prevalence={rank['prevalence']:.9f}; AUROC={rank['auroc']:.9f}, AP={rank['auprc_ap']:.9f}. These use window-cell normalized→Root→Scene weighting; the pooled persistent-UNKNOWN fraction99.1805% uses a different denominator. AP prevalence baseline={rank['prevalence']:.9f}. High-resolution histogram bounds are discretization bounds, not statistical confidence intervals.",'',
        '|Rule|Label-awareIoU|BinaryIoU|RevelationP|RevelationR|RevelationF1|Correct-labelR|FREEP|FREER|BLOCKEDP|BLOCKEDR|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rule,r in summary['rules'].items():
        values=[r['label_aware_delta_iou'],r['revelation_iou'],r['revelation_precision'],r['revelation_recall'],r['revelation_f1'],r['terminal_label_agreement_on_true_revelations'],r['free']['precision'],r['free']['recall'],r['blocked']['precision'],r['blocked']['recall']]
        text.append('|'+rule+'|'+'|'.join(f'{x:.6f}' for x in values)+'|')
    text+=['',f"Both full-development maxima are threshold0.20. IoU={best['label_aware_delta_iou']:.9f}; paired95%scene-bootstrap change versusargmax={ci['label_aware_delta_iou_minus_argmax']}. Same-data optimum/CI is exploratory. The70scene-selected rule is also0.20; frozen38scene diagnostic-validationIoU={v['argmax']['label_aware_delta_iou']:.9f}→{v['threshold_0.20']['label_aware_delta_iou']:.9f}. These38scenes were previously evaluated in research; they are not untouched confirmation.",'',
        f"At0.20 FREEprecision={best['free']['precision']:.6f}, predictedFREEactuallyUNKNOWNfraction={best['false_free_unknown_fraction']:.6f}, predictedFREEactuallyBLOCKEDfraction={best['false_free_blocked_fraction']:.6f}. ActualBLOCKED mislabeledFREErate={best['false_free_on_actual_blocked_rate']:.6f}, versus{base['false_free_on_actual_blocked_rate']:.6f} underargmax. UNKNOWN is an unobserved legal target, not proven physicallyBLOCKED. These updates cannot be treated as measured traversability.",'',
        f"Posthoc conditional FREE/BLOCKED classaccuracy on actualrevealedcells={head['conditional_class_accuracy']:.6f};95%sceneCI={head['conditional_class_accuracy_scene_ci95']}. This GTmask is scoring only, never an inference input or achievable oracle forecast.",'',
        'Changing the hard rule changes neither probabilities nor Brier/logloss. Thresholding improves hard-map recall/IoU but introduces many false revelations. At0.20 missedrevelations remain50.71% ofactualrevelations; wrongFREE/BLOCKED labels are9.77%; false revelations are the larger cost. Label classification has not become the dominant error.','',
        'Complete per-classF1, false-free decomposition, false-revelation rates and pooled3x3 confusion counts are in thresholds.csv and probability/summary.json. Full class-conditioned probability quantiles, reliability bins and binning bounds are retained in summary.json.','',
        '![Thresholds](figures/threshold_tradeoff.png)','', '![Probabilities](figures/probability_diagnostics.png)','']
    (dest/'A_probability_thresholds.md').write_text('\n'.join(text))
    print(dest/'A_probability_thresholds.md')

if __name__=='__main__':main()
