"""Plot completed conditional-density coverage receipts; performs no inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json


def support(receipt):
    return {(r['root_id'],r['scene'],r['window'],r['horizon'],r['stratum'],r['agents'])
            for r in receipt['records']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',action='append',required=True,help='label=receipt.json')
    parser.add_argument('--population',choices=('origin','all_windows'),default='all_windows')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output)
    if any((out/name).exists() for name in ('result.json','conditional_calibration.png','conditional_calibration.pdf')):
        raise FileExistsError('Existing figure artifact; use a new output directory')
    paths=dict(value.split('=',1) for value in args.model);receipts={name:read(path) for name,path in paths.items()}
    reference=next(iter(receipts.values()));nominal=reference['nominal_finite_rank_coverage']
    for receipt in receipts.values():
        if receipt.get('status')!='conditional_coverage_complete':raise ValueError('Completed coverage receipt required')
        if (receipt['nominal_finite_rank_coverage']!=nominal or receipt['draws']!=reference['draws'] or
            receipt['manifest_sha256']!=reference['manifest_sha256'] or support(receipt)!=support(reference)):
            raise ValueError('Coverage protocol or evaluation support differs')
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    levels=(.1,.25,.5,.75,.9);ages=('age_lt025','age_025_to1','age_1_to2','age_2_to3','age_ge3')
    visibility=('all','visible','lost');colors=plt.get_cmap('tab10')
    for row,horizon in enumerate(('1.0','3.0')):
        root_counts={}
        for index,(name,receipt) in enumerate(receipts.items()):
            summary=receipt['summary'][args.population][horizon];color=colors(index)
            all_rows=[summary['all']['coverage'+str(round(p*100))] for p in levels]
            x=np.array([nominal[str(p)] for p in levels]);y=np.array([r['mean'] for r in all_rows])
            axes[row,0].plot(x,y,'o-',label=name,color=color)
            axes[row,0].fill_between(x,[r['low'] for r in all_rows],[r['high'] for r in all_rows],alpha=.1,color=color)
            for column,groups in ((1,ages),(2,visibility)):
                values=[summary[g]['coverage90'] for g in groups];root_counts[column]=[v['roots'] for v in values]
                if any(v['mean'] is None for v in values):raise ValueError('Plot requires supported declared cohorts')
                mean=np.array([v['mean'] for v in values]);low=np.array([v['low'] for v in values]);high=np.array([v['high'] for v in values])
                offset=(index-(len(receipts)-1)/2)*.12
                axes[row,column].errorbar(np.arange(len(groups))+offset,mean,yerr=np.stack([mean-low,high-mean]),
                    fmt='o',capsize=2,color=color,label=name)
        axes[row,0].plot([0,1],[0,1],'k--',alpha=.5);axes[row,0].set(xlim=(0,1),xlabel='Nominal finite-rank coverage')
        for column,labels in ((1,('<.25 s','.25–1 s','1–2 s','2–3 s','≥3 s')),(2,('All','Visible','Lost'))):
            axes[row,column].axhline(nominal['0.9'],linestyle='--',color='black',alpha=.5)
            axes[row,column].set_xticks(np.arange(len(labels)))
            axes[row,column].set_xticklabels([label+'\n'+str(count)+' roots' for label,count in zip(labels,root_counts[column])])
        for column,title in enumerate(('Joint density coverage','Conditional on observation age','Conditional on visibility')):
            axes[row,column].set(title=f'{horizon}s: {title}',ylim=(0,1.02),ylabel='Observed coverage')
            axes[row,column].grid(alpha=.2)
    axes[0,0].legend(fontsize=9)
    fig.suptitle(f'Joint density-rank calibration — {args.population.replace("_"," ")}\n'
        'Windows averaged within root; shaded bands/error bars: scene-bootstrap 95% intervals. '
        'Dashed conditional target: 58/65.',fontsize=11)
    out.mkdir(parents=True,exist_ok=True);fig.savefig(out/'conditional_calibration.png',dpi=180)
    fig.savefig(out/'conditional_calibration.pdf');plt.close(fig)
    save_json(out/'result.json',dict(status='calibration_plot_complete',population=args.population,
        inputs={name:dict(path=path,sha256=sha256(path)) for name,path in paths.items()},
        source_sha256=sha256(__file__),no_new_inference=True))
    print(json.dumps(dict(output=str(out),models=list(paths))))


if __name__=='__main__':main()
