"""Export immutable training curves, including root exposure and optimizer state."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json


def lines(path):
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    # A crash before checkpoint publication can replay logged steps on resume.
    return list({r['step']:r for r in rows}.values())


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='append',required=True,help='NAME=directory')
    p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    # The research launcher creates the operational directory before execution.
    # Preserve completed/partial figures while allowing its logs and lock files.
    if any((out/name).exists() for name in ('result.json','learning_curves.png','learning_curves.pdf')):
        raise FileExistsError('Existing curve artifact; use a new output directory: '+str(out))
    out.mkdir(parents=True,exist_ok=True);fig,axes=plt.subplots(3,2,figsize=(13,12),constrained_layout=True);records={}
    for name,path in (x.split('=',1) for x in a.run):
        directory=Path(path);training=lines(directory/'training.jsonl');validation=lines(directory/'validation.jsonl')
        if not training or not validation:raise ValueError('Curves need executed optimizer and validation steps')
        steps=np.array([r['step'] for r in training]);energy=np.array([r['energy'] for r in training])
        smooth=np.array([energy[max(0,i-15):i+1].mean() for i in range(len(energy))])
        axes[0,0].plot(steps,smooth,label=name);axes[0,1].plot([r['step'] for r in validation],[r['scores']['3.0']['energy'] for r in validation],label=name)
        axes[1,0].plot([r['step'] for r in validation],[r['scores']['3.0']['nll'] for r in validation],label=name)
        axes[1,1].plot(steps,[r['gradient_norm'] for r in training],label=name,alpha=.7)
        axes[2,0].plot(steps,[r['effective_epochs'] for r in training],label=name)
        axes[2,1].plot(steps,[r['effective_modes'] for r in training],label=name)
        final=read(directory/'result.json') if (directory/'result.json').exists() else read(directory/'progress.json')
        records[name]=dict(path=str(directory),complete=(directory/'result.json').exists(),status=final,
            source_hashes={n:sha256(directory/n) for n in ('training.jsonl','validation.jsonl','identity.json')},
            learning_rate=[dict(step=r['step'],lr=r['lr']) for r in training],
            latest_unique_roots=training[-1]['unique_roots'],latest_optimizer_step=training[-1]['step'],
            training_curve=training,validation_curve=validation)
    titles=['Training batch Energy Score (16-log moving mean)','Development 3 s Energy Score',
        'Development 3 s joint NLL / coordinate','Gradient norm before clipping',
        'Uniform-root presentations / fit-root count','Training batch effective scene modes']
    for ax,title in zip(axes.flat,titles):ax.set(title=title,xlabel='Optimizer step');ax.grid(alpha=.2)
    axes[1,1].set_yscale('log');axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    fig.suptitle('Train-internal development curves; training uses random windows and partial support\nTraining/development ordinates describe different sampling populations',fontsize=12)
    fig.savefig(out/'learning_curves.png',dpi=180);fig.savefig(out/'learning_curves.pdf');plt.close(fig)
    save_json(out/'result.json',dict(runs=records,source_sha256=sha256(__file__),
        scope='Exploratory curves. Fixed-step and convergence-budget comparisons are distinct; random-window training is not directly comparable to complete-root-origin development.'))
    print(json.dumps(dict(output=str(out),runs=list(records))))


if __name__=='__main__':main()
