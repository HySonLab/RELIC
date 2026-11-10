"""Generate figures and descriptive tables from recorded experiment data."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return list(csv.DictReader(Path(path).open()))


def instance_summary(rows, metric):
    """Average policy seeds within each instance, then mean/SEM over instances.

    This avoids treating correlated runs on the same graph as independent data.
    """
    groups=defaultdict(list)
    for r in rows:
        if r[metric]!='':
            groups[r['instance']].append(float(r[metric]))
    values=np.array([np.mean(v) for v in groups.values()])
    return float(values.mean()), float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0., len(values)


def main(root='results'):
    root=Path(root); out=root/'evaluation'; plt.style.use('seaborn-v0_8-whitegrid')
    ising,mis=read(out/'ising.csv'),read(out/'mis.csv')
    fig,ax=plt.subplots(figsize=(7,4))
    for seed in range(3):
        rows=read(root/f'seed{seed}'/'history.csv')
        ax.plot([int(r['epoch']) for r in rows],[float(r['validation_normalized_energy']) for r in rows],label=f'Seed {seed}')
    ax.set(xlabel='Epoch',ylabel='Validation energy / sum of |weights|',title='RELIC training on 25-node graphs (lower is better)')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'training.png',dpi=160); plt.close(fig)
    methods=['relic_greedy','relic_best8','untrained_greedy','greedy_contraction','local_search_8','random_compression']
    fig,ax=plt.subplots(figsize=(8,4.5))
    for method in methods:
        sizes=sorted(set(int(r['n']) for r in ising)); means=[]; errors=[]
        for n in sizes:
            mean,sem,_=instance_summary([r for r in ising if r['method']==method and int(r['n'])==n],'normalized_energy')
            means.append(mean); errors.append(sem)
        ax.errorbar(sizes,means,yerr=errors,marker='o',label=method)
    ax.set(xlabel='Number of spins',ylabel='Normalized energy (lower is better)',title='Held-out Ising problems: ER, BA and WS')
    ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(out/'ising.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in ['relic_greedy','relic_best8','greedy_mis','pignn_reference','local_search_8']:
        degrees=[3,5,7,9]; values=[]; errors=[]; times=[]
        for d in degrees:
            selected=[r for r in mis if r['method']==method and int(r['degree'])==d and int(r['n'])==100]
            mean,sem,_=instance_summary(selected,'repaired_size'); values.append(mean); errors.append(sem)
            times.append(instance_summary(selected,'seconds')[0])
        axes[0].errorbar(degrees,values,yerr=errors,marker='o',label=method)
        axes[1].plot(degrees,times,marker='o',label=method)
    axes[0].set(xlabel='Regular degree',ylabel='Feasible independent set size',title='MIS on 100-node graphs, after deletion repair')
    axes[1].set(xlabel='Regular degree',ylabel='Seconds (log scale)',title='CPU time; PI-GNN includes per-instance training',yscale='log')
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'mis.png',dpi=160); plt.close(fig)
    lines=['# Measured results','',
           'These are desktop reproduction measurements, not the numbers reported in the paper. Lower Ising energy is better; larger feasible MIS size is better.',
           '', '## Training','', '| Seed | Initial validation energy / scale | Best validation energy / scale | Best epoch | Epochs run | Seconds |',
           '|---|---:|---:|---:|---:|---:|']
    for seed in range(3):
        s=json.loads((root/f'seed{seed}'/'summary.json').read_text())
        lines.append(f"| {seed} | {s['initial_validation_normalized_energy']:.4f} | {s['best_validation_normalized_energy']:.4f} | {s['best_epoch']} | {s['epochs_run']} | {s['seconds']:.1f} |")
    lines += ['','![Training](training.png)','','## Exact small Ising comparison (12 spins)','','| Method | Mean energy gap to optimum | Mean normalized gap | Optimum hit fraction |','|---|---:|---:|---:|']
    for method in methods:
        rows=[r for r in ising if r['method']==method and int(r['n'])==12]
        gap,_,_=instance_summary(rows,'absolute_gap'); norm,_,_=instance_summary(rows,'normalized_gap'); hit,_,_=instance_summary(rows,'optimal')
        lines.append(f'| {method} | {gap:.4f} | {norm:.4f} | {hit:.3f} |')
    lines += ['','## Generalization to 100-spin Ising problems','','| Method | Normalized energy (mean ± SEM) | Mean seconds |','|---|---:|---:|']
    for method in methods:
        rows=[r for r in ising if r['method']==method and int(r['n'])==100]
        mean,sem,_=instance_summary(rows,'normalized_energy'); sec,_,_=instance_summary(rows,'seconds')
        lines.append(f'| {method} | {mean:.4f} ± {sem:.4f} | {sec:.4f} |')
    lines += ['','![Ising](ising.png)','','## MIS results at 100 nodes','','| Method | Raw feasible fraction | Mean raw size | Mean repaired size | Mean seconds |','|---|---:|---:|---:|---:|']
    for method in sorted(set(r['method'] for r in mis)):
        rows=[r for r in mis if r['method']==method and int(r['n'])==100]
        valid,_,_=instance_summary(rows,'raw_valid'); raw,_,_=instance_summary(rows,'raw_size'); repaired,_,_=instance_summary(rows,'repaired_size'); sec,_,_=instance_summary(rows,'seconds')
        lines.append(f'| {method} | {valid:.3f} | {raw:.2f} | {repaired:.2f} | {sec:.4f} |')
    lines += ['','![MIS](mis.png)','','## Interpretation and limits','',
              '- Three training seeds share the same fixed training and validation graphs. Test graph seeds are disjoint. No test result selects checkpoints.',
              '- For descriptive SEM, first average model seeds within each graph, then treat graphs as the sampling units. Error bars are one SEM, not 95% confidence intervals. There are only three graphs per topology/size or degree/size configuration.',
              '- The best-of-8 policy uses eight sampled trajectories; greedy RELIC uses one. The greedy trajectory is not included among those samples, so best-of-8 can be worse, as it is in these runs. Runtime reports those unequal budgets explicitly.',
              '- Random compression, greedy contraction, and restarted local search are controls. The PI-GNN reference is an independent small implementation, not the paper baseline or a tuned state of the art.',
              '- MIS repair only deletes vertices. Raw validity is reported separately. Empty independent sets are valid but have zero size.',
              '- The exact oracles are restricted to 12-spin Ising and 16-node MIS problems. Larger-instance quality has no optimality certificate.',
              '- Neither the 4,000-node claims, original dataset, nor RTX 3060 timing comparison is reproduced here. This code recomputes message passing at every contraction; linear total runtime is not guaranteed.',
              '', 'Raw data: [Ising CSV](ising.csv), [MIS CSV](mis.csv), [instances](instances.npz), [solutions](solutions.json), [verification](verification.json), [protocol](protocol.json).','']
    (out/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results')
    main(parser.parse_args().results)
