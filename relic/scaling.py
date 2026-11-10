"""Supplementary CPU scaling pilot; one graph per topology/size, no exact oracle."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .data import TOPOLOGIES, random_ising
from .policy import load_policy
from .evaluate import policy_solve, write_csv
from .baselines import compression_baseline
from .train import seed_everything


def main(root):
    root=Path(root); output=root/'scaling'; output.mkdir(parents=True,exist_ok=True)
    seed_everything(2026)
    models=[load_policy(root/f'seed{s}'/'best.pt')[0] for s in range(3)]
    rows=[]; solutions=[]; arrays={}
    for n in [100,250,500,1000]:
        for ti,topology in enumerate(TOPOLOGIES):
            p=random_ising(n,topology,4000000+n*10+ti)
            key=f'scaling_{topology}_{n}'; arrays[key+'_J']=p.J; arrays[key+'_h']=p.h
            for seed,model in enumerate(models):
                for method in ['relic_greedy','greedy_contraction']:
                    start=time.perf_counter()
                    s,e=policy_solve(model,p) if method=='relic_greedy' else compression_baseline(p)
                    elapsed=time.perf_counter()-start
                    assert np.isclose(e,p.energy(s))
                    rows.append({'instance':key,'topology':topology,'n':n,'seed':seed,'method':method,
                                 'energy':e,'normalized_energy':e/p.scale(),'seconds':elapsed})
                    solutions.append({'instance':key,'seed':seed,'method':method,'spins':s.tolist()})
        print(f'Completed scaling n={n}',flush=True)
    write_csv(output/'scaling.csv',rows)
    np.savez_compressed(output/'instances.npz',**arrays)
    (output/'solutions.json').write_text(json.dumps(solutions,separators=(',',':'))+'\n')
    lines=['# Supplementary size-generalization pilot','','One graph per topology and size, three trained model seeds. No exact optimum, no MIS experiments, and no confidence-interval claim. Single-thread CPU timings are descriptive; these are different instances from the main evaluation. The models were not retrained for this pilot.','','| Nodes | Method | Mean normalized energy | Mean seconds |','|---|---|---:|---:|---:|']
    for n in [100,250,500,1000]:
        for method in ['relic_greedy','greedy_contraction']:
            selected=[r for r in rows if r['n']==n and r['method']==method]
            lines.append(f"| {n} | {method} | {np.mean([r['normalized_energy'] for r in selected]):.4f} | {np.mean([r['seconds'] for r in selected]):.4f} |")
    lines+=['','This pilot demonstrates executable inference at larger sizes, not reproduction of the paper\'s 4,000-node quality or GPU speedup claims. Greedy contraction is deterministic and repeated across model seeds only for a paired table layout. Raw data and reconstructed spins are included alongside this report.','']
    (output/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--results',default='results')
    main(parser.parse_args().results)
