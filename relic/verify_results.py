"""Independently recompute published objectives and MIS feasibility from saved spins."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .ising import Ising, exact_solve


def verify(root):
    root=Path(root)
    data=np.load(root/'instances.npz',allow_pickle=False)
    solutions=json.loads((root/'solutions.json').read_text())
    lookup={(s['instance'],str(s['seed']),s['method']):s for s in solutions}
    count=0
    for filename in ['ising.csv','mis.csv']:
        for row in csv.DictReader((root/filename).open()):
            key=row['instance']; solution=lookup[(key,row['seed'],row['method'])]
            offset=float(data[key+'_offset']) if key+'_offset' in data else 0.
            p=Ising(data[key+'_J'],data[key+'_h'],offset)
            if filename=='ising.csv':
                energy=p.energy(solution['spins'])
                assert np.isclose(energy,float(row['energy']))
                assert np.isclose(energy/p.scale(),float(row['normalized_energy']))
            else:
                s=np.full(p.n,-1); s[solution['selected']]=1
                assert np.isclose(p.energy(s),float(row['raw_energy']))
                selected=solution['selected']; repaired=solution['repaired']
                assert len(selected)==len(set(selected))==int(row['raw_size'])
                assert len(repaired)==len(set(repaired))==int(row['repaired_size'])
                assert set(repaired)<=set(selected)
                conflicts=np.count_nonzero(np.triu(p.J[np.ix_(selected,selected)],1))
                assert conflicts==int(row['violations'])
                assert not np.any(p.J[np.ix_(repaired,repaired)])
            count+=1
    # Recompute each distinct exact oracle rather than trusting CSV labels.
    exact_checked=0
    for filename,field in [('ising.csv','exact_energy'),('mis.csv','exact_mis_size')]:
        seen=set()
        for row in csv.DictReader((root/filename).open()):
            key=row['instance']
            if not row[field] or key in seen:
                continue
            seen.add(key)
            offset=float(data[key+'_offset']) if key+'_offset' in data else 0.
            p=Ising(data[key+'_J'],data[key+'_h'],offset)
            optimum=exact_solve(p)[0]
            assert np.isclose(optimum if filename=='ising.csv' else -optimum,float(row[field]))
            exact_checked+=1
    scaling_count=0
    scaling=root.parent/'scaling'
    if (scaling/'scaling.csv').exists():
        scale_data=np.load(scaling/'instances.npz',allow_pickle=False)
        scale_solutions=json.loads((scaling/'solutions.json').read_text())
        scale_lookup={(s['instance'],str(s['seed']),s['method']):s for s in scale_solutions}
        for row in csv.DictReader((scaling/'scaling.csv').open()):
            key=row['instance']; sol=scale_lookup[(key,row['seed'],row['method'])]
            p=Ising(scale_data[key+'_J'],scale_data[key+'_h'])
            energy=p.energy(sol['spins'])
            assert np.isclose(energy,float(row['energy']))
            assert np.isclose(energy/p.scale(),float(row['normalized_energy']))
            scaling_count+=1
    result={'scaling_rows_verified':scaling_count,'solution_rows_verified':count,'exact_instances_recomputed':exact_checked,'status':'passed'}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results/evaluation')
    verify(parser.parse_args().results)
