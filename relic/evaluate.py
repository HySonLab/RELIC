"""Held-out Ising and MIS experiments with raw solutions and exact small oracles."""
import argparse
import csv
import json
import time
from pathlib import Path
import networkx as nx
import numpy as np
import torch
from .baselines import compression_baseline, local_search, pignn_mis
from .data import TOPOLOGIES, random_ising, mis_ising, repair_independent_set, greedy_independent_set
from .ising import exact_solve
from .policy import load_policy, rollout
from .train import seed_everything


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def policy_solve(policy, problem, samples=1):
    if samples == 1:
        r = rollout(policy, problem, greedy=True)
    else:
        r = min((rollout(policy, problem) for _ in range(samples)), key=lambda r: r.energy)
    return r.spins, r.energy


def evaluate(root, instances=3, samples=8, pignn_steps=300):
    root = Path(root)
    output = root / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    models = []
    for seed in range(3):
        trained, _ = load_policy(root / f'seed{seed}' / 'best.pt')
        initial, _ = load_policy(root / f'seed{seed}' / 'initial.pt')
        models.append((seed, trained, initial))
    rows, solutions, arrays = [], [], {}
    started = time.perf_counter()
    for n in [12, 25, 50, 100]:
        for ti, topology in enumerate(TOPOLOGIES):
            for instance in range(instances):
                data_seed = 2000000 + n*1000 + ti*100 + instance
                problem = random_ising(n, topology, data_seed)
                key = f'ising_{topology}_{n}_{instance}'
                arrays[key+'_J'], arrays[key+'_h'] = problem.J, problem.h
                exact = exact_solve(problem)[0] if n <= 12 else None
                for seed, trained, initial in models:
                    methods = {
                        'relic_greedy': lambda: policy_solve(trained, problem),
                        f'relic_best{samples}': lambda: policy_solve(trained, problem, samples),
                        'untrained_greedy': lambda: policy_solve(initial, problem),
                        'random_compression': lambda: compression_baseline(problem, data_seed+seed, False),
                        'greedy_contraction': lambda: compression_baseline(problem),
                        'local_search_8': lambda: local_search(problem, data_seed+seed, 8),
                    }
                    for method, solve in methods.items():
                        torch.manual_seed(data_seed + seed)
                        start = time.perf_counter(); spins, energy = solve(); seconds = time.perf_counter()-start
                        assert np.isclose(problem.energy(spins), energy)
                        rows.append({'instance': key, 'topology': topology, 'n': n, 'seed': seed,
                                     'method': method, 'energy': energy, 'normalized_energy': energy/problem.scale(),
                                     'exact_energy': exact if exact is not None else '',
                                     'absolute_gap': max(0., energy-exact) if exact is not None else '',
                                     'normalized_gap': max(0., energy-exact)/problem.scale() if exact is not None else '',
                                     'optimal': int(np.isclose(energy, exact, atol=1e-7)) if exact is not None else '',
                                     'seconds': seconds})
                        solutions.append({'instance': key, 'seed': seed, 'method': method, 'spins': spins.tolist()})
        print(f'Ising n={n} completed after {time.perf_counter()-started:.1f}s', flush=True)
    write_csv(output/'ising.csv', rows)
    mis_rows = []
    for n in [16, 50, 100]:
        for degree in [3, 5, 7, 9]:
            for instance in range(instances):
                data_seed = 3000000 + n*1000 + degree*100 + instance
                graph = nx.random_regular_graph(degree, n, seed=data_seed)
                problem = mis_ising(graph)
                key = f'mis_d{degree}_{n}_{instance}'
                arrays[key+'_J'], arrays[key+'_h'] = problem.J, problem.h
                arrays[key+'_offset'] = np.array(problem.offset)
                optimum = -exact_solve(problem)[0] if n <= 16 else None
                for seed, trained, initial in models:
                    methods = {
                        'relic_greedy': lambda: np.flatnonzero(policy_solve(trained, problem)[0]>0).tolist(),
                        f'relic_best{samples}': lambda: np.flatnonzero(policy_solve(trained, problem,samples)[0]>0).tolist(),
                        'untrained_greedy': lambda: np.flatnonzero(policy_solve(initial, problem)[0]>0).tolist(),
                        'greedy_contraction': lambda: np.flatnonzero(compression_baseline(problem)[0]>0).tolist(),
                        'local_search_8': lambda: np.flatnonzero(local_search(problem,data_seed+seed,8)[0]>0).tolist(),
                        'greedy_mis': lambda: greedy_independent_set(graph),
                        'pignn_reference': lambda: pignn_mis(graph,data_seed+seed,pignn_steps)[0],
                    }
                    for method, solve in methods.items():
                        torch.manual_seed(data_seed+seed)
                        start = time.perf_counter(); selected = solve(); solve_seconds = time.perf_counter()-start
                        start = time.perf_counter(); repaired = repair_independent_set(graph,selected); repair_seconds=time.perf_counter()-start
                        violations = graph.subgraph(selected).number_of_edges()
                        assert graph.subgraph(repaired).number_of_edges() == 0
                        spins = np.full(n,-1); spins[selected]=1
                        mis_rows.append({'instance':key,'degree':degree,'n':n,'seed':seed,'method':method,
                                         'raw_size':len(selected),'violations':violations,'raw_valid':int(violations==0),
                                         'repaired_size':len(repaired),'repaired_valid':1,'raw_energy':problem.energy(spins),
                                         'exact_mis_size':optimum if optimum is not None else '',
                                         'size_ratio':len(repaired)/optimum if optimum is not None else '',
                                         'solve_seconds':solve_seconds,'repair_seconds':repair_seconds,
                                         'seconds':solve_seconds+repair_seconds})
                        solutions.append({'instance':key,'seed':seed,'method':method,'selected':selected,'repaired':repaired})
        print(f'MIS n={n} completed after {time.perf_counter()-started:.1f}s', flush=True)
    write_csv(output/'mis.csv',mis_rows)
    np.savez_compressed(output/'instances.npz',**arrays)
    (output/'solutions.json').write_text(json.dumps(solutions,separators=(',',':'))+'\n')
    (output/'protocol.json').write_text(json.dumps({'instances_per_configuration':instances,
        'training_seeds':[0,1,2],'ising_sizes':[12,25,50,100],'mis_sizes':[16,50,100],
        'mis_degrees':[3,5,7,9],'stochastic_samples':samples,'pignn_steps':pignn_steps,
        'penalty':2.,'seconds':time.perf_counter()-started,
        'timing_note':'Single CPU thread; includes Python graph manipulation; excludes model loading and RELIC training. PI-GNN includes per-instance training. All times measured while other desktop activity may occur.'},indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results')
    parser.add_argument('--instances',type=int,default=3)
    parser.add_argument('--samples',type=int,default=8)
    parser.add_argument('--pignn-steps',type=int,default=300)
    args=parser.parse_args()
    evaluate(args.results,args.instances,args.samples,args.pignn_steps)
