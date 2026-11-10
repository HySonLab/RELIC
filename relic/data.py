"""Seeded synthetic training graphs and the MIS-to-Ising mapping."""
import networkx as nx
import numpy as np
from .ising import Ising

TOPOLOGIES = ('ER', 'BA', 'WS')


def random_ising(n, topology, seed, degree=4, fields=False):
    if n < 5:
        raise ValueError('Synthetic generators require n >= 5')
    if topology == 'ER':
        graph = nx.erdos_renyi_graph(n, min(degree / (n - 1), 1), seed=seed)
    elif topology == 'BA':
        graph = nx.barabasi_albert_graph(n, max(1, min(degree // 2, n - 1)), seed=seed)
    elif topology == 'WS':
        k = min(degree, n - 1)
        k -= k % 2
        graph = nx.watts_strogatz_graph(n, max(k, 2), 0.3, seed=seed)
    else:
        raise ValueError(f'Unknown topology: {topology}')
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n))
    for u, v in sorted(graph.edges()):
        J[u, v] = J[v, u] = rng.uniform(-5, 5)
    h = rng.uniform(-2, 2, n) if fields else np.zeros(n)
    return Ising(J, h)


def dataset(count, n, seed):
    return [random_ising(n, TOPOLOGIES[i % 3], seed + i) for i in range(count)]


def mis_ising(graph, penalty=2.0):
    """Q(x)=-sum x_i + P sum_edges x_i*x_j, x=(s+1)/2.

    P>1 guarantees that every global minimizer is an independent set.
    Local/approximate solutions can violate constraints and must be reported.
    """
    if penalty <= 1:
        raise ValueError('MIS penalty must exceed 1')
    if graph.is_directed() or nx.number_of_selfloops(graph):
        raise ValueError('MIS requires a simple undirected graph')
    if set(graph) != set(range(len(graph))):
        raise ValueError('Use integer node labels 0,...,n-1')
    A = nx.to_numpy_array(graph, nodelist=range(len(graph)), weight=None)
    return Ising(penalty * A / 4, -0.5 + penalty * A.sum(axis=1) / 4,
                 -len(graph) / 2 + penalty * graph.number_of_edges() / 4)


def repair_independent_set(graph, selected):
    """Delete conflict vertices deterministically; never augment the output."""
    selected = set(map(int, selected))
    while True:
        conflicts = {v: sum(u in selected for u in graph[v]) for v in selected}
        if not conflicts or max(conflicts.values()) == 0:
            return sorted(selected)
        remove = max(selected, key=lambda v: (conflicts[v], graph.degree[v], v))
        selected.remove(remove)


def greedy_independent_set(graph):
    remaining, selected = set(graph), []
    while remaining:
        v = min(remaining, key=lambda u: (sum(w in remaining for w in graph[u]), u))
        selected.append(v)
        remaining.difference_update({v, *graph[v]})
    return sorted(selected)
