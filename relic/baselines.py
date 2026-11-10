"""Transparent classical controls and a small PI-GNN-style MIS reference."""
import numpy as np
import torch
from torch import nn
from .ising import Compression


def compression_baseline(problem, seed=0, greedy=True):
    env, rng = Compression(problem), np.random.default_rng(seed)
    while not env.done:
        env.finish_free_nodes()
        if env.done:
            break
        edges = env.edges()
        if greedy:
            u, v, w = max(edges, key=lambda e: abs(e[2]))
            direction = -1 if w > 0 else 1
        else:
            u, v, _ = edges[int(rng.integers(len(edges)))]
            direction = int(rng.choice([-1, 1]))
        env.contract(u, v, direction)
    return env.solution()


def local_search(problem, seed=0, restarts=8):
    """Best improving single-spin flips, restarted from seeded random spins."""
    rng = np.random.default_rng(seed)
    best_s, best_e = None, float('inf')
    for _ in range(restarts):
        spins = rng.choice([-1, 1], problem.n)
        while True:
            delta = -2 * spins * (problem.J @ spins + problem.h)
            i = int(delta.argmin())
            if delta[i] >= -1e-10:
                break
            spins[i] *= -1
        energy = problem.energy(spins)
        if energy < best_e:
            best_s, best_e = spins.copy(), energy
    return best_s, best_e


class PIGNN(nn.Module):
    """Two GCN layers, trained separately for each MIS instance (reference only)."""
    def __init__(self, n, width=32):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(n, width) * 0.1)
        self.first, self.second = nn.Linear(width, width), nn.Linear(width, 1)

    def forward(self, normalized_adjacency):
        h = torch.relu(self.first(normalized_adjacency @ self.embedding))
        return torch.sigmoid(self.second(normalized_adjacency @ h)).flatten()


def pignn_mis(graph, seed=0, steps=300, penalty=2.0):
    """Optimize relaxed QUBO, select best discrete candidate across iterations.

    This independently implemented control is NOT the paper's PI-GNN code.
    """
    import networkx as nx
    torch.manual_seed(seed)
    A = torch.tensor(nx.to_numpy_array(graph, nodelist=range(len(graph))), dtype=torch.float32)
    augmented = A + torch.eye(len(graph))
    inverse_sqrt = augmented.sum(1).rsqrt()
    normalized = inverse_sqrt[:, None] * augmented * inverse_sqrt[None, :]
    model = PIGNN(len(graph))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_q, best_x = float('inf'), None
    for _ in range(steps):
        optimizer.zero_grad()
        p = model(normalized)
        loss = -p.sum() + penalty * (p @ A @ p) / 2
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x = (p >= 0.5).float()
            q = float(-x.sum() + penalty * (x @ A @ x) / 2)
            if q < best_q:
                best_q, best_x = q, x.numpy().astype(int)
    return np.flatnonzero(best_x).tolist(), best_q
