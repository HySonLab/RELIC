"""Ising problems and exact, signed graph contraction (no ML dependencies)."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Ising:
    """E(s) = sum_{i<j} J[i,j] s[i] s[j] + h @ s + offset.

    J is symmetric with zero diagonal; each undirected edge is counted once.
    """
    J: np.ndarray
    h: np.ndarray
    offset: float = 0.0

    def __post_init__(self):
        self.J = np.array(self.J, dtype=np.float64, copy=True)
        self.h = np.array(self.h, dtype=np.float64, copy=True)
        n = len(self.h)
        if n < 1 or self.J.shape != (n, n):
            raise ValueError('Expected a nonempty square J and matching h')
        if not np.isfinite(self.J).all() or not np.isfinite(self.h).all() or not np.isfinite(self.offset):
            raise ValueError('All coefficients must be finite')
        if not np.allclose(self.J, self.J.T) or np.any(np.diag(self.J) != 0):
            raise ValueError('J must be symmetric with a zero diagonal')

    @property
    def n(self):
        return len(self.h)

    def energy(self, spins):
        s = np.asarray(spins)
        if s.shape != (self.n,) or not np.isin(s, [-1, 1]).all():
            raise ValueError('Expected one +/-1 spin per node')
        return float(0.5 * s @ self.J @ s + self.h @ s + self.offset)

    def scale(self):
        """Instance-only reward scale, independent of sampled actions."""
        return max(float(np.abs(self.J).sum() / 2 + np.abs(self.h).sum()), 1.0)


def exact_solve(problem, max_n=22, chunk_size=32768):
    """Exhaustive oracle, memory-bounded in chunks; never used for training."""
    if problem.n > max_n:
        raise ValueError(f'Exhaustive search limited to n <= {max_n}')
    best_e, best_s = float('inf'), None
    for start in range(0, 1 << problem.n, chunk_size):
        ids = np.arange(start, min(start + chunk_size, 1 << problem.n), dtype=np.uint64)
        s = 2 * ((ids[:, None] >> np.arange(problem.n, dtype=np.uint64)) & 1).astype(np.float64) - 1
        energies = 0.5 * np.sum((s @ problem.J) * s, axis=1) + s @ problem.h + problem.offset
        k = int(energies.argmin())
        if energies[k] < best_e:
            best_e, best_s = float(energies[k]), s[k].astype(np.int8)
    return best_e, best_s


class Compression:
    """Sparse mutable graph with a reversible contraction trace.

    Node 0 is the auxiliary field node; physical node i is stored at i+1.
    Contracting (u,v,d) enforces s_v = d*s_u. This preserves energies ONLY
    on that restricted subspace; an arbitrary action can exclude the optimum.
    """
    def __init__(self, problem):
        self.problem = problem
        self.adj = {i: {} for i in range(problem.n + 1)}
        self.offset = float(problem.offset)
        self.trace = []
        for i in range(problem.n):
            if problem.h[i] != 0:
                self._set_edge(0, i + 1, problem.h[i])
            for j in range(i + 1, problem.n):
                if problem.J[i, j] != 0:
                    self._set_edge(i + 1, j + 1, problem.J[i, j])

    def _set_edge(self, u, v, w):
        if w == 0:  # Exact cancellation; do not silently drop small nonzero weights.
            self.adj[u].pop(v, None)
            self.adj[v].pop(u, None)
        else:
            self.adj[u][v] = float(w)
            self.adj[v][u] = float(w)

    def edges(self):
        return [(u, v, w) for u in sorted(self.adj)
                for v, w in sorted(self.adj[u].items()) if u < v]

    @property
    def done(self):
        return len(self.adj) == 1

    def contract(self, u, v, d):
        if u == v or u not in self.adj or v not in self.adj or d not in (-1, 1):
            raise ValueError('Invalid contraction')
        # Nonedge contractions are only needed after all residual weights vanish.
        if v not in self.adj[u] and any(self.adj.values()):
            raise ValueError('Policy actions must select an existing edge')
        self.offset += d * self.adj[u].get(v, 0.0)
        for w, weight in list(self.adj[v].items()):
            if w != u:
                self._set_edge(u, w, self.adj[u].get(w, 0.0) + d * weight)
            self.adj[w].pop(v, None)
        del self.adj[v]
        self.trace.append((u, v, int(d)))

    def finish_free_nodes(self):
        """Deterministically join zero-interaction components without policy actions."""
        if any(self.adj.values()):
            return
        while not self.done:
            u, v = sorted(self.adj)[:2]
            self.contract(u, v, 1)

    def lift(self, residual_spins):
        """Lift any residual assignment to all original augmented nodes."""
        spins = dict(residual_spins)
        if set(spins) != set(self.adj) or any(s not in (-1, 1) for s in spins.values()):
            raise ValueError('Residual spin assignment does not match active nodes')
        for u, v, d in reversed(self.trace):
            spins[v] = d * spins[u]
        return np.array([spins[i] for i in range(self.problem.n + 1)], dtype=np.int8)

    def solution(self):
        if not self.done:
            raise ValueError('Compression is not finished')
        spins = self.lift({next(iter(self.adj)): 1})
        # Gauge-fix the auxiliary spin, even if node 0 was previously absorbed.
        physical = spins[1:] * spins[0]
        energy = self.problem.energy(physical)
        if not np.isclose(energy, self.offset, atol=1e-8, rtol=1e-9):
            raise AssertionError(f'Reconstruction mismatch: {energy} != {self.offset}')
        return physical, energy
