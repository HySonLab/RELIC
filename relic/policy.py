"""Three-layer joint node/edge message passing and compression rollouts."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .ising import Compression


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


def graph_tensors(env, device='cpu'):
    nodes, edges = sorted(env.adj), env.edges()
    index = {v: i for i, v in enumerate(nodes)}
    node_features = []
    for v in nodes:
        degree = len(env.adj[v])
        node_features.append([degree, 1 / degree if degree else 0, sum(abs(w) for w in env.adj[v].values())])
    endpoints = np.array([(index[u], index[v]) for u, v, _ in edges], dtype=np.int64).reshape(-1, 2)
    edge_features = np.array([(w, abs(w), np.sign(w)) for _, _, w in edges], dtype=np.float32).reshape(-1, 3)
    return (torch.tensor(node_features, dtype=torch.float32, device=device),
            torch.as_tensor(endpoints, device=device), torch.as_tensor(edge_features, device=device), edges)


class MessageLayer(nn.Module):
    def __init__(self, width, update_nodes=True):
        super().__init__()
        self.update_nodes = update_nodes
        self.message = mlp(2 * width, width, width)
        self.node = mlp(2 * width, width, width)
        self.edge = mlp(3 * width, width, width)
        self.node_norm = nn.LayerNorm(width)
        if not update_nodes:
            del self.message, self.node, self.node_norm
        self.edge_norm = nn.LayerNorm(width)

    def forward(self, h, e, endpoints):
        u, v = endpoints.T
        # Both directions contribute; the undirected edge is stored only once.
        next_h = h
        if self.update_nodes:
            targets = torch.cat([u, v])
            neighbors = torch.cat([v, u])
            messages = self.message(torch.cat([h[neighbors], torch.cat([e, e])], dim=1))
            aggregate = torch.zeros_like(h).index_add(0, targets, messages)
            next_h = self.node_norm(self.node(torch.cat([h, aggregate], dim=1)))
        # Symmetric endpoint inputs preserve permutation equivariance.
        next_e = self.edge_norm(self.edge(torch.cat([h[u] + h[v], torch.abs(h[u] - h[v]), e], dim=1)))
        return torch.relu(next_h), torch.relu(next_e)


class Policy(nn.Module):
    def __init__(self, width=32, layers=3):
        super().__init__()
        self.width, self.layers = width, layers
        self.node_input = nn.Linear(3, width)
        self.edge_input = nn.Linear(3, width)
        # The final node update cannot affect the edge-only readout (Eq. 6
        # uses previous-layer nodes), so omit that unused computation.
        self.blocks = nn.ModuleList([MessageLayer(width, i < layers - 1) for i in range(layers)])
        self.readout = mlp(width + 3, width, 2)

    def forward(self, node_features, endpoints, edge_features):
        h = torch.relu(self.node_input(node_features))
        e = torch.relu(self.edge_input(edge_features))
        for block in self.blocks:
            h, e = block(h, e, endpoints)
        # Flattened order: (edge0, merge), (edge0, flip), (edge1, merge), ...
        return self.readout(torch.cat([edge_features, e], dim=1)).reshape(-1)

    def distribution(self, env, epsilon=0.0):
        if not 0 <= epsilon <= 1:
            raise ValueError('epsilon must be in [0,1]')
        x, endpoints, e, edges = graph_tensors(env, next(self.parameters()).device)
        logits = self(x, endpoints, e)
        if not len(logits):
            raise ValueError('No weighted policy actions remain')
        # Differentiable exploratory policy; log probability matches sampling.
        probabilities = (1 - epsilon) * torch.softmax(logits, dim=0) + epsilon / len(logits)
        return torch.distributions.Categorical(probs=probabilities), edges


@dataclass
class Rollout:
    spins: np.ndarray
    energy: float
    log_probability: torch.Tensor
    entropy: torch.Tensor
    trace: list


def rollout(policy, problem, greedy=False, epsilon=0.0):
    env = Compression(problem)
    logs, entropies = [], []
    while not env.done:
        env.finish_free_nodes()
        if env.done:
            break
        distribution, edges = policy.distribution(env, epsilon)
        action = distribution.probs.argmax() if greedy else distribution.sample()
        logs.append(distribution.log_prob(action))
        entropies.append(distribution.entropy())
        edge, direction = divmod(int(action.item()), 2)
        u, v, _ = edges[edge]
        env.contract(u, v, 1 if direction == 0 else -1)
    spins, energy = env.solution()
    zero = next(policy.parameters()).sum() * 0
    return Rollout(spins, energy, torch.stack(logs).sum() if logs else zero,
                   torch.stack(entropies).mean() if entropies else zero, env.trace)


def load_policy(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    policy = Policy(checkpoint['width'], checkpoint['layers'])
    policy.load_state_dict(checkpoint['state_dict'])
    policy.eval()
    return policy, checkpoint
