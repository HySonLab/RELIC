import itertools
import networkx as nx
import numpy as np
import pytest
import torch
from relic.ising import Ising, Compression, exact_solve
from relic.data import random_ising, mis_ising, repair_independent_set
from relic.policy import Policy, rollout, graph_tensors
from relic.train import reinforce_objective


def assignments(n):
    return itertools.product([-1, 1], repeat=n)


@pytest.mark.parametrize('direction', [-1, 1])
def test_every_residual_assignment_preserves_restricted_energy(direction):
    p = random_ising(6, 'ER', 17, fields=True)
    p.offset = 3.25
    env = Compression(p)
    # Absorb auxiliary node 0 to exercise nontrivial gauge reconstruction.
    env.contract(1, 0, direction)
    u, v, _ = env.edges()[0]
    env.contract(u, v, -1)
    for values in assignments(len(env.adj)):
        residual = dict(zip(sorted(env.adj), values))
        reduced = env.offset + sum(w * residual[u] * residual[v] for u, v, w in env.edges())
        augmented = env.lift(residual)
        assert p.energy(augmented[1:] * augmented[0]) == pytest.approx(reduced)


@pytest.mark.parametrize('seed', range(10))
def test_complete_random_compression_and_reconstruction(seed):
    p = random_ising(8, 'ER', seed, fields=True)
    rng, env = np.random.default_rng(seed), Compression(p)
    while not env.done:
        env.finish_free_nodes()
        if env.done:
            break
        edges = env.edges()
        u, v, _ = edges[int(rng.integers(len(edges)))]
        env.contract(u, v, int(rng.choice([-1, 1])))
    s, e = env.solution()
    assert len(env.trace) == p.n
    assert e == pytest.approx(p.energy(s))
    assert e == pytest.approx(env.offset)


def test_cancelled_edges_disconnected_graph_and_empty_graph():
    J = np.array([[0,1,2,0],[1,0,-2,0],[2,-2,0,0],[0,0,0,0]], dtype=float)
    env = Compression(Ising(J, np.zeros(4)))
    env.contract(1,2,1)
    assert env.edges() == []
    env.finish_free_nodes()
    assert env.solution()[1] == 1
    p = Ising(np.zeros((3,3)), np.zeros(3), 2)
    assert rollout(Policy(8), p).energy == 2


def test_bad_alignment_can_exclude_ground_state():
    p = Ising([[0,1],[1,0]], [0,0])
    assert exact_solve(p)[0] == -1
    env = Compression(p); env.contract(1,2,1); env.finish_free_nodes()
    assert env.solution()[1] == 1


def test_maxcut_sign_convention():
    g = nx.cycle_graph(5)
    A = nx.to_numpy_array(g)
    for s in assignments(5):
        cut = sum(s[u] != s[v] for u,v in g.edges())
        assert Ising(A, np.zeros(5)).energy(s) == g.number_of_edges() - 2*cut


def test_mis_mapping_all_assignments_and_exact_optimum():
    g = nx.cycle_graph(7); p = mis_ising(g)
    for x in itertools.product([0,1], repeat=7):
        s = 2*np.array(x)-1
        q = -sum(x) + 2*sum(x[u]*x[v] for u,v in g.edges())
        assert p.energy(s) == pytest.approx(q)
    e,s = exact_solve(p)
    assert e == -3
    assert len(repair_independent_set(g, np.flatnonzero(s > 0))) == 3
    repaired = repair_independent_set(g, list(g))
    assert g.subgraph(repaired).number_of_edges() == 0


def test_policy_permutation_equivariance():
    torch.manual_seed(3)
    model = Policy(16)
    x, ij, e, _ = graph_tensors(Compression(random_ising(8, 'BA', 21, fields=True)))
    perm = torch.randperm(len(x)); inverse = torch.argsort(perm)
    expected = model(x, ij, e)
    # Also reverse each edge: endpoint order must not alter scores.
    actual = model(x[perm], inverse[ij.flip(1)], e)
    torch.testing.assert_close(expected, actual, atol=2e-6, rtol=2e-6)


def test_policy_gradient_matches_enumerated_expected_return():
    # One-edge Ising has exactly two actions; compare analytic expectation and
    # score-function gradient under the SAME epsilon mixture distribution.
    torch.manual_seed(4)
    model = Policy(8)
    p = Ising([[0,2],[2,0]], [0,0])
    dist,_ = model.distribution(Compression(p), epsilon=0.2)
    rewards = torch.tensor([-2.,2.])
    expected = (dist.probs * rewards).sum()
    params = tuple(model.parameters())
    g_exact = torch.autograd.grad(expected, params, retain_graph=True)
    score_objective = (dist.probs.detach() * rewards * dist.probs.log()).sum()
    g_score = torch.autograd.grad(score_objective, params)
    for a,b in zip(g_exact,g_score):
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    assert sum(g.abs().sum().item() for g in g_exact) > 0


def test_reinforce_sign_and_leave_one_out_baseline():
    logs = torch.tensor([-1.,-2.], requires_grad=True)
    loss, advantage = reinforce_objective(logs, [3.,1.], greedy_reward=2.)
    torch.testing.assert_close(advantage, torch.tensor([1.5,-1.5]))
    loss.backward()
    assert logs.grad[0] < 0 and logs.grad[1] > 0


def test_actual_rollout_backpropagates_and_updates_gnn():
    torch.manual_seed(8)
    model=Policy(8)
    p=random_ising(6,'ER',12)
    before=model.blocks[0].message[0].weight.detach().clone()
    r=rollout(model,p,epsilon=0.2)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
    loss=r.energy*r.log_probability
    loss.backward(); optimizer.step()
    assert not torch.equal(before, model.blocks[0].message[0].weight)
    assert all(torch.isfinite(v).all() for v in model.parameters())


def test_checkpoint_roundtrip_and_deterministic_rollout(tmp_path):
    from relic.policy import load_policy
    torch.manual_seed(9)
    model=Policy(8)
    p=random_ising(7,'WS',2026,fields=True)
    checkpoint=tmp_path/'model.pt'
    torch.save({'width':8,'layers':3,'state_dict':model.state_dict()},checkpoint)
    loaded,_=load_policy(checkpoint)
    with torch.no_grad():
        original=rollout(model,p,greedy=True)
        recovered=rollout(loaded,p,greedy=True)
    assert original.trace==recovered.trace
    assert original.energy==recovered.energy
    np.testing.assert_array_equal(original.spins,recovered.spins)


def test_training_smoke_saves_best_and_refuses_overwrite(tmp_path):
    from relic.train import train
    config={'seed':0,'data_seed':6000000,'nodes':6,'train_graphs':3,'validation_graphs':3,
            'width':8,'layers':3,'epochs':1,'batch_size':3,'rollouts':2,'lr':0.001,
            'epsilon_start':0.2,'epsilon_decay_epochs':1,'gradient_clip':1.,'patience':2,'threads':1}
    summary=train(config,tmp_path/'run')
    assert np.isfinite(summary['best_validation_normalized_energy'])
    assert (tmp_path/'run'/'best.pt').exists()
    assert (tmp_path/'run'/'history.csv').exists()
    with pytest.raises(FileExistsError):
        train(config,tmp_path/'run')
