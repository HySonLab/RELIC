# RELIC: Reinforcement Learning Based Ising Optimization via Graph Compression

A student-friendly, independent implementation of **RELIC**, based on the six-page paper by Duy Do Le, Truong-Son Hy, and Thang N. Dinh, IEEE QCE 2025, pp. 193–198. [Paper DOI](https://doi.org/10.1109/QCE65121.2025.10318) · [Author-hosted paper](https://thang-dinh.github.io/files/QCE25_RELIC_2025.pdf)

**Scope:** this repository reproduces the central algorithm—signed graph compression, a joint node/edge GNN, and terminal-reward REINFORCE—and runs a small, auditable CPU study. It is newly written code, not recovered author code. It does **not** reproduce the paper's numerical tables, original training dataset, 4,000-node experiments, or GPU speedup claims. Unspecified details and mathematical ambiguities are explained below.

Start with the [measured results](results/evaluation/REPORT.md) and the [1,000-node scaling pilot](results/scaling/REPORT.md). The repository includes three trained seeds, initial and final checkpoints, training logs, test graphs, raw predictions, CSV measurements, plots, and an independent result-verification script.

## 1. What RELIC learns

An Ising problem assigns one spin, either −1 or +1, to each vertex. Neighboring spins contribute energy through their edge weights. Our convention is

```text
E(s) = sum over i<j of J[i,j] * s[i] * s[j]
       + sum over i of h[i] * s[i] + offset.
```

We **minimize** this energy. A positive edge prefers opposite spins; a negative edge prefers equal spins. `J` is symmetric and has a zero diagonal. The factor `1/2` in the implementation's `s @ J @ s` prevents counting each edge twice.

RELIC does not predict all spins at once. It repeatedly chooses an edge and one of two decisions:

- **Merge:** require `s[v] = s[u]` and absorb vertex `v` into `u`.
- **Flip-merge:** require `s[v] = -s[u]` and absorb vertex `v` into `u`.

The GNN decides which edge and relationship to choose. After compression, reversing the recorded decisions recovers the original spin assignment. During training, only the final energy supplies a reward. There are no optimal-spin labels, intermediate energy rewards, or exact-solver calls in training.

### A two-spin example

Suppose the only edge has weight `J[0,1] = +2` and there are no fields. Equal spins give energy `+2`; opposite spins give `−2`. Flip-merging this edge adds `−2` to the offset. Reversing that decision recovers opposite spins and therefore the same energy.

An incorrect merge would give `+2`. **Compression preserves energies under the chosen spin relationship; an arbitrary relationship does not preserve the global optimum.** Learning good relationships is the optimization problem.

## 2. Install and run

Python 3.10 or newer is required. CPU execution is sufficient; PyTorch Geometric and a GPU are not required.

```bash
git clone https://github.com/HySonLab/RELIC.git
cd RELIC
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU-only PyTorch; omit this line if you already have suitable PyTorch.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m pytest -q
python -m relic.demo
```

The demo prints the complete contraction trace, reconstructed spins, energy, and exact optimum for a tiny problem. It loads the included `results/seed0/best.pt` checkpoint.

The actual execution environment is recorded in [environment.json](results/environment.json). Exact package versions are in [requirements-tested.txt](requirements-tested.txt); the portable minimum requirements are in `pyproject.toml`. Installing exact versions is optional and depends on wheel availability for your platform. For strict comparisons, use the recorded versions and one CPU thread.

### Solve your own Ising problem

```python
import numpy as np
import torch
from relic.ising import Ising
from relic.policy import load_policy, rollout

problem = Ising(
    J=np.array([[0.0, 2.0], [2.0, 0.0]]),
    h=np.array([0.3, -0.1]),
)
policy, metadata = load_policy("results/seed0/best.pt")
with torch.no_grad():
    solution = rollout(policy, problem, greedy=True)
print(solution.spins, solution.energy)
assert np.isclose(problem.energy(solution.spins), solution.energy)
```

The public model is CPU-oriented, and the included checkpoints use `weights_only=True` loading. Physical spin indices start at 0 in `Ising`; contraction traces use the augmented numbering described above.

### Recompute the saved result checks

```bash
python -m relic.verify_results
python -m relic.report
```

Verification reads the saved instances and spin assignments, recomputes energies and conflicts, and reruns each distinct small exact oracle. It does not trust an experiment's reported objective without checking it.

### Train a quick smoke model

```bash
python -m relic.train --config configs/smoke.json --output runs/smoke
```

This is a pipeline check with 8-node graphs and two epochs, not a quality experiment.

### Repeat the desktop study

```bash
bash scripts/reproduce.sh runs/my-desktop-study
```

The script runs tests, trains seeds 0/1/2, evaluates them, verifies results, and produces plots and a report. It writes to a **new** destination, leaving the committed `results/` intact. Existing training histories are not overwritten: choose a new directory for another run. The script runs seeds sequentially; the published training runs used separate concurrent processes.

Individual commands are also available:

```bash
python -m relic.train --config configs/desktop.json --seed 0 --output runs/example/seed0
# Train seeds 1 and 2 into runs/example/seed1 and runs/example/seed2 before evaluation.
python -m relic.evaluate --results runs/example
python -m relic.verify_results --results runs/example/evaluation
python -m relic.report --results runs/example
```

### Repeat the size-generalization pilot

```bash
python -m relic.scaling --results results
python -m relic.verify_results
```

This optional pilot evaluates the same trained models on one new ER, BA and WS graph at each of 100, 250, 500 and 1,000 nodes. It saves every spin assignment and instance. There are no exact optima or MIS instances in this supplementary pilot.

### Increase the budget

```bash
python -m relic.train --config configs/paper_scale.json --seed 0 --output runs/paper-scale/seed0
```

This config requests 5,000 training graphs, 25 nodes, up to 100 epochs, and exploration decay over 30 epochs, matching the paper's stated counts. It is **not an exact paper configuration**: hidden width, optimizer details, graph-generator parameters, validation size, and other missing choices remain ours. It was not run for the published desktop results. There is no automatic checkpoint resume; checkpoints are inference snapshots, not full optimizer/RNG snapshots.

## 3. How the code is organized

| File | Purpose |
|---|---|
| `relic/ising.py` | Ising energies, exact enumeration, compression, reconstruction |
| `relic/data.py` | Seeded ER/BA/WS graphs; MIS mapping and feasibility repair |
| `relic/policy.py` | Joint node/edge GNN, global action softmax, policy rollouts |
| `relic/train.py` | REINFORCE, baselines, clipping, scheduler, validation selection |
| `relic/baselines.py` | Random/greedy contraction, local search, PI-GNN-style control |
| `relic/evaluate.py` | Held-out experiments, raw solutions and runtime measurements |
| `relic/verify_results.py` | Independent checks of saved results and exact optima |
| `relic/report.py` | Descriptive statistics, figures and result tables |
| `relic/scaling.py` | Supplementary inference pilot up to 1,000 nodes |
| `relic/demo.py` | Tiny worked example with printed decisions |
| `tests/test_core.py` | Algebra, reconstruction, symmetry and gradient checks |
| `configs/` | Smoke, desktop and larger training budgets |
| `results/` | Actual measured runs and artifacts |

## 4. Compression mathematics

To enforce `s[v] = d*s[u]`, where `d` is +1 or −1, substitute that identity into the Hamiltonian:

```text
offset_new = offset_old + d * J[u,v]
J_new[u,w] = J_old[u,w] + d * J_old[v,w]
```

Then delete `v`. An absent edge has weight zero. Exact cancellation removes an edge. All original coefficients and compression calculations use NumPy float64; neural computations use float32.

### Local fields and the auxiliary vertex

Add an auxiliary spin `z[0]`, and represent field `h[i]` by an edge from node `0` to physical node `i+1`. Physical spins are `s[i] = z[i+1]*z[0]`. Pair products are unchanged by a global sign flip, so this gauge choice exactly recovers the original field problem.

We allow the auxiliary node to be absorbed. Reconstruction first lifts every augmented spin, then multiplies physical spins by the reconstructed auxiliary spin. Simply assigning the last node +1 without this gauge correction would be wrong for some biased problems.

### Disconnected and edgeless graphs

The paper specifies edge actions but does not say what to do when no nonzero edges remain before reaching one vertex. This can happen initially, through disconnected components, or through weight cancellation. Our implementation processes available edges, then deterministically joins remaining zero-interaction vertices with `d=+1`. These joins contribute no reward or policy log probability. They do not change energy because the residual graph has no interactions. With `n` physical spins and one auxiliary node, the trace always contains `n` contractions; the number of learned decisions can be smaller.

### The terminal reward and the offset

We compute the terminal reward as **minus the original reconstructed energy**, once. The final offset equals that energy after complete contraction, and every rollout checks this equality.

Section III-A writes a reward containing both the accumulated offset and `E_final`, while describing `E_final` as the reconstructed energy. Adding the offset to a full reconstructed energy would count it twice. Our interpretation uses `E_original(s)` or equivalently the final offset, never their sum. A baseline is applied separately in the policy-gradient loss.

## 5. The GNN

The input features follow Section III-B:

```text
node: [degree, inverse_degree, sum of absolute incident weights]
edge: [weight, absolute_weight, sign(weight)]
```

The inverse degree is zero for isolated vertices. Input projections map three features to 32 hidden channels. Three message-passing layers jointly update nodes and edges:

1. Build a message from each neighboring node and its incident edge embedding.
2. Sum incoming messages; update the node with an MLP.
3. Update each edge using its previous embedding and the previous node embeddings.

For an undirected edge, we use endpoint sum and absolute difference. This makes edge scores invariant to swapping endpoints, and equivariant to node relabeling before tie-breaking. MLPs use ReLU, and hidden updates use LayerNorm. These details are implementation choices, not specified hyperparameters from the paper. Because the final edge update uses the *previous* node embeddings, a final node update cannot influence the edge-only output; that unused computation is omitted.

Concatenate the final edge embedding with the three raw edge features. A readout MLP emits two scores per edge: merge and flip-merge. **One softmax covers all `2*number_of_edges` actions**, rather than a separate softmax for each edge. The network learns both which edge to compress and which relationship to impose.

No built-in rule forces the GNN to satisfy the strongest edge. `greedy_contraction` is a separate classical baseline, not a hidden part of RELIC.

## 6. Reinforcement learning

For a sampled trajectory, add the log probabilities of all learned actions. Minimize

```text
loss = -(terminal_return - baseline) * sum_of_action_log_probabilities
```

A higher-than-baseline return increases the probabilities of its sampled actions. The graph mutations are discrete and are not differentiated; gradients flow through the sampled actions' log probabilities.

Our implementation makes the following choices explicit:

- **Terminal return:** `-energy / instance_scale`, where `instance_scale = max(sum |J_edges| + sum |h|, 1)`. This scale is fixed for the initial problem and independent of the sampled trajectory. It balances instances while preserving the optimum for each instance.
- **Greedy rollout baseline:** run the current network deterministically on the same input graph, without gradients.
- **Stochastic baseline:** average the returns of the *other* independent rollouts of that same graph. Excluding the current rollout avoids the self-baseline bias.
- **Combined baseline:** average the greedy and leave-one-out baselines. All baseline values and advantages are detached from autograd.
- **Exploration:** sample from `(1-epsilon)*softmax(logits) + epsilon/number_of_actions`. The log probability uses that exact mixture. This is epsilon-uniform exploration of a stochastic policy, an explicit interpretation of the paper's underspecified “epsilon-greedy” language. Literal argmax epsilon-greedy is not used for training.
- **Normalization:** divide advantages by a running RMS estimated from *previous* batches, with a floor of 0.05. This avoids making the normalization depend on the current sampled actions. It differs from literal current-minibatch normalization.
- **Optimization:** Adam; clip the global gradient norm at 1; reduce learning rate on validation plateaus; retain the best greedy validation checkpoint; early-stop after a configured number of non-improving epochs.

The desktop configuration uses 64 training graphs, 18 validation graphs, 25 physical nodes, two stochastic rollouts per training graph, batch size 8, learning rate 0.001, and up to 30 epochs. Seeds 0, 1 and 2 initialize independent models and sampling streams. They share training/validation graph sets so seed variation is easier to interpret. Training graphs have zero fields and weights uniform in [−5,5].

## 7. Data and evaluation

All datasets are generated locally with NetworkX. Original GRANITE data were not supplied with the paper and are not claimed to be reconstructed.

| Family | Chosen parameters |
|---|---|
| ER | edge probability `4/(n-1)` |
| BA | attach each new node to 2 existing nodes |
| WS | ring degree 4; rewiring probability 0.3 |
| MIS | random regular graphs, degrees 3, 5, 7 and 9 |

Graph seeds, model seeds, and evaluation budgets are in code/configs and saved artifacts. Training graph seeds start at 1,000; validation at 101,000; Ising tests use a separate range above 2,000,000; MIS tests use a separate range above 3,000,000. Checkpoints are selected only on validation, never on test graphs.

The desktop evaluation contains:

- Ising: 12, 25, 50 and 100 spins; ER, BA and WS; three instances per configuration.
- MIS: 16, 50 and 100 vertices; four regular degrees; three instances per configuration.
- Three model seeds for every graph; deterministic baselines repeat across seeds for a paired table layout, not independent replication.
- Exact exhaustive search for all 12-spin Ising and 16-node MIS test instances.
- Greedy RELIC inference and best-of-eight stochastic trajectories, reported separately.

### Baselines

`random_compression` chooses a random edge and direction. `greedy_contraction` chooses the largest-magnitude residual edge and satisfies its preferred relationship. `local_search_8` uses eight random restarts and best improving single-spin flips. `greedy_mis` repeatedly selects a minimum residual degree vertex and removes its neighbors. These inexpensive methods are important controls: learning need not outperform them.

`pignn_reference` is a small independent physics-inspired GNN control: trainable random node embeddings, two normalized graph-convolution layers, a sigmoid output, and the relaxed MIS QUBO loss. It trains separately on each test instance for 300 Adam steps at learning rate 0.01 and keeps the best thresholded candidate encountered. There is no dropout or hyperparameter search. **It is not the original PI-GNN implementation or a validated replication of the paper's baseline**, and these experiments cannot establish superiority over PI-GNN generally.

### MIS encoding and repair

For `x[i]` in {0,1}, minimize

```text
Q(x) = -sum x[i] + P * sum_edges x[i]*x[j], with P=2.
x = (s+1)/2
J[i,j] = P/4 on graph edges
h[i] = -1/2 + P*degree[i]/4
offset = -n/2 + P*number_of_edges/4
```

For `P>1`, removing a selected conflict vertex strictly improves the objective, so a global minimizer is independent. An approximate solution may still violate constraints. We report its raw cardinality, number of conflicting edges, and validity. The common repair procedure only deletes conflict vertices until independence holds; it does **not** greedily augment the set. Repaired size is reported separately. A zero-sized set is valid but useless.

Best-of-eight selection uses raw Ising energy, before repair, rather than choosing the largest repaired independent set. The same deletion repair is applied to all MIS solvers.

### Timing and statistics

Single-thread CPU timings include Python graph handling and inference/optimization. RELIC model loading and offline training are excluded from inference timing; training duration is reported separately. PI-GNN timing includes its per-instance training. Repair time is separately recorded and included in MIS total time. Best-of-eight consumes eight trajectories, so its time must not be compared as though it were a single greedy solve.

Means and one-standard-error bars first average model seeds within each graph, then use graphs as sampling units. These are descriptive results on a small synthetic sample, not hypothesis tests or 95% confidence intervals. Exact raw CSVs are provided for alternative analyses.

## 8. What is faithful, and what is a reconstruction choice?

| Component | Status |
|---|---|
| Ising Hamiltonian and auxiliary-node field representation | Implemented and algebraically tested |
| Merge and flip-merge, accumulated offset, reconstruction | Implemented with explicit gauge handling |
| Three message-passing layers, node/edge features, two actions per edge | Matches the specified structure |
| Terminal-only REINFORCE and rollout baselines | Implemented; baseline combination specified here |
| 25-node mixed-topology training | Matches size and named families; fewer graphs locally |
| Hidden size, activations, normalization, optimizer hyperparameters | Explicit choices because the paper does not specify them |
| Original 5,000-graph dataset | Replaced by seeded synthetic generation |
| Epsilon-greedy and reward normalization details | Clarified stochastic interpretation and previous-batch RMS |
| Residual graphs with no edges | Deterministic zero-weight completion, absent from paper details |
| PI-GNN comparison | Educational reference only |
| Original large-graph quality and GPU runtime results | Not reproduced |

Additional issues students should notice when reading the paper:

1. Under Equation (1)'s **positive** `J*s_i*s_j` convention, unweighted MaxCut uses `J=+1`: `cut_size=(number_of_edges-E)/2`. Section II-A's `J=-1` example instead favors equal spins. A unit test fixes the convention used here.
2. Merge/flip-merge preserves a **restricted** energy landscape, not necessarily the original ground state. A two-spin counterexample is tested.
3. Figure 1 describes random regular MIS graphs, whereas surrounding text also mentions ER. This study follows the figure's regular-graph setup and labels it explicitly.
4. “Linear runtime” is an empirical claim, not a property guaranteed by this algorithm. A fresh GNN forward pass occurs after each contraction. Summing work over all residual graphs can be quadratic or worse in the initial size; Python overhead also matters.
5. Section III-A's offset/reconstructed-energy ambiguity is resolved as described above; the accumulated offset is never counted twice.

## 9. What the measured study found

The three seeds improved their greedy validation energy from initialization. On the 12-spin test cases, greedy RELIC reached an exact optimum in 22 of 27 model/graph runs (81.5%). Greedy contraction and eight-restart local search solved all nine distinct small instances optimally in this sample. At 100 spins, greedy RELIC's mean normalized energy was −0.7543, compared with −0.7615 for greedy contraction (lower is better).

On 100-node MIS graphs, greedy RELIC produced an average repaired independent set of 30.36 vertices, compared with 33.83 for minimum-degree greedy MIS. Its raw solutions were feasible in 33 of 36 runs; all repaired solutions were feasible. The untuned PI-GNN reference often collapsed to empty sets, so that comparison is not evidence against well-tuned PI-GNN methods.

Best-of-eight **stochastic** inference performed worse than deterministic greedy inference here. Its eight candidates do not include the greedy trajectory, so it has no guarantee of matching the greedy result. The sampled policy remains diffuse after the small training budget, and selecting a checkpoint by greedy validation does not optimize stochastic performance. This limitation is retained in the report rather than hidden by changing the decoder after inspecting test results.

The supplementary pilot successfully ran the same 25-node-trained models on graphs up to 1,000 nodes. At 1,000 nodes, mean greedy RELIC inference time was about 2.58 seconds, versus 0.21 seconds for greedy contraction. These CPU measurements support executable size transfer, not the paper's claimed GPU speedup or linear scaling.

## 10. Correctness checks and suggested exercises

The unit suite checks restricted-energy equality for **every residual spin assignment** of a small biased graph, complete random contraction traces, auxiliary-node absorption, exact cancellation, isolated vertices, MIS QUBO equivalence for every assignment on a small graph, the MaxCut sign, node-label equivariance, and policy gradients against an analytically enumerated expectation. A further test performs a real sampled rollout and verifies that a GNN weight updates.

Passing these checks validates implementation properties. It does not prove the learned policy finds global optima or matches the paper's performance.

Suggested student exercises:

1. Work through the two-spin example and then change the coupling sign.
2. Derive the MIS coefficients by substituting `x=(s+1)/2` into the QUBO.
3. Remove the gauge correction in a local copy and identify which test fails.
4. Compare initial and trained policies on the **same** held-out graphs.
5. Compare greedy RELIC with best-of-eight, including the extra computation.
6. Change graph density and study distribution shift without selecting models on the test set.
7. Investigate whether more training improves performance beyond greedy contraction or local search.
8. Add simulated annealing or an established PI-GNN implementation and document equal-budget comparisons.

## Citation and provenance

```bibtex
@inproceedings{le2025relic,
  title={RELIC: Reinforcement Learning Based Ising Optimization via Graph Compression},
  author={Le, Duy Do and Hy, Truong-Son and Dinh, Thang N.},
  booktitle={2025 IEEE International Conference on Quantum Computing and Engineering (QCE)},
  year={2025},
  pages={193--198},
  doi={10.1109/QCE65121.2025.10318}
}
```

The implementation was prepared from the user-provided published PDF. Its SHA-256 is recorded in `results/environment.json`. The PDF itself is not redistributed. Supporting references: [GRANITE](https://arxiv.org/abs/2412.18571), [physics-inspired GNN optimization](https://doi.org/10.1038/s42256-022-00465-9), and [Lucas's Ising formulations](https://doi.org/10.3389/fphy.2014.00005). See the paper for the complete bibliography.
