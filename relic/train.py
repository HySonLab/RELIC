"""REINFORCE training with terminal energy and action-independent baselines."""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from .data import dataset
from .policy import Policy, rollout


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def reinforce_objective(log_probabilities, rewards, greedy_reward, scale=1.0):
    """Average greedy + leave-one-out baseline; detach all returns.

    Including a trajectory in its own stochastic baseline biases its gradient.
    At least two independent trajectories per graph avoid that issue.
    """
    rewards = torch.as_tensor(rewards, dtype=log_probabilities.dtype, device=log_probabilities.device).detach()
    if len(rewards) < 2:
        raise ValueError('At least two rollouts are required for the leave-one-out baseline')
    other_mean = (rewards.sum() - rewards) / (len(rewards) - 1)
    baseline = 0.5 * (other_mean + float(greedy_reward))
    advantage = (rewards - baseline).detach()
    return -(advantage / scale * log_probabilities).mean(), advantage


@torch.no_grad()
def validation_score(policy, problems):
    energies = [rollout(policy, p, greedy=True).energy / p.scale() for p in problems]
    return float(np.mean(energies))


def train(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'history.csv').exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {output}')
    seed_everything(config['seed'], config.get('threads', 1))
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    # Same train/validation instances across policy seeds; test uses separate ranges.
    training = dataset(config['train_graphs'], config['nodes'], config['data_seed'])
    validation = dataset(config['validation_graphs'], config['nodes'], config['data_seed'] + 100000)
    policy = Policy(config['width'], config['layers'])
    optimizer = torch.optim.Adam(policy.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    start = time.perf_counter()
    initial = validation_score(policy, validation)
    best_score, best_epoch, stale = initial, 0, 0

    def save(name, epoch, score):
        torch.save({'state_dict': policy.state_dict(), 'width': policy.width, 'layers': policy.layers,
                    'epoch': epoch, 'validation_normalized_energy': score, 'config': config}, output / name)

    save('initial.pt', 0, initial)
    save('best.pt', 0, initial)
    rms = 1.0
    rows = [{'epoch': 0, 'train_normalized_energy': '', 'validation_normalized_energy': initial,
             'loss': '', 'gradient_norm': '', 'epsilon': '', 'lr': config['lr'], 'elapsed_seconds': time.perf_counter()-start}]
    rng = np.random.default_rng(config['seed'])
    for epoch in range(1, config['epochs'] + 1):
        policy.train()
        epsilon = config['epsilon_start'] * max(0, 1 - (epoch - 1) / config['epsilon_decay_epochs'])
        losses, energies, norms = [], [], []
        order = rng.permutation(len(training))
        for start_idx in range(0, len(training), config['batch_size']):
            indices = order[start_idx:start_idx + config['batch_size']]
            optimizer.zero_grad()
            square_advantages = []
            # Backpropagate graph by graph to bound memory without changing batch loss.
            for idx in indices:
                problem = training[idx]
                with torch.no_grad():
                    greedy_reward = -rollout(policy, problem, greedy=True).energy / problem.scale()
                samples = [rollout(policy, problem, epsilon=epsilon) for _ in range(config['rollouts'])]
                rewards = [-sample.energy / problem.scale() for sample in samples]
                logs = torch.stack([sample.log_probability for sample in samples])
                # Previous-batch RMS is action-independent for the current batch.
                loss, advantage = reinforce_objective(logs, rewards, greedy_reward, max(math.sqrt(rms), 0.05))
                (loss / len(indices)).backward()
                losses.append(float(loss.detach()))
                energies.extend([-r for r in rewards])
                square_advantages.extend(advantage.square().tolist())
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            norms.append(float(norm))
            optimizer.step()
            rms = 0.9 * rms + 0.1 * float(np.mean(square_advantages))
        policy.eval()
        score = validation_score(policy, validation)
        scheduler.step(score)
        if score < best_score - 1e-8:
            best_score, best_epoch, stale = score, epoch, 0
            save('best.pt', epoch, score)
        else:
            stale += 1
        rows.append({'epoch': epoch, 'train_normalized_energy': float(np.mean(energies)),
                     'validation_normalized_energy': score, 'loss': float(np.mean(losses)),
                     'gradient_norm': float(np.mean(norms)), 'epsilon': epsilon,
                     'lr': optimizer.param_groups[0]['lr'], 'elapsed_seconds': time.perf_counter()-start})
        with (output / 'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
        print(json.dumps({'seed': config['seed'], **rows[-1]}), flush=True)
        if stale >= config['patience']:
            break
    save('last.pt', epoch, score)
    summary = {'initial_validation_normalized_energy': initial, 'best_validation_normalized_energy': best_score,
               'best_epoch': best_epoch, 'epochs_run': epoch, 'seconds': time.perf_counter()-start}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/desktop.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.seed is not None:
        config['seed'] = args.seed
    train(config, args.output)


if __name__ == '__main__':
    main()
