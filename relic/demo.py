"""Print every contraction for a tiny biased Ising instance."""
import argparse
import numpy as np
import torch
from .ising import Ising, exact_solve
from .policy import load_policy, rollout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='results/seed0/best.pt')
    args=parser.parse_args()
    torch.set_num_threads(1)
    p=Ising([[0,2,-1],[2,0,1],[-1,1,0]],[0.4,-0.6,0.2],offset=1.5)
    model,_=load_policy(args.checkpoint)
    with torch.no_grad():
        r=rollout(model,p,greedy=True)
    print('Augmented node 0 is the auxiliary spin; nodes 1..3 are physical spins.')
    for step,(u,v,d) in enumerate(r.trace,1):
        print(f'{step}: absorb node {v} into {u}, require s[{v}] = {d:+d} * s[{u}]')
    print('Physical spins:',r.spins.tolist())
    print('Reconstructed energy:',r.energy)
    print('Exact optimum:',exact_solve(p)[0])


if __name__=='__main__':
    main()
