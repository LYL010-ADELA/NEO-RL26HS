# One (method, seed) run on the frozen NYC graph (C2 / C3 of REPRODUCTION_PLAN.md).
# One process per run, so seeds can run in parallel and failed runs can be restarted.
#
# python run_nyc.py --method neo --seed 0
# python run_nyc.py --method neo1 --seed 3 --graph nyc
import os
import sys
import time
import random
import pickle
import argparse
import platform
import numpy as np

# method -> (option_name in the code, options per update)
METHODS = {
    'neo':   ('hotspot_options', 4),    # NEO
    'neo1':  ('hotspot_options', 1),    # NEO (first eigenvector only)
    'sp':    ('oracle', 4),             # SP-novelty; the NYC notebook uses 4 per update, plots.ipynb lists 1
    'qlnov': ('Qlearning-novelty', 4),  # Q-learning + 0.01/sqrt(n(s,a)); no options
    'eigen': ('eigen_options', 4),      # eigen-options (polar decomposition in the directed case)
}


class Timestamped:
    """Prefix every printed line with seconds since start, so logs give wall-clock per step count."""
    def __init__(self, stream):
        self.stream, self.t0, self.bol = stream, time.time(), True

    def write(self, s):
        for part in s.splitlines(keepends=True):
            if self.bol:
                self.stream.write(f'[{time.time() - self.t0:9.1f}s] ')
            self.stream.write(part)
            self.bol = part.endswith('\n')

    def flush(self):
        self.stream.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--num-options', type=int, default=None, help='override options per update')
    p.add_argument('--graph', default='nyc', help='folder in data/ written by nyc_graph.py')
    p.add_argument('--out', default=os.path.join('results', 'nyc'))
    a = p.parse_args()

    sys.stdout = Timestamped(sys.stdout)
    option_name, num_options = METHODS[a.method]
    if a.num_options is not None:
        num_options = a.num_options
    # the original code seeds only `random` (start/goal); numpy drives epsilon-greedy, options and env,
    # so we seed it too (deviation, see DEVIATIONS.md). One seed per process = seeded at the start of the run.
    np.random.seed(a.seed)
    random.seed(a.seed)

    from nyc_graph import load_graph
    from Many_Streets_Eval import run_training_on_env
    G = load_graph(a.graph)
    t0 = time.time()
    training, evals, nodes, edges, _agent, visitation, info = run_training_on_env(
        G, 'street', option_name=option_name, num_options=num_options, directed=False,  # forced to True inside
        option_eps=0.1, det_option_policy=True, reward_eval=True, RW_LAP=True, seeds=[a.seed])
    wall = time.time() - t0

    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f'{a.graph}_{a.method}_k{num_options}_seed{a.seed}.pkl')
    # the final agent is not saved: every option keeps a deep copy of the agent's graph (hundreds of MB)
    with open(out, 'wb') as f:
        pickle.dump({
            'method': a.method, 'option_name': option_name, 'num_options': num_options, 'seed': a.seed,
            'graph': a.graph, 'eval_interval': 2000,
            'eval_returns': evals[0], 'node_counts': nodes[0], 'edge_counts': edges[0],
            'training_returns': training[0], 'visitation': visitation, 'info': info[0],
            'wall_clock_s': wall, 'python': platform.python_version(), 'numpy': np.__version__,
        }, f)
    print(f'saved {out} ({wall / 60:.1f} min)')


if __name__ == '__main__':
    main()
