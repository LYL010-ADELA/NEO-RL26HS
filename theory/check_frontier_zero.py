# Does NEO's eigenvector vanish on the never-visited ("source", zeta = 0) nodes of the agent's graph?
# A node that is only ever a next state has no observed out-edge, so its row of P is zero, and
# row i of L f = lambda f reads (1 + zeta_i) f_i = lambda f_i  ->  f_i = 0 unless lambda = 1 + zeta_i.
#
# python theory/check_frontier_zero.py --seed 0      (run from the repo root)
import os
import sys
import random
import argparse
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import all_functions


class Captured(Exception):
    pass


def capture_first_update(seed, method='hotspot_options', num_options=4):
    """Run the NYC experiment until the first option update and return the agent's graph and visit counts there."""
    from nyc_graph import load_graph
    from Many_Streets_Eval import run_training_on_env
    seen = {}

    def add_option(self, directed, new_G, visitation_mat, *args, **kwargs):
        seen['G'], seen['N'] = new_G.copy(), np.array(visitation_mat, float)
        raise Captured

    all_functions.AgentQ2.add_option = add_option
    np.random.seed(seed)
    random.seed(seed)
    try:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            run_training_on_env(load_graph('nyc'), 'street', option_name=method, num_options=num_options, seeds=[seed])
    except Captured:
        pass
    return seen['G'], seen['N']


def source_laplacian(G, N, delta=0.5, k=64):
    """Same construction as AgentQ2.add_option for NEO (directed, RW Laplacian, F(N) = delta (N/max)^(1/k))."""
    nodes = list(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=nodes).toarray().astype(float)
    out = A.sum(axis=1)
    P = np.divide(A, out[:, None], out=np.zeros_like(A), where=out[:, None] > 0)
    counts = N[nodes]
    zeta = delta * (counts / counts.max()) ** (1 / k)
    return nodes, counts, zeta, out, np.eye(len(nodes)) + np.diag(zeta) - P


def greedy_ascent(G, nodes, mag, start):
    """NEO option policy: move to the out-neighbour with the largest |f| while it is strictly larger."""
    idx = {n: i for i, n in enumerate(nodes)}
    path = [start]
    while True:
        nbrs = [v for v in G.successors(path[-1])]
        if not nbrs:
            return path
        best = max(nbrs, key=lambda v: mag[idx[v]])
        if mag[idx[best]] <= mag[idx[path[-1]]]:
            return path
        path.append(best)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    G, N = capture_first_update(a.seed)
    nodes, counts, zeta, out, L = source_laplacian(G, N)
    lam, f = eigs(csr_matrix(L), k=4, which='SR')
    o = np.argsort(lam)
    lam, f = lam[o], f[:, o]
    src = counts == 0
    dangling = out == 0
    print(f'agent graph at first update: {len(nodes)} nodes, {G.number_of_edges()} edges')
    print(f'never-visited (zeta = 0) nodes: {src.sum()}, dangling (no out-edge) nodes: {dangling.sum()}, '
          f'never-visited AND dangling: {(src & dangling).sum()}')
    for j in range(4):
        mag = np.abs(f[:, j]) / np.abs(f[:, j]).max()
        top = np.argmax(mag)
        start = nodes[int(np.argmax(counts))]
        path = greedy_ascent(G, nodes, mag, start)
        end = nodes.index(path[-1])
        print(f'\nlambda_{j + 1} = {lam[j].real:.4f}{lam[j].imag:+.1e}i')
        print(f'  max |f| on never-visited nodes: {mag[src].max() if src.any() else float("nan"):.2e}   (global max = 1)')
        print(f'  argmax |f|: count = {counts[top]:.0f}, zeta = {zeta[top]:.4f}, out-degree = {out[top]:.0f}')
        print(f'  ascent from most-visited node: {len(path) - 1} steps, ends at count = {counts[end]:.0f}, '
              f'zeta = {zeta[end]:.4f} (< Re lambda: {zeta[end] < lam[j].real})')
