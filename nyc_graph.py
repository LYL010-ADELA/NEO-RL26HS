# Freeze a street graph to disk once, so every run uses the same graph.
# The original notebook downloads it live from OpenStreetMap on every run, and OSM data changes over time.
#
# python nyc_graph.py freeze                 -> download NYC (same settings as the paper code) to data/nyc/
# python nyc_graph.py stats                  -> print graph statistics of data/nyc/
import os
import sys
import random
import datetime
import argparse
import numpy as np
import networkx as nx
import osmnx as ox
from all_functions import renumber_graph

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def freeze(name, center_point, initial_radius, desired_nodes=10000):
    from Many_Streets_Eval import get_street_graph
    G = get_street_graph(center_point=center_point, initial_radius=initial_radius, desired_nodes=desired_nodes)
    osmids = list(G.nodes())
    G = renumber_graph(G)  # same relabelling as run_training_on_env, fixed here once
    out = os.path.join(DATA_DIR, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'nodes.csv'), 'w') as f:
        f.write('id,osmid,x,y\n')
        for n, osmid in zip(G.nodes(), osmids):
            f.write(f"{n},{osmid},{G.nodes[n]['x']!r},{G.nodes[n]['y']!r}\n")
    # edges in adjacency order, so that G.neighbors(n) (= the action order) is the same after loading
    with open(os.path.join(out, 'edges.csv'), 'w') as f:
        f.write('u,v\n')
        for u, v in G.edges():
            f.write(f'{u},{v}\n')
    with open(os.path.join(out, 'meta.yaml'), 'w') as f:
        f.write(f'downloaded: {datetime.datetime.now().isoformat(timespec="seconds")}\n')
        f.write(f'osmnx: {ox.__version__}\nnetworkx: {nx.__version__}\n')
        f.write(f'center_point: [{center_point[0]}, {center_point[1]}]\ninitial_radius_m: {initial_radius}\n')
        f.write(f'desired_nodes: {desired_nodes}\nnetwork_type: drive\nsimplify: true\n')
        f.write(f'nodes: {G.number_of_nodes()}\nedges: {G.number_of_edges()}\n')
    loaded = load_graph(name)
    assert all(list(loaded.neighbors(n)) == list(G.neighbors(n)) for n in G.nodes()), 'round trip changed action order'
    print(f'saved {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges -> {out}')
    return G


def load_graph(name='nyc'):
    """Rebuild the frozen graph exactly as get_street_graph + renumber_graph return it (MultiDiGraph with 'pos')."""
    d = os.path.join(DATA_DIR, name)
    G = nx.MultiDiGraph()
    with open(os.path.join(d, 'nodes.csv')) as f:
        next(f)
        for line in f:
            n, osmid, x, y = line.strip().split(',')
            G.add_node(int(n), osmid=int(osmid), x=float(x), y=float(y), pos=(float(x), float(y)))
    with open(os.path.join(d, 'edges.csv')) as f:
        next(f)
        for line in f:
            u, v = line.strip().split(',')
            G.add_edge(int(u), int(v))
    return G


def graph_stats(G, seeds=range(20)):
    from Many_Streets_Eval import get_start_and_goal_state
    D = nx.DiGraph(G)
    sccs = sorted(nx.strongly_connected_components(D), key=len, reverse=True)
    largest = sccs[0]
    out_deg = np.array([D.out_degree(n) for n in D.nodes()])
    in_deg = np.array([D.in_degree(n) for n in D.nodes()])
    # nodes from which the largest SCC can be reached / that can be reached from it
    rev_reach = nx.ancestors(D, next(iter(largest))) | {next(iter(largest))}
    fwd_reach = nx.descendants(D, next(iter(largest))) | {next(iter(largest))}
    s = {
        'nodes': D.number_of_nodes(),
        'edges (MultiDiGraph, as counted by the code)': G.number_of_edges(),
        'distinct directed edges': D.number_of_edges(),
        'undirected edges': nx.Graph(D).number_of_edges(),
        'two-way street pairs': sum(1 for u, v in D.edges() if u < v and D.has_edge(v, u)),
        'weakly connected components': nx.number_weakly_connected_components(D),
        'strongly connected components': len(sccs),
        'largest SCC size': len(largest),
        'largest SCC share': round(len(largest) / D.number_of_nodes(), 4),
        'SCCs of size 1': sum(1 for c in sccs if len(c) == 1),
        'dead ends (out-degree 0)': int((out_deg == 0).sum()),
        'unreachable entries (in-degree 0)': int((in_deg == 0).sum()),
        'nodes that can reach the largest SCC': len(rev_reach),
        'nodes reachable from the largest SCC': len(fwd_reach),
        'max out-degree (action space size)': int(out_deg.max()),
        'mean out-degree': round(float(out_deg.mean()), 3),
    }
    # start/goal pairs used by the experiment code (seed = run index)
    pairs = []
    for run in seeds:
        start, goal = get_start_and_goal_state(run, None, G)
        hops = nx.shortest_path_length(D, start, goal)
        back = nx.has_path(D, goal, start)
        pairs.append((run, start, goal, hops, start in largest, goal in largest, back, D.out_degree(goal)))
    return s, pairs


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['freeze', 'stats'])
    p.add_argument('--name', default='nyc')
    p.add_argument('--center', type=float, nargs=2, default=(40.7128, -74.0060))  # NYC City Hall, as in the paper code
    p.add_argument('--radius', type=int, default=7000)
    a = p.parse_args()
    if a.cmd == 'freeze':
        freeze(a.name, tuple(a.center), a.radius)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):  # get_start_and_goal_state prints a lot
        s, pairs = graph_stats(load_graph(a.name))
    for k, v in s.items():
        print(f'{k:48s} {v}')
    print('\nseed  start  goal  hops  start_in_SCC  goal_in_SCC  goal->start_path  goal_out_deg')
    for r in pairs:
        print('  '.join(str(x) for x in r))
